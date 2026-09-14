import jax.numpy as jnp
from jax import jit, random, vmap, value_and_grad, debug
from jax_md import space, energy, smap, util
from jaxopt import LBFGS, ScipyMinimize, GradientDescent
from functools import partial


###### Basic functions ######

def get_R_model(global_options, shift, sep_mac, sep_src, bound=True, dist_ini=0.01):
    """ Construct coupled model of machine and source. It returns coordinates of 2*2 patch particles in different
    coupled states. Machine and source are modelled by Dumbbells in 3D with colloid diameter = 1.
    We hard-coded in that dimension=3.

    Args:
      - global_options:
      - shift:
      - sep_mac:
      - sep_src:

    Return:
      R_model: Array of positions with shape (4,3).
    """

    # Get patch-particle locations
    # initialize machine at center of periodic simulation box
    particle1_mac = global_options['side_length'] / 2 * jnp.array([1, 1, 1])
    particle2_mac = shift(particle1_mac, sep_mac * jnp.array([0, 1, 0]))

    # bound states
    if bound == True:
        particle1_src = shift(particle1_mac, dist_ini * jnp.array([1, 0, 0]))
        particle2_src = shift(particle1_src, sep_src * jnp.array([0, 1, 0]))
    # unbound states
    else:
        particle1_src = shift(particle1_mac, dist_ini * jnp.array([1, 0, 0]))
        particle2_src = shift(particle1_src, sep_src * jnp.array([1, 0, 0]))

    # Combined (4,3) array of all patch-particle coordinates
    R_model = jnp.array([particle1_src, particle2_src, particle1_mac, particle2_mac])

    return R_model


def get_R_initial(params, global_options, shift):
    """ Get initial states for escape time calculation. """

    # Generate initial states
    R_ini1 = get_R_model(global_options, shift, params['mac_sigma0'], params['src_sigma0'], bound=False)
    R_ini2 = get_R_model(global_options, shift, params['mac_sigma0'], params['mac_sigma0'], bound=True)
    R_ini3 = get_R_model(global_options, shift, params['mac_sigma1'], params['mac_sigma1'], bound=True)
    R_ini4 = get_R_model(global_options, shift, params['mac_sigma1'], params['src_sigma1'], bound=False)
    R_ini5 = get_R_model(global_options, shift, params['mac_sigma0'], params['src_sigma0'], bound=True)
    # R_ini4 = get_R_model(global_options, shift, params['mac_sigma1'], params['src_sigma1'], bound=True)
    R_initial = jnp.array([R_ini1, R_ini2, R_ini3, R_ini4, R_ini5])

    return R_initial


def initialize_system(global_options, params, shift, key, caps=False):
    """ Initialize the system of Sources, Machine, and caps. Place FSs and M in their closed length in a random
    orientation. Each Source is capped, the remaining caps are randomly placed.
    1. Start with a random (N,3) vector
    2. Adjust every 2nd entry of FSs and M to match dumbbell length in random orientation
    3. Locations of first 2*Nfs caps are identical to locations of FSs.
    Args:
        - global_options.
        - params.
        - shift.
        - key.
        - caps: If True, the first 2*Nfs caps are placed at exact locations of FSs.
    Return:
        - Array of initial locations of all 3 species.
     """

    dimension = global_options['dimension']
    key, subkey = random.split(key)
    Nfs, Nm, Nc = global_options['species_numbers']
    N = 2 * (Nfs + Nm + Nc)

    ## R_initial
    R_init = random.uniform(subkey, (N, dimension), dtype=jnp.float64, maxval=global_options['side_length'])
    # Locations FS
    for n in jnp.arange(0, 2 * Nfs, 2):
        p1, p2 = n, n + 1
        key, subkey = random.split(key)
        v = random.normal(subkey, (dimension,), dtype=jnp.float64)
        v_hat = v * params['src_sigma0'] / jnp.linalg.norm(v)
        R_init = R_init.at[p2].set(shift(R_init[p1], v_hat))
    # Locations M
    for n in jnp.arange(2 * Nfs, 2 * Nfs + 2 * Nm, 2):
        p1, p2 = n, n + 1
        key, subkey = random.split(key)
        v = random.normal(subkey, (dimension,), dtype=jnp.float64)
        v_hat = v * params['mac_sigma0'] / jnp.linalg.norm(v)
        R_init = R_init.at[p2].set(shift(R_init[p1], v_hat))
    # Locations of first 2*Nfs caps
    if caps == True:
        R_init = R_init.at[2 * Nfs + 2 * Nm:4 * Nfs + 2 * Nm].set(R_init[0:2 * Nfs])
    return R_init



### Energy functions

def Morse_bond(displacement_or_metric, bond, sigma=0, epsilon=2, alpha=5):
    """Convenience wrapper to compute interaction energy of patches bonded by harmonic Morse potential."""
    return smap.bond(
        energy.morse,
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        bond,
        sigma=sigma,
        epsilon=epsilon,
        alpha=alpha)


def setup_energy_double_morse(params):
    """ Machine-readable parameters. Translation to human-readable parameters not done during optimization
         Credits: Carl Goodrich.

         Args:
             params: parameters of 1D double-well potential.

         Returns:
             energy function.
    """
    epsilon0, epsilon1, sigma0, sigma1, alpha0, alpha1, A, B = params

    def energy_fn(r):
        r = jnp.array(r, jnp.float64)
        E0 = energy.morse(r - sigma0, sigma=0, epsilon=epsilon0, alpha=alpha0)
        E1 = energy.morse(sigma1 - r, sigma=0, epsilon=epsilon1, alpha=alpha1)
        return E0 + E1 + A + B * r

    return energy_fn


def get_energy_double_morse(params):
    params_mac = jnp.array([params['mac_epsilon0'], params['mac_epsilon1'], params['mac_sigma0'], params['mac_sigma1'],
                            params['mac_alpha0'], params['mac_alpha1'], params['mac_A'], params['mac_B']])
    params_src = jnp.array([params['src_epsilon0'], params['src_epsilon1'], params['src_sigma0'], params['src_sigma1'],
                            params['src_alpha0'], params['src_alpha1'], params['src_A'], params['src_B']])
    energy_mac_fn = setup_energy_double_morse(params_mac)
    energy_src_fn = setup_energy_double_morse(params_src)
    return energy_mac_fn, energy_src_fn


def setup_energy_individual(displacement, mac=True):
    """ Set up energy function for Machine or Source. Note that indices of M and S are baked in."""
    def get_energy_fn(params):
        energy_mac_fn, energy_src_fn = get_energy_double_morse(params)
        if mac:
            energy_idv_fn = energy_mac_fn
        else:
            energy_idv_fn = energy_src_fn
        @jit
        def energy_fn(R, **kwargs):
            e1 = energy_idv_fn(jnp.linalg.norm(displacement(R[1], R[0])))
            return e1
        return energy_fn
    return get_energy_fn


def setup_energy(params, global_options, displacement):
    """ Set up full energy function of a general system with Nfs Sources and 1 Machine.
    We avoid crosstalk, so each dimer has a “left” and a “right” monomer/patch, and a left FS patch can only bind to a
    left M patch, and so on. The interaction energy is given by the pair potentials of L/R particles, the dimer energies
    are given by double Morse potentials. Current assumption: Nleft = Nright.
        Args:
            - params.
            - global_options.
            - displacement.
        Returns:
            Full energy function.
        """
    Nfs, Nm, Nc = global_options['species_numbers']
    assert Nm == 1  # This is hard-core baked in...
    assert Nc == 0

    metric = space.canonicalize_displacement_or_metric(displacement)

    fm_bonds = jnp.array([jnp.arange(2 * Nfs), jnp.array([-2, -1] * Nfs) + 2 * (Nfs + 1)]).T
    energy_patch_fn = smap.bond(energy.morse,
                                metric,
                                fm_bonds,
                                sigma=0.,
                                epsilon=params['epsilon_pp'],
                                alpha=params['alpha_pp']
                                )

    V_m, V_s = get_energy_double_morse(params)  # Functions with all parameters baked in
    fs_bonds = jnp.arange(2 * Nfs).reshape((Nfs, 2))
    energy_fs_fn = smap.bond(V_s, metric, fs_bonds)

    m_bonds = jnp.array([[-2, -1]])
    energy_m_fn = smap.bond(V_m, metric, m_bonds)

    @jit
    def energy_fn(R, **kwargs):
        """ Calculate full energy of the system.
        Args:
            - R:   Locations of all particles with shape (2*Nfs+2,dimension).
        """

        energy_patch = energy_patch_fn(R)
        energy_fs = energy_fs_fn(R)
        energy_m = energy_m_fn(R)

        return energy_patch + energy_fs + energy_m

    return energy_fn, fm_bonds


def setup_energy_dynamic(params, global_options, displacement):
    """ Set up full energy function of a general system with Nfs Sources and 1 Machine. Include bounded times for
    each pair of FS-M monomers as dynamic variable.
    We avoid crosstalk, so each dimer has a “left” and a “right” monomer/patch, and a left FS patch can only bind to a
    left M patch, and so on. The interaction energy is given by the pair potentials of L/R particles, the dimer energies
    are given by double Morse potentials. Current assumption: Nleft = Nright.
        Args:
            - params.
            - global_options.
            - displacement.
        Returns:
            Full energy function.
        """
    Nfs, Nm, Nc = global_options['species_numbers']
    assert Nm == 1  # This is hard-core baked in...
    assert Nc == 0

    metric = space.canonicalize_displacement_or_metric(displacement)

    fm_bonds = jnp.array([jnp.arange(2 * Nfs), jnp.array([-2, -1] * Nfs) + 2 * (Nfs + 1)]).T
    energy_patch_fn = smap.bond(energy.morse,
                                metric,
                                fm_bonds,
                                sigma=0.,
                                epsilon=999999999999999999,  # This should be dynamically overwritten every time!
                                alpha=params['alpha_pp']
                                )

    def set_epsilon(pp_times):
        return params['epsilon_pp'] * jnp.exp(- pp_times / global_options['pp_decay_time'])

    V_m, V_s = get_energy_double_morse(params)  # Functions with all parameters baked in
    fs_bonds = jnp.arange(2 * Nfs).reshape((Nfs, 2))
    energy_fs_fn = smap.bond(V_s, metric, fs_bonds)

    m_bonds = jnp.array([[-2, -1]])
    energy_m_fn = smap.bond(V_m, metric, m_bonds)

    @jit
    def energy_fn(R, pp_times):
        """ Calculate full energy of the system.
        Args:
            - R:        Locations of all particles with shape (2*Nfs+2,dimension).
            - pp_times: dynamic variable of binding times for each pair of FS-M monomers, array with shape (2*Nfs,).
        """
        epsilon = set_epsilon(pp_times)

        energy_patch = energy_patch_fn(R, epsilon=epsilon)
        energy_fs = energy_fs_fn(R)
        energy_m = energy_m_fn(R)

        return energy_patch + energy_fs + energy_m

    return energy_fn, fm_bonds