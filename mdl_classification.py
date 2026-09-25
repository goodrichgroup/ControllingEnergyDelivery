import jax.numpy as jnp
from jax import vmap, value_and_grad, debug
from jax_md import space, energy
# from jax_transformations3d import jax_transformations3d as jts
from scipy.signal import butter, filtfilt

### import my own modules
from mdl_model import setup_energy, get_energy_double_morse


def get_separation_from_R(R, displacement, type='mac'):
    if type == 'mac':
        p1, p2 = 2, 3    # 0, 1
    elif type == 'src':
        p1, p2 = 0, 1    # 2, 3
    elif type == 'pat1':
        p1, p2 = 0, 2
    elif type == 'pat2':
        p1, p2 = 1, 3
    else:
        assert False
    metric = space.map_product(space.canonicalize_displacement_or_metric(displacement))
    dists = metric(R, R)
    sep = dists[p1, p2]
    return sep


def get_dimer_lengths(R, displacement):
    """ Get dimer lengths of all Sources and one Machine.
        Args:
            - R: Array with all locations of shape (N,3)
            - displacement.
        Return:
            Arrays of Source lengths (Nfs,) and M length (1,).
        """
    lengths = space.distance(vmap(displacement)(R[::2], R[1::2]))
    return lengths[:-1], lengths[-1]


def get_angle(R, displacement):
    """ Calculate angle between Machine and Fuel Source. """
    v_src = displacement(R[0], R[1])
    v_mac = displacement(R[3], R[2])
    angle = jnp.arccos(jnp.dot(v_src, v_mac) / (jnp.linalg.norm(v_src) * jnp.linalg.norm(v_mac)))
    return angle


def get_thresholds(params):   # , association=True
    """ Get thresholds for patch separation and reaction coordinates of Machine and Source.
    Threshold on patch separation is determined as location where Morse potential is equal to an energy_threshold, set
    to 0.1*epsilon (association) and 0.01*epsilon (dissociation). Thresholds on reaction coordinates r and s are given
    by locations of transition states.
    Args:
        - params.
        - association: Flag if we take association or dissociation threshold.
    Return:
        Array with 4 thresholds.
    """

    # Threshold on separation of patches
    # if association == True:
    #     prefactor = 0.5
    # else:
    #     prefactor = 0.01
    prefactor1 = 0.01  # association: 0.95, dissociation: 0.01. escape_time: 0.01 (start in doubly-bound state),
    prefactor2 = 0.95  # escape_time: 0.95 (start in singly-bound state A-X, B-Y) 0.01 (start in doubly-bound state D1, D2)
    rs = jnp.linspace(0, 10 / params['alpha_pp'], 1000)
    Epp2 = energy.morse(rs, sigma=0, epsilon=params['epsilon_pp'], alpha=params['alpha_pp'])
    thresh_p1 = rs[jnp.min(jnp.argwhere(Epp2 > -prefactor1 * params['epsilon_pp']))]
    thresh_p2 = rs[jnp.min(jnp.argwhere(Epp2 > -prefactor2 * params['epsilon_pp']))]

    # Thresholds on reaction coordinates r and s
    energy_mac_fn, energy_src_fn = get_energy_double_morse(params)
    rs_mac = jnp.linspace(params['mac_sigma0'], params['mac_sigma1'], 1000)
    rs_src = jnp.linspace(params['src_sigma0'], params['src_sigma1'], 1000)
    Es_mac = energy_mac_fn(rs_mac)
    Es_src = energy_src_fn(rs_src)
    thresh_M = rs_mac[jnp.argmax(Es_mac)]   # +0.25, -0.25
    thresh_FS = rs_src[jnp.argmax(Es_src)]  # +0.5, -0.2

    thresholds = jnp.array([thresh_p1, thresh_p2, thresh_M, thresh_FS])
    return thresholds


def get_state_parameters(R, displacement):
    """ Get state parameters from current positions. Calculate the dumbbell lengths of the closest Machine-Source pair
    and their separations.
    Assumption: 1 Machine,1N Source.
    Args:
        - R: Array of positions with shape (N,3).
        - global_options.
        - displacement.
    Return:
        Array with 4 state parameters.
        """
    r1 = get_separation_from_R(R, displacement, 'pat1')
    r2 = get_separation_from_R(R, displacement, 'pat2')
    m = get_separation_from_R(R, displacement, 'mac')
    s = get_separation_from_R(R, displacement, 'src')
    state_parameters = jnp.array([r1, r2, m, s])

    return state_parameters


def determine_state(state_parameters, thresholds):
    """ Determine current state of system based on 4 state parameters.
    Assumption: If 2 patches are bound, distinguish two states (I1 and I2). If 0 or 1 patches are bound, distinguish 4 states.
    Note that in some cases less than 4 parameters could be enough, but we include all to have redundancy.
    Args:
         - state_parameters: Array with shape (4,) including
           p1: distance of 1st pair of patches.
           p2: distance of 2nd pair of patches.
           r:  reaction coordinate of Machine.
           s:  reaction coordinate of Source.
         - thresholds: thresholds for patches, r, and s.
    Return:
         State of the system according to classification.
    """

    r1, r2, m, s = state_parameters

    r1_tilde = r1 - thresholds[0]  # negative if 1st pair of patches are bound, positive if they are unbound
    r2_tilde = r2 - thresholds[1]  # negative if 2nd pair of patches are bound, positive if they are unbound
    m_tilde = m - thresholds[2]    # negative for state A, positive for state B
    s_tilde = s - thresholds[3]    # negative for state X, positive for state Y

    # 1. Distinguish states with 0, 1 or 2 bonds
    condlist = [jnp.logical_and(r1_tilde > 0, r2_tilde > 0),
                jnp.logical_xor(jnp.logical_and(r1_tilde > 0, r2_tilde < 0),
                                jnp.logical_and(r1_tilde < 0, r2_tilde > 0)),
                jnp.logical_and(r1_tilde < 0, r2_tilde < 0)]
    choicelist = [0, 10, 20]
    state = jnp.select(condlist, choicelist)

    # 2. Distinguish states in closed or open configuration
    condlist = [jnp.logical_and(m_tilde < 0, s_tilde < 0),  # (A+X, AX, D1)
                jnp.logical_and(m_tilde < 0, s_tilde > 0),  # (A+Y, AY, D1)
                jnp.logical_and(m_tilde > 0, s_tilde < 0),  # (B+X, BX, D2)
                jnp.logical_and(m_tilde > 0, s_tilde > 0)]  # (B+Y, BY, D2)
    choicelist = jnp.where(state == 20, jnp.array([1, 1, 2, 2]), jnp.array([1, 2, 3, 4]))
    state += jnp.select(condlist, choicelist)

    return state


def get_state(R, thresholds, displacement):
    state_parameters = get_state_parameters(R, displacement)
    state = determine_state(state_parameters, thresholds)
    return state


def determine_dissociation(R, thresholds, global_options, displacement):
    Nfs, Nm, Nc = global_options['species_numbers']
    metric = space.map_product(space.canonicalize_displacement_or_metric(displacement))
    dists = metric(R, R)
    drs_fs_m = dists[:2 * Nfs, 2 * Nfs:2 * Nfs + 2 * Nm]          # distances of all Sources to M
    drs_min = jnp.min(drs_fs_m)                                   # minimum distance of all Machine-Source pairs
    dissociation = jnp.where(drs_min > thresholds[0], 1, 0)
    return dissociation


def determine_association(dists, thresholds, global_options):
    """ Determine if Machine is associated to a Source based on minimum distance of all Machine-Source pairs.
    Args:
        - R // dists: distance matrix of array R with all locations, with shape (N,N).
        - thresholds.
        - global_options.
        - displacement. (only needed for metric)
    Returns: Number of associated bonds 0, 1, or 2.
    """

    Nfs, Nm, Nc = global_options['species_numbers']
    # metric = space.map_product(space.canonicalize_displacement_or_metric(displacement))
    # dists = metric(R, R)
    drs_fs_m = dists[:2 * Nfs, 2 * Nfs:2 * Nfs + 2 * Nm]          # distances of all Sources to Machine
    drs_min = jnp.min(drs_fs_m, axis=1)                           # 2 smallest distances of all Machine-Source pairs
    association = jnp.where(drs_min[0] < thresholds[0], 1, 0)
    association += jnp.where(drs_min[1] < thresholds[1], 1, 0)
    return association