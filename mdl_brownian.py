import jax.numpy as jnp
from typing import Any, Callable, TypeVar, Union, Tuple, Dict, Optional
from jax import random, jit, lax, vmap, config
config.update("jax_enable_x64", True)
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_unflatten
from jax_md import space, quantity, util, dataclasses
import xarray as xr

### import my own modules
from mdl_classification import (get_thresholds, get_state_parameters, determine_state, determine_dissociation,
                                determine_association, get_dimer_lengths, get_separation_from_R, get_angle)

# Brownian dynamics are synonymous with the overdamped regime of Langevin dynamics. However, in this case we don't need
# to take into account velocity information and the dynamics simplify. Consequently, when Brownian dynamics can be used
# they will be faster than Langevin. [JAX MD documentation, Simulation Routines]

###### Code copied from jax_md.simulate ######

# static_cast issue: the stepsize cannot be too small (dt<3e-8), otherwise dR is uniformly zero. This has been reported
# in this issue as well https://github.com/jax-md/jax-md/issues/218 and linked to `dt, gamma = static_cast(dt, gamma)`
# in `jax_md.simulate.brownian`. I copied the entire brownian dynamics functions and manually removed that line.

Array = util.Array
f64 = jnp.float64

Box = space.Box

ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]

"""Dispatch By State Code.

JAX MD allows for simulations to be extensible using a dispatch strategy where
functions are dispatched to specific cases based on the type of state provided.
In particular, we make decisions about which function to call based on the type
of the position argument. For those familiar with C / C++, our dispatch code is
essentially function overloading based on the type of the positions.

If you are interested in setting up a simulation using a different type of
system you can do so in a relatively light weight manner by introducing a new
type for storing the state that is compatible with the JAX PyTree system
(we usually choose a dataclass) and then overriding the functions below.

These extensions allow a range of simulations to be run by just changing the
type of the position argument. There are essentially two types of functions to
be overloaded. Functions that compute physical quantities, such as the kinetic
energy, and functions that evolve a state according to the Suzuki-Trotter
decomposition. Specifically, one might want to override the position step,
momentum step for deterministic and stochastic simulations or the
`stochastic_step` for stochastic simulations (e.g Langevin).
"""


class dispatch_by_state:
    """Wrap a function and dispatch based on the type of positions."""

    def __init__(self, fn):
        self._fn = fn
        self._registry = {}

    def __call__(self, state, *args, **kwargs):
        if type(state.position) in self._registry:
            return self._registry[type(state.position)](state, *args, **kwargs)
        return self._fn(state, *args, **kwargs)

    def register(self, oftype):
        def register_fn(fn):
            self._registry[oftype] = fn

        return register_fn


@dispatch_by_state
def canonicalize_mass(state: T) -> T:
    """Reshape mass vector for broadcasting with positions."""

    def canonicalize_fn(mass):
        if isinstance(mass, float):
            return mass
        if mass.ndim == 2 and mass.shape[1] == 1:
            return mass
        elif mass.ndim == 1:
            return jnp.reshape(mass, (mass.shape[0], 1))
        elif mass.ndim == 0:
            return mass
        msg = (
            'Expected mass to be either a floating point number or a one-dimensional'
            'ndarray. Found {}.'.format(mass)
        )
        raise ValueError(msg)

    return state.set(mass=tree_map(canonicalize_fn, state.mass))


@dataclasses.dataclass
class BrownianState:
    """A tuple containing state information for Brownian dynamics.

    Attributes:
        position: The current position of the particles. An ndarray of floats with
          shape `[n, spatial_dimension]`.
        mass: The mass of particles. Will either be a float or an ndarray of floats
          with shape `[n]`.
        rng: The current state of the random number generator.
    """
    position: Array
    mass: Array
    rng: Array


def brownian(energy_or_force: Callable[..., Array],
             shift: ShiftFn,
             dt: float,
             kT: float,
             gamma: float = 0.1) -> Simulator:
    """Simulation of Brownian dynamics.

    Simulates Brownian dynamics which are synonymous with the overdamped
    regime of Langevin dynamics. However, in this case we don't need to take into
    account velocity information and the dynamics simplify. Consequently, when
    Brownian dynamics can be used they will be faster than Langevin. As in the
    case of Langevin dynamics our implementation follows Carlon et al. [#carlon]_

    Args:
        energy_or_force: A function that produces either an energy or a force from
          a set of particle positions specified as an ndarray of shape
          `[n, spatial_dimension]`.
        shift: A function that displaces positions, `R`, by an amount `dR`.
          Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
        dt: Floating point number specifying the timescale (step size) of the
          simulation.
        kT: Floating point number specifying the temperature in units of Boltzmann
          constant. To update the temperature dynamically during a simulation one
          should pass `kT` as a keyword argument to the step function.
        gamma: A float specifying the friction coefficient between the particles
          and the solvent.

    Returns:
        See above.
  """

    force_fn = quantity.canonicalize_force(energy_or_force)

    # dt, gamma = static_cast(dt, gamma)        # static_cast issue

    def init_fn(key, R, mass=f64(1)):
        state = BrownianState(R, mass, key)
        return canonicalize_mass(state)

    def apply_fn(state, **kwargs):
        _kT = kT if 'kT' not in kwargs else kwargs['kT']

        R, mass, key = dataclasses.astuple(state)

        key, split = random.split(key)

        F = force_fn(R, **kwargs)
        xi = random.normal(split, R.shape, R.dtype)

        nu = f64(1) / (mass * gamma)

        dR = F * dt * nu + jnp.sqrt(f64(2) * _kT * dt * nu) * xi
        R = shift(R, dR, **kwargs)

        return BrownianState(R, mass, key)  # pytype: disable=wrong-arg-count

    return init_fn, apply_fn


# Run a brownian dynamics simulation and save periodic snapshots
def run_brownian(energy_fn, R_init, shift, key,
                 num_total_steps, record_every, dt,
                 kT=1.0, gamma=0.1, measure_fn=lambda R: R, **kwargs):
    # define the simulation
    init, apply = brownian(energy_fn, shift, dt, kT, gamma, **kwargs)
    apply = jit(apply)

    @jit
    def apply_single_step(state, t):
        return apply(state, t=t), 0

    # apply_single_step = jit(apply_single_step, donate_argnums=0)

    @jit
    def apply_many_steps(state, t_outer):
        t_inner = t_outer * record_every + t_range_inner
        state, _ = lax.scan(apply_single_step, state, t_inner)
        return state, measure_fn(state.position)

    # apply_many_steps = jit(apply_many_steps, donate_argnums=0)

    # initialize the system
    key, split = random.split(key)
    initial_state = init(split, R_init)

    # Storing an array jnp.arange(num_total_steps) requires a lot of memory allocation for num_total_steps>=1e10.
    # Therefore, instead of jnp.arange(num_total_steps).reshape(num_total_steps // record_every, record_every)
    # we use 2 ranges for inner and outer scan
    t_range_inner = jnp.arange(record_every)
    t_range_outer = jnp.arange(num_total_steps // record_every)

    # run the simulation
    final_state, data = lax.scan(apply_many_steps, initial_state, t_range_outer)

    # return the trajectory
    return final_state, data


def get_state(R, thresholds, displacement):
    state_params = get_state_parameters(R, displacement)
    state = determine_state(state_params, thresholds)
    return state


def recharge_source(R, fm_bonds, params, displacement, shift):
    """ Recharge a Source if it is consumed (in state Y) and far away from Machine as well as reset the timer for
    binding energies. Every recharge event can only recharge 1 FS, that with smallest index. We assume that we do it
    often enough, such that only one Source has to be recharged at a time.
    If a Source is in open state Y and not within a sphere of radius dr_recharge to the Machine, then convert it
    into closed state X. This guarantees that the Machine "sees" a constant concentration of Sources.
    Choose dr_recharge such that even M and FS don't feel each other even in worst-case scenario: FS that is flipped
    compared to M (L/R outside spheres) recharges -> new L/R are much closer.
    Important note: This breaks physics. We have to be careful with implications of it, but it should be fine.
    Alternatively we could introduce source and sink regions and have a constant flow of Sources between them.

    Args:
        - R: Array of positions with shape (N,3).
        - fm_bonds:
        - params.
        - global_options.
        - displacement.
        - shift.

    Return:
        Locations of all particles where 1 Source has been recharged and updated timer for binding energies if conditions are met.
    """

    lengths_fs, _ = get_dimer_lengths(R, displacement)
    fm_bond_lengths = space.distance(vmap(lambda r: displacement(r[0], r[1]))(R[fm_bonds]))
    drs_min = jnp.min(fm_bond_lengths.reshape((-1, 2)), axis=1)
    # cutoff distance of sphere around Machine in which we do not recharge Sources
    dr_recharge = (params['src_sigma1'] - params['src_sigma0']) / 2 + 8 / params['alpha_pp']
    rechargeable = drs_min > dr_recharge  # True if far away from Machine

    # Get index of first Source that needs to be recharged, i.e., is in state Y and >dr_recharge away from Machine
    idx_fs = jnp.argwhere(jnp.logical_and(lengths_fs > params['src_sigma1'], rechargeable), size=1, fill_value=-1)[0][0]

    # Keep COM, place both patches at length sX in same direction as before
    def true_fn(idx, R):
        is_re = True
        i1 = 2 * idx
        i2 = i1 + 1
        dR = displacement(R[i2], R[i1])
        R_com = shift(R[i1], 0.5 * dR)
        r = 1 / 2 * params['src_sigma0'] * dR / jnp.linalg.norm(dR)
        # Place Source at new locations
        R = R.at[i1].set(shift(R_com, -r)).at[i2].set(shift(R_com, r))
        return R, is_re

    def false_fn(idx, R):
        is_re = False
        return R, is_re

    # reset positions if rechargeable
    R_new, is_re = lax.cond(idx_fs >= 0, true_fn, false_fn, idx_fs, R)

    return R_new, is_re


def recharge_and_reset(R, pp_times, fm_bonds, params, displacement, shift):
    """ Recharge a Source if it is consumed (in state Y) and far away from Machine as well as reset the timer for
    binding energies. Every recharge event can only recharge 1 FS, that with smallest index. We assume that we do it
    often enough, such that only one Source has to be recharged at a time.
    If a Source is in open state Y and not within a sphere of radius dr_recharge to the machine, then convert it
    into closed state X. This guarantees that the Machine "sees" a constant concentration of Sources.
    Choose dr_recharge such that even M and FS don't feel each other even in worst-case scenario: FS that is flipped
    compared to M (L/R outside spheres) recharges -> new L/R are much closer.
    Important note: This breaks physics. We have to be careful with implications of it, but it should be fine.
    Alternatively we could introduce source and sink regions and have a constant flow of Sources between them.

    Args:
        - R: Array of positions with shape (N,3).
        - pp_times: Array of patch-patch binding times with shape (2*Nfs,)
        - fm_bonds:
        - params.
        - global_options.
        - displacement.
        - shift.

    Return:
        Locations of all particles where 1 Source has been recharged and updated timer for binding energies if conditions are met.
    """

    lengths_fs, _ = get_dimer_lengths(R, displacement)
    fm_bond_lengths = space.distance(vmap(lambda r: displacement(r[0], r[1]))(R[fm_bonds]))
    drs_min = jnp.min(fm_bond_lengths.reshape((-1, 2)), axis=1)
    # cutoff distance of sphere around M in which we do not recharge Sources
    dr_recharge = (params['src_sigma1'] - params['src_sigma0']) / 2 + 8 / params['alpha_pp']
    rechargeable = drs_min > dr_recharge  # True if far away from Machine

    # Get index of first Source that needs to be recharged, i.e., is in state Y and >dr_recharge away from Machine
    idx_fs = jnp.argwhere(jnp.logical_and(lengths_fs > params['src_sigma1'], rechargeable), size=1, fill_value=-1)[0][0]
    # idx_fs.shape = (1,)

    # Keep COM, place both patches at length sX in same direction as before
    def true_fn(idx, R):
        is_re = True
        i1 = 2 * idx
        i2 = i1 + 1
        dR = displacement(R[i2], R[i1])
        R_com = shift(R[i1], 0.5 * dR)
        r = 1 / 2 * params['src_sigma0'] * dR / jnp.linalg.norm(dR)
        # Place Source at new locations
        R = R.at[i1].set(shift(R_com, -r)).at[i2].set(shift(R_com, r))
        return R, is_re

    def false_fn(idx, R):
        is_re = False
        return R, is_re

    # reset positions if rechargeable
    R_new, is_re = lax.cond(idx_fs >= 0, true_fn, false_fn, idx_fs, R)
    # reset the timer for binding energies if rechargeable
    pp_times_new = jnp.where(jnp.repeat(rechargeable, 2), 0, pp_times)

    return R_new, pp_times_new, is_re
