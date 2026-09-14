#!/usr/bin/env python3

print('Start running Brownian dynamics simulations of a coupled system with 1 Source and 1 Machine.')

###### Imports ######

from jax import jit, lax, vmap, random, config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
f32 = jnp.float32
f64 = jnp.float64
from jax_md import space

import os
import sys
import pickle
import xarray as xr

# import my own modules
from mdl_model import setup_energy, initialize_system
from mdl_brownian import brownian, BrownianState, recharge_source

###### Directory, parameters, and global options ######

job_idx = jnp.int64(sys.argv[1])
seed = 321 * job_idx + 7654               # used to generate PRNGKey
key = random.key(seed)  # PRNGKey

data_dir = (r'/link/to/directory/')
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
file = os.path.join(data_dir, 'params.pickle')
with open(file, 'rb') as f:
    params = pickle.load(f)
# params_upd = {'epsilon_pp': 15.}
# params = {**params, **params_upd}

# global parameters and options
global_options = {
    'dimension': 3,                     # physical dimensions in the system, typically 3D, but can also be 2D
    # Brownian
    'kT': 1.0,                          # thermal energy, specifying the temperature in units of Boltzmann constant; beta=1
    'gamma': 0.1,                       # friction coefficient between the particles and the solvent
    'dt': 1e-5,                         # timescale (step size) of Brownian dynamics simulation; small enough to resolve forces, such that no observable should change, for alternative run with dt/2 and 2*num_total_steps
    'num_total_steps': 2e10,            # Total number of MD steps
    'recharge_steps': 1e2,              # recharge Source every 100 steps (slows down if done every dt)
    'apply_N_steps': 1e5,               # apply N steps in inner loop of MD simulation
    'save_every': 1e4,                  # save trajectory (lengths) every x steps
    'key_reseed': 1e7,                  # reseed random keys every x steps
    'side_length': 10,                  # sidelength specifying the size of each side of the periodic box (hypercube) for periodic boundary conditions
    'species_numbers': [1, 1],          # amount of [Sources, Machines] in system
}
if job_idx == 1:
    file = os.path.join(data_dir, 'global_options.pickle')  # _{:02d} .format(job_idx)
    with open(file, 'wb') as f:
        pickle.dump(global_options, f)
    file = os.path.join(data_dir, 'params.pickle')
    with open(file, 'wb') as f:
        pickle.dump(params, f)

key, subkey = random.split(key)
seeds = random.bits(subkey, shape=int(global_options['num_total_steps'] / global_options['key_reseed']))

###### Run ######

### Initialization
displacement, shift = space.periodic(global_options['side_length'])
metric = space.map_product(space.canonicalize_displacement_or_metric(displacement))
Nfs, Nm = global_options['species_numbers']
m_thresh = 2.0  # (hard-coded) depends on parameters #
int_thresh = 2 / params['alpha_pp']

## Energy function
energy_fn, fm_bonds = setup_energy(params, global_options, displacement)

## Get initial locations of Sources, Machine, and caps
key, split = random.split(key)
R_init = initialize_system(global_options, params, shift, split, caps=False)


### Run a brownian dynamics simulation and save quantities of interest

def update_state(m, s):
    # update Machine state variable from current Machine length m and current state s.
    is_upd = jnp.where(s == 0, m >= params['mac_sigma1'], m >= params['mac_sigma0'])
    return jnp.where(is_upd, 1, 0)

def update_trajectory(trajectory, time_md, state):
    sep_m = space.distance(displacement(state.position[2], state.position[3]))
    sep_s = space.distance(displacement(state.position[0], state.position[1]))
    sep_r1 = space.distance(displacement(state.position[0], state.position[2]))
    sep_r2 = space.distance(displacement(state.position[1], state.position[3]))
    return trajectory.at[jnp.int64(time_md / global_options['save_every'] - 1)].set([sep_m, sep_s, sep_r1, sep_r2])

def is_usable(m, state):
    i1, i2 = fm_bonds[0]
    sep_r1 = space.distance(displacement(state.position[i1], state.position[i2]))
    i1, i2 = fm_bonds[1]
    sep_r2 = space.distance(displacement(state.position[i1], state.position[i2]))
    # Define usable Machine state by (i) open configuration with (ii) 0/1 bonds with Source
    is_free_0bond = jnp.logical_and(sep_r1 > int_thresh, sep_r2 > int_thresh)
    is_free_1bond = jnp.logical_or(sep_r1 > int_thresh, sep_r2 > int_thresh)
    is_B = m > m_thresh  # thresholds[2]
    is_use0 = jnp.where(jnp.logical_and(is_B, is_free_0bond), True, False)
    is_use1 = jnp.where(jnp.logical_and(is_B, is_free_1bond), True, False)
    return is_use0, is_use1

def reseed_key(state, time_md):
    # reseed key every x steps using seed from seeds array
    # there are 2^32 / key_reseed / 2 possible random number sequences per key_reseed steps
    reseed_idx = jnp.int64(time_md / global_options['key_reseed'] - 1)
    key_reseed = jnp.where(time_md % global_options['key_reseed'] == 0, random.key(seeds[reseed_idx]), state.rng)
    state_upd = BrownianState(state.position, state.mass, key_reseed)
    return state_upd

@jit
def apply_single_step(data, t):

    # 1. Update Brownian state
    (state_curr, s_curr, times_trans, n_trans, N_re, t_use0, t_use1, time_md, trajectory) = data
    state_upd = apply(state_curr)
    time_md += 1

    state_upd = reseed_key(state_upd, time_md)

    # 2. Update Machine state and running variables
    m = space.distance(displacement(state_upd.position[2], state_upd.position[3]))
    s_upd = update_state(m, s_curr)
    times_trans = jnp.where(s_upd == s_curr, times_trans, times_trans.at[n_trans].set(time_md))
    n_trans = jnp.where(s_upd == s_curr, n_trans, n_trans + 1)
    trajectory = jnp.where(time_md % global_options['save_every'] == 0, update_trajectory(trajectory, time_md, state_upd), trajectory)

    is_use0, is_use1 = is_usable(m, state_upd)
    t_use0 = jnp.where(is_use0, t_use0.at[n_trans].add(1), t_use0)
    t_use1 = jnp.where(is_use1, t_use1.at[n_trans].add(1), t_use1)

    return (state_upd, s_upd, times_trans, n_trans, N_re, t_use0, t_use1, time_md, trajectory), 0


@jit
def apply_recharge_step(data, t):

    data, _ = lax.scan(apply_single_step, data, length=int(global_options['recharge_steps']))
    (state_curr, s_curr, times_trans, n_trans, N_re, t_use0, t_use1, time_md, trajectory) = data

    # 3. Recharge Source
    R_re, is_re = recharge_fuel_source(state_curr.position, fm_bonds, params, displacement, shift)
    N_re_upd = jnp.where(is_re, N_re.at[n_trans].add(1), N_re)
    state_upd_re = BrownianState(R_re, state_curr.mass, state_curr.rng)

    return (state_upd_re, s_curr, times_trans, n_trans, N_re_upd, t_use0, t_use1, time_md, trajectory), 0


@jit
def apply_N_steps(data, t):
    # Apply N steps and recharge
    data, _ = lax.scan(apply_recharge_step, data, length=int(global_options['apply_N_steps'] / global_options['recharge_steps']))
    return data, 0


### Run the simulation

## 1. Running variables for certain quantities
time_md = 0                                      # running variable for current simulation time
n_trans = 0                                      # running variable for current number of Machine transition
s = 0                                            # running variable for Machine state: 0/1
Ntrans = 1000                                    # default number of Machine transitions to save
times_trans = -1 * jnp.ones(Ntrans, dtype=int)   # running variable for time points of Machine transitions
trajectory = jnp.zeros((int(global_options['num_total_steps'] / global_options['save_every']), 4))
# even = transition closed -> open, odd = transition open -> closed
# If there are more than Ntrans transitions they will not be added to times_trans.
# if there are less, we remove the empty lines from the array before saving data
N_re = jnp.zeros(Ntrans, dtype=int)             # running variable for number of Source recharges
t_use0 = jnp.zeros(Ntrans, dtype=int)           # running variable for time in usable Machine state with 0 bonds to Source
t_use1 = jnp.zeros(Ntrans, dtype=int)           # running variable for time in usable Machine state with 1 bond  to Source

## 2. Define the simulation
init, apply = brownian(energy_fn, shift, global_options['dt'], global_options['kT'], global_options['gamma'])
apply = jit(apply)

## 3. Run MD
key, split = random.split(key)
state_init = init(split, R_init)
data_init = (state_init, s, times_trans, n_trans, N_re, t_use0, t_use1, time_md, trajectory)

# we can't create big array, throws memory issues for 2e10 -> use 2 ranges for inner and outer scan
data_final, _ = lax.scan(apply_N_steps, data_init, length=int(global_options['num_total_steps'] / global_options['apply_N_steps']))
(state_final, s_final, times_trans, n_trans, N_re, t_use0, t_use1, time_md_final, trajectory) = data_final

# Save data
outdata = xr.Dataset({
    's_final': xr.DataArray(s_final),
    'n_trans': xr.DataArray(n_trans),
    'times_trans': xr.DataArray(times_trans[:n_trans], dims=('num_trans',)),
    'trajectory': xr.DataArray(trajectory, dims=('num_traj','lengths')),
    'N_re': xr.DataArray(N_re[:n_trans], dims=('num_trans',)),
    't_use0': xr.DataArray(t_use0[:n_trans], dims=('num_trans',)),
    't_use1': xr.DataArray(t_use1[:n_trans], dims=('num_trans',)),
})
# Compress data
comp = {'zlib': True, 'complevel': 5}
encoding = {var: comp for var in outdata.data_vars}
outdata.to_netcdf(os.path.join(data_dir, './data_{:03d}.nc'.format(job_idx)), encoding=encoding)

print('Finished.')