#!/usr/bin/env python3
print('Start running Brownian dynamics simulations of a system with one Machine.')

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
from mdl_model import setup_energy_individual
from mdl_brownian import brownian, BrownianState

###### Directory, parameters, and global options ######

job_idx = jnp.int64(sys.argv[1])          # from job_array
seed = 321 * job_idx + 7654               # used to generate PRNGKey
key = random.key(seed)  # PRNGKey

data_dir = (r'/link/to/directory/')
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
file = os.path.join(data_dir, 'params.pickle')
with open(file, 'rb') as f:
   params = pickle.load(f)

# global parameters and options
global_options = {
    # Brownian
    'kT': 1.0,                       # thermal energy, specifying the temperature in units of Boltzmann constant; beta=1
    'gamma': 0.1,                    # friction coefficient between the particles and the solvent
    'dt': 1e-5,                      # timescale (step size) of Brownian dynamics simulation; small enough to resolve all necessary forces
    'num_total_steps': 2e10,         # Total number of MD steps
    'apply_N_steps': 1e5,            # apply N steps in inner loop of MD simulation
    'save_every': 1e4,               # save trajectory (length) every x steps
    'key_reseed': 1e7,               # reseed random keys every x steps
    'side_length': 8.,               # sidelength specifying the size of each side of the periodic box (hypercube) for periodic boundary conditions
    'species_numbers': [0, 1],       # amount of [Sources, Machines] in system
}

if job_idx == 1:
    file = os.path.join(data_dir, 'global_options.pickle')
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

## Energy
get_energy_fn = setup_energy_individual(displacement, mac=True)
energy_fn = get_energy_fn(params)

## Get initial locations of Sources, Machine
N_src, N_mac = global_options['species_numbers']
N = 2 * (N_src + N_mac)
key, subkey = random.split(key)

## Prepare one Machine/Source in closed state
R_ini = random.uniform(subkey, (N, 3), dtype=jnp.float64, maxval=global_options['side_length'])
key, subkey = random.split(key)
v = random.uniform(subkey, (3,), dtype=jnp.float64, maxval=1)
v_hat = v * params['mac_sigma0'] / jnp.linalg.norm(v)
R_ini = R_ini.at[1].set(shift(R_ini[0], v_hat))

### Run a brownian dynamics simulation and save periodic snapshots

def update_state(m, s):
    # update Machine state variable from current Machine length m and current state s.
    is_upd = jnp.where(s == 0, m >= params['mac_sigma1'], m >= params['mac_sigma0'])
    return jnp.where(is_upd, 1, 0)

def update_trajectory(trajectory, time_md, m):
    return trajectory.at[jnp.int64(time_md / global_options['save_every'] - 1)].set(m)

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
    (state_cur, s_cur, times_trans_cur, n_trans_cur, time_cur, trajectory_cur) = data
    state_upd = apply(state_cur)
    time_upd = time_cur + 1

    state_upd = reseed_key(state_upd, time_upd)

    # 2. Update Machine state and running variables
    m_upd = space.distance(displacement(state_upd.position[0], state_upd.position[1]))
    s_upd = update_state(m_upd, s_cur)
    times_trans_upd = jnp.where(s_upd == s_cur, times_trans_cur, times_trans_cur.at[n_trans_cur].set(time_upd))
    n_trans_upd = jnp.where(s_upd == s_cur, n_trans_cur, n_trans_cur + 1)
    trajectory_upd = jnp.where(time_upd % global_options['save_every'] == 0, update_trajectory(trajectory_cur, time_upd, m_upd), trajectory_cur)

    return (state_upd, s_upd, times_trans_upd, n_trans_upd, time_upd, trajectory_upd), 0

@jit
def apply_N_steps(data, t):
    # Apply N steps
    data, _ = lax.scan(apply_single_step, data, length=int(global_options['apply_N_steps']))
    return data, 0

## 1. Running variables for certain quantities
time_ini = 0                                         # running variable for current simulation time
n_trans_ini = 0                                      # running variable for current number of Machine transition
s_ini = 0                                            # running variable for Machine state: 0/1
Ntrans = 10000                                       # default number of Machine transitions to save, N_machine_cycles = Ntrans - 2
times_trans_ini = -1 * jnp.ones(Ntrans, dtype=int)   # running variable for time points of Machine transitions
trajectory_ini = jnp.zeros(Ntrans)  # int(global_options['num_total_steps'] / global_options['save_every'])
# even = transition closed -> open, odd = transition open -> closed
# If there are more than Ntrans transitions they will not be added to times_trans.
# if there are less, we remove the empty lines from the array before saving data

## 2. Define the simulation
init, apply = brownian(energy_fn, shift, global_options['dt'], global_options['kT'], global_options['gamma'])
apply = jit(apply)

## 3. Run MD
key, split = random.split(key)
state_ini = init(split, R_ini)
data_ini = (state_ini, s_ini, times_trans_ini, n_trans_ini, time_ini, trajectory_ini)

data_fin, _ = lax.scan(apply_N_steps, data_ini, length=int(global_options['num_total_steps'] / global_options['apply_N_steps']))
(state_fin, s_fin, times_trans_fin, n_trans_fin, time_fin, trajectory_fin) = data_fin

# Save data
outdata = xr.Dataset({
    's_final': xr.DataArray(s_fin),
    'n_trans': xr.DataArray(n_trans_fin),
    'times_trans': xr.DataArray(times_trans_fin[:n_trans_fin], dims=('num_trans',)),
    # 'trajectory': xr.DataArray(trajectory_fin, dims=('num_traj',)),
})
# Compress data
comp = {'zlib': True, 'complevel': 5}
encoding = {var: comp for var in outdata.data_vars}
outdata.to_netcdf(os.path.join(data_dir, './data_{:03d}.nc'.format(job_idx)), encoding=encoding)

print('Finished.')
