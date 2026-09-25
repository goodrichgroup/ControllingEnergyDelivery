#!/bin/bash
#
#-------------------------------------------------------------
#example script for running a single-CPU serial job via SLURM
#-------------------------------------------------------------
#
#SBATCH --job-name=dimers
#SBATCH --output=results-%A-%a.log
#        %A and %a are placeholders for the jobid and taskid, resp.
#
#SBATCH --array=0-199
#
#Number of CPU cores to use within one node
#SBATCH -c 1
#
#Define the number of hours the job should run.
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=120:00:00
#
#Define the amount of RAM used by your job in GigaBytes
#SBATCH --mem=2G  # total memory
#
#SBATCH --no-requeue
#
#for single-CPU jobs make sure that they use a single thread
export OMP_NUM_THREADS=1
#
#activate Python environment with JAX and JAX MD installed with all relevant packages
source /link/to/cluster/env_jax_cpu/bin/activate
#
#run the respective binary through SLURM's srun
srun --cpu_bind=verbose /usr/bin/time -v python3 /link/to/scripts/coupled_simulation_state.py ${SLURM_ARRAY_TASK_ID}
