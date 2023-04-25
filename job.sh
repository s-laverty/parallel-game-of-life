#!/usr/bin/env bash

#SBATCH --partition=el8
#SBATCH --gres=gpu:6
#SBATCH -t 0:30:00

while getopts "l" o; do
    case "$o" in
        l)
            xl=true ;;
        ?)
            echo Unrecognized optional param ;;
    esac
done

module load spectrum-mpi

# Create checkpoint files and test output speed
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s s -o small_$SLURM_JOB_NUM_NODES.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s m -o medium_$SLURM_JOB_NUM_NODES.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s l -o large_$SLURM_JOB_NUM_NODES.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s s -w -o small-weak_$SLURM_JOB_NUM_NODES.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s m -w -o medium-weak_$SLURM_JOB_NUM_NODES.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s l -w -o large-weak_$SLURM_JOB_NUM_NODES.grid striped 1
if [[ $xl = true ]]; then
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s xl -o x-large_$SLURM_JOB_NUM_NODES.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s xl -w -o x-large-weak_$SLURM_JOB_NUM_NODES.grid striped 1
fi

# Striped tests
taskset -c 0-159:4 mpirun -N 6 simulation -l small_$SLURM_JOB_NUM_NODES.grid -s s striped 1000
taskset -c 0-159:4 mpirun -N 6 simulation -l medium_$SLURM_JOB_NUM_NODES.grid -s m striped 100
taskset -c 0-159:4 mpirun -N 6 simulation -l large_$SLURM_JOB_NUM_NODES.grid -s l striped 10
taskset -c 0-159:4 mpirun -N 6 simulation -l small-weak_$SLURM_JOB_NUM_NODES.grid -s s -w striped 1000
taskset -c 0-159:4 mpirun -N 6 simulation -l medium-weak_$SLURM_JOB_NUM_NODES.grid -s m -w striped 100
taskset -c 0-159:4 mpirun -N 6 simulation -l large-weak_$SLURM_JOB_NUM_NODES.grid -s l -w striped 10

# Brick tests
taskset -c 0-159:4 mpirun -N 6 simulation -l small_$SLURM_JOB_NUM_NODES.grid -s s brick 1000
taskset -c 0-159:4 mpirun -N 6 simulation -l medium_$SLURM_JOB_NUM_NODES.grid -s m brick 100
taskset -c 0-159:4 mpirun -N 6 simulation -l large_$SLURM_JOB_NUM_NODES.grid -s l brick 10
taskset -c 0-159:4 mpirun -N 6 simulation -l small-weak_$SLURM_JOB_NUM_NODES.grid -s s -w brick 1000
taskset -c 0-159:4 mpirun -N 6 simulation -l medium-weak_$SLURM_JOB_NUM_NODES.grid -s m -w brick 100
taskset -c 0-159:4 mpirun -N 6 simulation -l large-weak_$SLURM_JOB_NUM_NODES.grid -s l -w brick 10

# xl tests
if [[ $xl = true ]]; then
taskset -c 0-159:4 mpirun -N 6 simulation -l x-large_$SLURM_JOB_NUM_NODES.grid -s xl striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -l x-large_$SLURM_JOB_NUM_NODES.grid -s xl brick 1
taskset -c 0-159:4 mpirun -N 6 simulation -l x-large-weak_$SLURM_JOB_NUM_NODES.grid -s xl -w striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -l x-large-weak_$SLURM_JOB_NUM_NODES.grid -s xl -w brick 1
fi
