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

# Striped tests
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s s striped 1000
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s m striped 100
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s l striped 10

# Brick tests
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s s brick 1000
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s m brick 100
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s l brick 10

# xl tests
if [[ $xl = true ]]; then
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s xl brick 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s xl striped 1
fi
