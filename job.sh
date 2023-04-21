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
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s m striped 1000
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s l striped 1000
if [ $xl = true ]; then
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s xl striped 1000
fi

# Brick tests
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s s brick 1000
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s m brick 1000
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s l brick 1000
if [ $xl = true ]; then
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s xl brick 1000
fi
