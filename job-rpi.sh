#!/usr/bin/env bash

#SBATCH --partition=el8-rpi
#SBATCH --gres=gpu:4
#SBATCH -t 0:30:00

while getopts "l" o; do
    case "$o" in
        l)
            l=true ;;
        ?)
            echo Unrecognized optional param ;;
    esac
done

module load spectrum-mpi

# Striped tests
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s s striped 1000
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s m striped 100

# Brick tests
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s s brick 1000
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s m brick 100

# Large tests
if [ $l = true ]; then
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s l striped 10
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s l brick 10
fi
