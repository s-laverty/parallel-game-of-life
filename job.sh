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
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s s -o small.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s m -o medium.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s l -o large.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s s -w -o small-weak.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s m -w -o medium-weak.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s l -w -o large-weak.grid striped 1
if [[ $xl = true ]]; then
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s xl -o x-large.grid striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -i acorn -s xl -w -o x-large-weak.grid striped 1
fi

# Striped tests
taskset -c 0-159:4 mpirun -N 6 simulation -l small.grid -s s striped 1000
taskset -c 0-159:4 mpirun -N 6 simulation -l medium.grid -s m striped 100
taskset -c 0-159:4 mpirun -N 6 simulation -l large.grid -s l striped 10
taskset -c 0-159:4 mpirun -N 6 simulation -l small-weak.grid -s s -w striped 1000
taskset -c 0-159:4 mpirun -N 6 simulation -l medium-weak.grid -s m -w striped 100
taskset -c 0-159:4 mpirun -N 6 simulation -l large-weak.grid -s l -w striped 10

# Brick tests
taskset -c 0-159:4 mpirun -N 6 simulation -l small.grid -s s brick 1000
taskset -c 0-159:4 mpirun -N 6 simulation -l medium.grid -s m brick 100
taskset -c 0-159:4 mpirun -N 6 simulation -l large.grid -s l brick 10
taskset -c 0-159:4 mpirun -N 6 simulation -l small-weak.grid -s s -w brick 1000
taskset -c 0-159:4 mpirun -N 6 simulation -l medium-weak.grid -s m -w brick 100
taskset -c 0-159:4 mpirun -N 6 simulation -l large-weak.grid -s l -w brick 10

# xl tests
if [[ $xl = true ]]; then
taskset -c 0-159:4 mpirun -N 6 simulation -l x-large.grid -s xl striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -l x-large.grid -s xl brick 1
taskset -c 0-159:4 mpirun -N 6 simulation -l x-large-weak.grid -s xl -w striped 1
taskset -c 0-159:4 mpirun -N 6 simulation -l x-large-weak.grid -s xl -w brick 1
fi
