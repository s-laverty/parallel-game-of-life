#!/usr/bin/env bash

#SBATCH --partition=el8
#SBATCH --gres=gpu:1
#SBATCH -t 00:01:00
#SBATCH -o debug.out

module load spectrum-mpi cuda/11.2

taskset -c 0-159:4 mpirun -N 6 ./simulation-debug brick 3
