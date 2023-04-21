#!/usr/bin/env bash

nodes=(1 2 4 8)

for i in {0..3}
do
    if (( nodes[i] < 4 )); then
    sbatch -N ${nodes[i]} -o ${nodes[i]}_nodes_rpi.out job-rpi.sh
    else
    sbatch -N ${nodes[i]} -o ${nodes[i]}_nodes_rpi.out job-rpi.sh -l
    fi
done
