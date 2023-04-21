#!/usr/bin/env bash

nodes=(1 2 4 8 16)

for i in {0..3}
do
    if (( nodes[i] < 8 )); then
    sbatch -N ${nodes[i]} -o ${nodes[i]}_nodes.out job.sh
    else
    sbatch -N ${nodes[i]} -o ${nodes[i]}_nodes.out job.sh -l
    fi
done
