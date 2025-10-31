#!/bin/bash
# example for 2 nodes with 4 GPUs each
# fill in slurm details first

nodes=2
gpus_per_node=4
ntasks=$((nodes * gpus_per_node))
srun \
    --account= \
    --partition= \
    --qos= \
    --nodes=$nodes \
    --ntasks=$ntasks \
    --cpus-per-task=4 \
    --gpus=$ntasks \
    --ntasks-per-node=4 \
    ./nccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 1