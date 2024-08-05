#!/bin/bash

#SBATCH --job-name=RNADiffusion
#SBATCH --nodes=1
#SBATCH --partition=gpu-1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=1-00:00:00


# Alternatively, partition can be set as follows: gpu-1 gpu-2 gpu-a100-Partition

sh $HOME/Projects/RNADiffusion/run.sh