#!/bin/bash
#SBATCH --job-name=pp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

MASTER_PORT=$((RANDOM % 50001 + 10000))


export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

srun --gres=gpu:1 --ntasks=1 --cpus-per-task=1 \ 
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
test_cot.py \
