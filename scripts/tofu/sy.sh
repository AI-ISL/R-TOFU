#!/bin/bash
#SBATCH --job-name=pepe
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

MASTER_PORT=$((RANDOM % 50001 + 10000))

forget_losses=(
    # FINETUNE
    RETRAIN
)

# You can specify any forget task from 1 to 10
# the standard TOFU benchmark is task 1
task_list=(1)

# pass to python script
export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

learning_rates=(
    1e-5
)

model_path=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
mask=true

use_LoRA=false
save_root=results/tofu

forget_coeff=1.0
regularization_coeff=1.0

save_checkpoint=false

num_epochss=(5)


save_steps=last
eval_steps=(last)

# split=forget01 # forget01/forget05/forget10
# for forget_loss in ${forget_losses[@]}; do
#     for num_epochs in ${num_epochss[@]}; do
#         for lr in ${learning_rates[@]}; do
#             for task_id in ${task_list[@]}; do
#                 COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
#                     mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint"
#                 srun --gres=gpu:4 --ntasks=1 --cpus-per-task=16 \ 
#                 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
#                         forget.py \
#                         --config-name=tofu.yaml \
#                         task_id=$task_id \
#                         save_steps=$save_steps \
#                         $COMMON
                
#                 for step in ${eval_steps[@]}; do
#                     srun --gres=gpu:1 --ntasks=1 --cpus-per-task=8 \ 
#                     torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
#                             eval.py \
#                             --config-name=tofu.yaml \
#                             task_id=$task_id \
#                             eval_unlearn_step=$step \
#                             $COMMON
#                 done
#             done
#         done
#     done
# done

split=forget100 # forget01/forget05/forget10
for forget_loss in ${forget_losses[@]}; do
    for num_epochs in ${num_epochss[@]}; do
        for lr in ${learning_rates[@]}; do
            for task_id in ${task_list[@]}; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint"
                srun --gres=gpu:4 --ntasks=1 --cpus-per-task=16 \ 
                torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
                        forget.py \
                        --config-name=tofu.yaml \
                        task_id=$task_id \
                        save_steps=$save_steps \
                        $COMMON
            done
        done
    done
done

# split=forget01 # forget01/forget05/forget10
# for forget_loss in ${forget_losses[@]}; do
#     for num_epochs in ${num_epochss[@]}; do
#         for lr in ${learning_rates[@]}; do
#             for task_id in ${task_list[@]}; do
#                 COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
#                     mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint"
                
#                 for step in ${eval_steps[@]}; do
#                     srun --gres=gpu:1 --ntasks=1 --cpus-per-task=8 \ 
#                     torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
#                             eval2.py \
#                             --config-name=tofu.yaml \
#                             task_id=$task_id \
#                             eval_unlearn_step=$step \
#                             $COMMON
#                 done
#             done
#         done
#     done
# done