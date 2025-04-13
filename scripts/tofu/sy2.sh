#!/bin/bash
#SBATCH --job-name=pepepe
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

MASTER_PORT=$((RANDOM % 50001 + 10000))

forget_losses=(
    # GA3+GD3
    # GA1+GD3
    # GA2+GD3
    # GA3+GD1
    # GA3+GD2
    # SDK+GD1
    # SDK+GD2
    SDK+GD3
    NPO3+GD3
)

# You can specify any forget task from 1 to 10
# the standard TOFU benchmark is task 1
task_list=(1)

# pass to python script
export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

learning_rates=(
    1e-5
)

model_path=sangyon/LRM-unlearning-target
mask=true

use_LoRA=false
save_root=results/steps

forget_coeff=1.0
regularization_coeff=1.0

save_checkpoint=false
num_epochss=(1 2 3 4 5 6 7 8 9 10)


save_steps=last
eval_steps=(last)


split=forget01
for forget_loss in ${forget_losses[@]}; do
    for num_epochs in ${num_epochss[@]}; do
        for lr in ${learning_rates[@]}; do
            for task_id in ${task_list[@]}; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                srun --gres=gpu:2 --ntasks=1 --cpus-per-task=16 \
                torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                        forget.py \
                        --config-name=tofu.yaml \
                        task_id=$task_id \
                        save_steps=$save_steps \
                        $COMMON
            done
            for step in ${eval_steps[@]}; do
                srun --gres=gpu:1 --ntasks=1 --cpus-per-task=16 \
                torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                        eval.py \
                        --config-name=tofu.yaml \
                        task_id=$task_id \
                        eval_unlearn_step=$step \
                        $COMMON
            done
            for step in ${eval_steps[@]}; do
                srun --gres=gpu:1 --ntasks=1 --cpus-per-task=16 \ 
                torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                        eval2.py \
                        --config-name=tofu.yaml \
                        task_id=$task_id \
                        eval_unlearn_step=$step \
                        $COMMON
            done
        done
    done
done

# for forget_loss in ${forget_losses[@]}; do
#     for num_epochs in ${num_epochss[@]}; do
#         for lr in ${learning_rates[@]}; do
#             for task_id in ${task_list[@]}; do
#                 COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
#                     mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"

#                 for step in ${eval_steps[@]}; do
#                     srun --gres=gpu:1 --ntasks=1 --cpus-per-task=16 \ 
#                     torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
#                             eval.py \
#                             --config-name=tofu.yaml \
#                             task_id=$task_id \
#                             eval_unlearn_step=$step \
#                             $COMMON
#                 done
#                 for step in ${eval_steps[@]}; do
#                     srun --gres=gpu:1 --ntasks=1 --cpus-per-task=16 \ 
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
