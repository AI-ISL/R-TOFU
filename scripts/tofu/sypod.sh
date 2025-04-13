#!/bin/bash
MASTER_PORT=$((RANDOM % 50001 + 10000))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

forget_losses=(
    # GA1+GD1
    # GA2+GD2
    # GA3+GD3
    # NPO1+GD1
    # NPO2+GD2
    # NPO3+GD3
    # GA1+KL1
    # GA2+KL2
    # GA3+KL3
    # NPO1+KL1
    # NPO2+KL2
    # NPO3+KL3
    # GA1
    # GA2
    # GA3
    # NPO1
    # NPO2
    # NPO3
    # IDK2
    # IDK1
    # IDK3
    IDK2+GD2
    IDK1+GD1
    IDK3+GD3
    # IDK2+KL2
    # IDK1+KL1
    # IDK3+KL3
)
cuda_id=2


task_list=(1)

learning_rates=(
    1e-5
)
task_list=(1)

# pass to python script
export TASK_LIST=$(IFS=,; echo "${task_list[*]}")
model_path=sangyon/LRM-unlearning-target
mask=true

use_LoRA=false
save_root=results/steps

forget_coeff=1.0
regularization_coeff=1.0

save_checkpoint=false


save_steps=last
eval_steps=(last)

num_epochss=(1 2 3 4 5 6 7 8)
split=forget01

for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            done
            for step in "${eval_steps[@]}"; do
                CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    eval.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    eval_unlearn_step=$step \
                    $COMMON
            done
            # for step in "${eval_steps[@]}"; do
            #     CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
            #         eval2.py \
            #         --config-name=tofu.yaml \
            #         task_id=$task_id \
            #         eval_unlearn_step=$step \
            #         $COMMON
            # done
        done
    done
done

split=forget05
num_epochss=(1 2 3 4 5)
for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            done
            for step in "${eval_steps[@]}"; do
                CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    eval.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    eval_unlearn_step=$step \
                    $COMMON
            done
            # for step in "${eval_steps[@]}"; do
            #     CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
            #         eval2.py \
            #         --config-name=tofu.yaml \
            #         task_id=$task_id \
            #         eval_unlearn_step=$step \
            #         $COMMON
            # done
        done
    done
done

split=forget10
for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            done
            for step in "${eval_steps[@]}"; do
                CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    eval.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    eval_unlearn_step=$step \
                    $COMMON
            done
            # for step in "${eval_steps[@]}"; do
            #     CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
            #         eval2.py \
            #         --config-name=tofu.yaml \
            #         task_id=$task_id \
            #         eval_unlearn_step=$step \
            #         $COMMON
            # done
        done
    done
done