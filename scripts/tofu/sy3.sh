MASTER_PORT=$((RANDOM % 50001 + 10000))

forget_losses=(
    # IDK3+GD3
    # IDK1+GD1
    # IDK2+GD2
    GA1+GD1
    GA2+GD2
    GA3+GD3
)

task_list=(1)

export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

learning_rates=(
    1e-5
)

model_path=results/tofu/llama3-8b/forget01/FINETUNE/seed_1001/epoch5_1e-05_FixRef_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last
mask=true

use_LoRA=false
save_root=results/5

forget_coeff=1.0
regularization_coeff=1.0

save_checkpoint=false

num_epochss=(5)

save_steps=last
eval_steps=(last)

split=forget01 # forget01/forget05/forget10

for forget_loss in "${forget_losses[@]}"; do
    for num_epochs in "${num_epochss[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for task_id in "${task_list[@]}"; do
                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                    mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint model_path=$model_path"
                
                for step in "${eval_steps[@]}"; do
                    CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                        eval2.py \
                        --config-name=tofu.yaml \
                        task_id=$task_id \
                        eval_unlearn_step=$step \
                        $COMMON
                done
            done
        done
    done
done
