export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --main_process_port 29531 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file deepspeed_config/stage2_no_offloading_accelerate.conf \
    fine_tuning.py \
    --model_type Special_Llama \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --use_peft \
    --use_custom_peft \
    --mode multi-round \
    --prompt_tuning_init_text "Answer questions by iteratively reasoning and retrieving knowledge" \
    --use_custom_data_collator \
    --num_virtual_tokens 100 \
    --use_slow_tokenizer \
    --train_file data/memory_reasoning/comqa/train.json \
    --eval_steps 100 \
    --val_file data/memory_reasoning/comqa/dev.json \
    --max_seq_length 1224 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 12 \
    --output_dir output/memory/comqa/AB/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens

