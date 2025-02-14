export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --main_process_port 29501 \
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
    --output_order A \
    --use_custom_data_collator \
    --use_slow_tokenizer \
    --train_file data/compress_QA/hotpotqa/train.json \
    --eval_steps 400 \
    --val_file data/compress_QA/hotpotqa/dev.json \
    --max_seq_length 1524 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir output/hotpotqa/A/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens
