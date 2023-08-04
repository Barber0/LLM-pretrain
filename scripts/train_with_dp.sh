#!/usr/bin/bash

script_path=$(readlink -f "$0")
script_dir=$(dirname "$script_path")
proj_home_dir=$script_dir/..
cd $proj_home_dir

# export NCCL_P2P_DISABLE=1

GPUS=2

MODEL_ARGS="
    --hidden_states 3584 \
    --n_heads 32 \
    --n_layers 25 \
    --max_len 1024 \
    --ext_factor 1 
"

PROG_ARGS="
    --deepspeed_cfg ./config/ds_cfg_finetune.json \
    --train_path /root/autodl-tmp/orca_train \
    --validate_path /root/autodl-tmp/orca_vali \
    --tokenizer_path ./tokenizer \
    --tensorboard_path /root/tf-logs \
    --log_path ./tmp/train.log 
"

TRAIN_ARGS="
    --start_batch 0 \
    --validate_batch_num 30 \
    --save_period 500 \
    --validate_period 500 \
    --replicate_period 5000 \
    --deepspeed_ckpt_tag main \
    --deepspeed_ckpt_home /root/autodl-tmp/sfllm-4B-finetune2 \
    --torch_ckpt_home /root/autodl-tmp/sfllm-4B/main \
    --torch_ckpt_tag sfllm-4B-pile03-68k
"

deepspeed \
    --num_gpus=$GPUS \
    train.py \
    $PROG_ARGS \
    $MODEL_ARGS \
    $TRAIN_ARGS