#!/usr/bin/bash

script_path=$(readlink -f "$0")
script_dir=$(dirname "$script_path")
proj_home_dir=$script_dir/..
cd $proj_home_dir

export NCCL_P2P_DISABLE=1

GPUS=3

MODEL_ARGS="
    --hidden_states 3584 \
    --n_heads 32 \
    --n_layers 25 \
    --max_len 1024 \
    --ext_factor 1
"

PROG_ARGS="
    --deepspeed_cfg ./config/ds_cfg_pt3.json \
    --train_path /root/autodl-tmp/pile00-parsed \
    --validate_path /root/autodl-tmp/pile02-parsed \
    --tokenizer_path ./tokenizer \
    --tensorboard_path /root/tf-logs \
    --log_path ./tmp/train.log 
"

TRAIN_ARGS="
    --start_batch 0 \
    --deepspeed_ckpt_tag main \
    --deepspeed_ckpt_home /root/autodl-tmp/myllm5-rope-flash-pile-1klen-pile00 \
    --torch_ckpt_home /root/autodl-tmp/tmp-arch/ \
    --torch_ckpt_tag SFLLM-4B-29400
"

deepspeed \
    --num_gpus=$GPUS \
    train.py \
    $PROG_ARGS \
    $MODEL_ARGS \
    $TRAIN_ARGS