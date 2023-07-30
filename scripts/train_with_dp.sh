#!/usr/bin/bash

script_path=$(readlink -f "$0")
script_dir=$(dirname "$script_path")
proj_home_dir=$script_dir/..
cd $proj_home_dir

export NCCL_P2P_DISABLE=1

MODEL_ARGS="
    --hidden_states 3584 \
    --n_heads 32 \
    --n_layers 25 \
    --max_len 1024 \
    --ext_factor 1
"

PROG_ARGS="
    --deepspeed_cfg ./ds_cfg_pt2.json \
    --train_path /root/autodl-tmp/pile02-parsed \
    --validate_path /root/autodl-tmp/pile00-parsed \
    --tokenizer_path ./tokenizer \
    --tensorboard_path /root/tf-logs \
    --log_path ./tmp/train.log
"

TRAIN_ARGS="
    --start_batch 29400 \
    --ckpt_home /root/autodl-tmp/myllm5-rope-flash-pile-1klen-pile02-18k/ \
    --deepspeed_ckpt_tag main
"

deepspeed \
    --num_gpus=4 \
    train.py \
    $PROG_ARGS \
    $MODEL_ARGS \
    $TRAIN_ARGS