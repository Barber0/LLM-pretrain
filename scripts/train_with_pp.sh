#!/usr/bin/bash

script_path=$(readlink -f "$0")
script_dir=$(dirname "$script_path")
proj_home_dir=$script_dir/..
cd $proj_home_dir

# export NCCL_P2P_DISABLE=1
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLE=1
# export NCCL_SOCKET_TIMEOUT_MS=10000

# export NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=2.14.3-1+cuda11.7
# export NV_LIBNCCL_DEV_PACKAGE_VERSION=2.14.3-1
# export NV_LIBNCCL_PACKAGE=libnccl2=2.14.3-1+cuda11.7
# export NV_LIBNCCL_PACKAGE_VERSION=2.14.3-1
# export NCCL_VERSION=2.14.3-1

GPUS=2

MODEL_ARGS="
    --hidden_states 3200 \
    --n_heads 32 \
    --n_layers 32 \
    --max_len 1024 \
    --ext_factor 1
"

PROG_ARGS="
    --deepspeed_cfg ./config/pipeline.json \
    --train_path /root/autodl-tmp/pile03-parsed \
    --validate_path /root/autodl-tmp/pile04-parsed \
    --tokenizer_path ./tokenizer \
    --tensorboard_path /root/tf-logs \
    --log_path ./tmp/train.log 
"

TRAIN_ARGS="
    --start_batch 160000 \
    --save_period 500 \
    --validate_period 500 \
    --replicate_period 5000 \
    --deepspeed_ckpt_tag pipe \
    --deepspeed_ckpt_home /root/autodl-tmp/sfllm-pipe \
    --torch_ckpt_home /root/autodl-tmp/sfllm-magic32 \
    --torch_ckpt_tag main-0_160000 \
    --pipeline
"

deepspeed \
    --num_gpus=$GPUS \
    train_pipeline.py \
    $PROG_ARGS \
    $MODEL_ARGS \
    $TRAIN_ARGS