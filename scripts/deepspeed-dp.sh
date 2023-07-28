#!/usr/bin/bash

export NCCL_P2P_DISABLE=1
# export NCCL_BLOCKING_WAIT=1
# export NCCL_SOCKET_TIMEOUT=200

deepspeed \
    --num_gpus=4 \
    pretrain.py \
    --d_model=3584 \
    --n_head=32 \
    --n_block=25 \
    --start_batch=300 \
    --batch_period=50 \
    --overlap_factor=0 \
    --save_period=300 \
    --model_save_period=3000 \
    --data_path=/root/autodl-tmp/pile02-parsed \
    --ckpt=/root/autodl-tmp/myllm5-rope-flash-pile-1klen/ \
    --model_name=main
