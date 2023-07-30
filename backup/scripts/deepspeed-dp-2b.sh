#!/usr/bin/bash

export NCCL_P2P_DISABLE=1
deepspeed \
    --num_gpus=2 \
    pretrain.py \
    --d_model=2560 \
    --n_head=32 \
    --n_block=24 \
    --batch_period=50 \
    --overlap_factor=0 \
    --start_batch=0 \
    --save_period=500 \
    --data_path=/root/autodl-tmp/pile00-parsed \
    --ckpt=/root/autodl-tmp/myllm5-rope-pile-2b/ \
    --model_name=main
