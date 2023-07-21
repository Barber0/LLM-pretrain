#!/usr/bin/bash

deepspeed --num_gpus=1 pretrain.py \
    --d_model=2560 \
    --n_head=32 \
    --n_block=24 \
    --batch_period=50 \
    --batch_size=6 \
    --data_path=/root/autodl-tmp/wiki/ \
    --ckpt=/root/autodl-tmp/rope-llm/ \
    --model_name=main
