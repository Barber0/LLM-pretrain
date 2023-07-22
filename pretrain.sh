#!/usr/bin/bash

deepspeed --num_gpus=1 pretrain.py \
    --d_model=3840 \
    --n_head=32 \
    --n_block=22 \
    --batch_period=50 \
    --flush_period=10 \
    --max_len=450 \
    --batch_size=4 \
    --data_path=/root/wiki/ \
    --ckpt=/root/autodl-tmp/flash-llm/ \
    --model_name=main
