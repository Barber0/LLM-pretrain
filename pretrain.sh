#!/usr/bin/bash

deepspeed --num_gpus=1 pretrain.py \
    --d_model=512 \
    --n_head=32 \
    --n_block=12 \
    --data_path=/root/autodl-tmp/content/drive/MyDrive/webtext-datasets/arch/ \
    --ckpt=/root/autodl-tmp/rope-llm2/