#!/usr/bin/bash

deepspeed --num_gpus=1 pretrain.py \
    --d_model=2560 \
    --n_head=32 \
    --n_block=24 \
    --batch_period=50 \
    --batch_size=20 \
    --start_batch=0 \
    --overlap_factor=0 \
    --data_path=/root/autodl-tmp/ultrachat/ \
    --ckpt=/root/autodl-tmp/chat/ \
    --load_home=/root/autodl-tmp/backup-model/ \
    --data_name=ultrachat \
    --model_name=myllm5-2B-flash-wiki-25k5 \
    --ds_cfg=ds_cfg_ft.json \
    --epochs=1
