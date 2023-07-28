#!/usr/bin/bash

deepspeed --num_gpus=3 pretrain_pipeline_deepspeed.py \
    --d_model=2560 \
    --n_head=32 \
    --n_block=24 \
    --batch_period=50 \
    --batch_size=15 \
    --overlap_factor=0 \
    --data_path=/root/autodl-tmp/00.jsonl \
    --ckpt=/root/myllm5-fixed-flash-wt/ \
    --ds_cfg=ds_cfg_pipe.json \
    --model_name=main
