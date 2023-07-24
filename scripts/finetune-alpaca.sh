#!/usr/bin/bash

deepspeed --num_gpus=1 pretrain.py \
    --d_model=2560 \
    --n_head=32 \
    --n_block=24 \
    --batch_period=50 \
    --batch_size=20 \
    --overlap_factor=0 \
    --data_name=alpaca \
    --data_path=YorickHe/alpaca_data_cleaned \
    --data_vendor=ms \
    --ckpt=/root/autodl-tmp/ft-alpaca/ \
    --load_home=/root/autodl-tmp/myllm5-fixed-flash-wiki-72k/main/ \
    --model_name=main \
    --ds_cfg=ds_cfg_ft.json \
    --epochs=1
