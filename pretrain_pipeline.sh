#!/usr/bin/bash

python pretrain_pipeline.py \
    --data_path=/root/autodl-tmp/content/drive/MyDrive/webtext-datasets/arch/ \
    --d_model=1280 \
    --n_head=32 \
    --n_block=12 \
    --save_home=/root/autodl-tmp/test_model \
    --load_home=/root/autodl-tmp/test_model