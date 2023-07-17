#!/usr/bin/bash

DATA_HOME=/root/autodl-tmp/wiki
SAVE_HOME=/root/autodl-tmp/test_model

python pretrain_pipeline.py \
    --data_path=$DATA_HOME \
    --d_model=2560 \
    --n_head=32 \
    --n_block=24 \
    --save_home=$SAVE_HOME \
    --load_home=$SAVE_HOME