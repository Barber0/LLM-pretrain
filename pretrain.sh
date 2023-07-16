#!/usr/bin/bash

# torchrun --nproc_per_node=2 --master-port=29501 pretrain2.py --num_nodes=2
deepspeed --num_gpus=1 pretrain.py \
    --d_model=2560 \
    --n_head=32 \
    --n_block=24 \
    --data_path=/root/autodl-tmp/wiki/ \
    --ckpt=/root/autodl-tmp/rope-llm/ \
    --model_name=main