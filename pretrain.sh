#!/usr/bin/bash

torchrun --nproc_per_node=2 --master-port=29501 pretrain2.py --num_nodes=2