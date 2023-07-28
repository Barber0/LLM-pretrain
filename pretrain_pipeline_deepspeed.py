import torch
import deepspeed
import os
from time import time
from torch.utils.tensorboard import SummaryWriter

from base_model import LLM
from data_loader3 import EfficientTextDataset, get_batch_collater
from deepspeed.utils import RepeatingLoader
from deepspeed.pipe import PipelineModule
from torch.utils.data import DataLoader
from utils import build_logger, get_args, save_ds_chkpt, prepare_tokenizer, count_parameters, load_model_chkpt
from consts import *
import json

import random
import numpy as np
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed = 168
set_random_seed(seed)


def run(args):
    logger = build_logger(
        name='pretrain',
        log_filename=args.my_log
    )

    tkn, VOCAB_SIZE = prepare_tokenizer(args.tkn_path)

    base_model,  loss_fn = LLM(
        vocab=VOCAB_SIZE,
        pad_token_id=tkn.pad_token_id,
        d_model=args.d_model,
        num_head=args.n_head,
        num_blocks=args.n_block,
        max_len=args.max_len,
    ).pipeline()

    pipe_model = PipelineModule(
        layers=base_model,
        loss_fn=loss_fn,
        num_stages=args.world_size
    )

    param_amount_b = count_parameters(base_model) * 1e-9
    logger.info('Model parameter amount: %.6f B', param_amount_b)

    if args.load_home is not None:
        load_model_chkpt(
            base_model,
            None,
            args,
            logger
        )

    model_eng, opt = deepspeed.initialize(
        model=pipe_model,
        config=args.ds_cfg,
    )[:2]

    base_loader = DataLoader(
        dataset=EfficientTextDataset(
            args.data_path, lambda line: json.loads(line)['text']),
        batch_size=model_eng.micro_batch_size,
        shuffle=True,
        num_workers=32,
        collate_fn=get_batch_collater(tkn, args.max_len)
    )
    micro_batch_num = len(base_loader)
    rep_loader = RepeatingLoader(base_loader)
    data_iter = iter(rep_loader)

    # data_loader = DIYDataLoader(args.data_name, args.data_path,
    #                             args.max_len, tkn, args.data_vendor, batch_size=args.batch_size)
    # data_iter = data_loader()

    if args.load_home is None and args.ckpt is not None and os.path.exists(args.ckpt):
        model_eng.load_checkpoint(args.ckpt, args.model_name)

    writer = SummaryWriter(log_dir=args.log_path)

    period_loss = 0
    stime = time()

    for bidx in range(micro_batch_num):
        if bidx < args.start_batch:
            continue

        loss = model_eng.train_batch(data_iter)

        try:
            writer.add_scalar('Train Loss', loss, bidx)
        except Exception as e:
            logger.warn('batch: %d, tensorboard error: %s', bidx, e)

        period_loss += loss.item()

        next_bidx = bidx + 1
        if next_bidx % args.batch_period == 0:
            time_period = time() - stime

            cur_lr = opt.param_groups[0]['lr']
            logger.info(
                'batch: %d, time: %.2f, loss: %f, lr: %f',
                next_bidx, time_period, period_loss / args.batch_period, cur_lr
            )

            try:
                writer.add_scalar('Learning Rate', cur_lr, bidx)
            except Exception as e:
                logger.warn('batch: %d, tensorboard error: %s', bidx, e)
            writer.flush()

            if next_bidx >= args.save_period and next_bidx % args.save_period == 0:
                os.makedirs(args.ckpt, exist_ok=True)
                save_ds_chkpt(str(next_bidx), model_eng,
                              args.ckpt, args.model_name)

            period_loss = 0
            stime = time()

    save_ds_chkpt(f'ep-{ep}', model_eng, args.ckpt, args.model_name)


if __name__ == '__main__':
    args = get_args()
    deepspeed.init_distributed(dist_backend='nccl')
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(args.local_rank)

    run(args=args)
