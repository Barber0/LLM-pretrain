import torch
import os
from time import time
from torch.utils.tensorboard import SummaryWriter

from rope_model import create_sequential_model
# from data_loader2 import DataLoader
from data_loader3 import EfficientTextDataset
from torch.utils.data import DataLoader
from utils import (build_logger, get_args, save_model_chkpt, prepare_tokenizer,
                   count_parameters, get_partition_balance, load_model_chkpt)
from consts import *
import fairscale.nn as fnn
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import json

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=32)
import random
import numpy as np
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed = 42
set_random_seed(seed)

def MyDataLoader(ds, batch_size, num_workers=32):
    return lambda: DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def run(args):
    logger = build_logger(
        name='pretrain',
        log_filename=args.my_log
    )

    logger.info('Preparing dataset...')
    tkn, VOCAB_SIZE = prepare_tokenizer(args.tkn_path)
    ds = EfficientTextDataset(args.data_path, lambda line: json.loads(line)['text'])
    data_loader = MyDataLoader(ds, args.batch_size)
    logger.info('Dataset is ready.')

    base_model,  num_layers = create_sequential_model(
        vocab=VOCAB_SIZE,
        d_model=args.d_model,
        num_head=args.n_head,
        num_blocks=args.n_block,
        max_len=args.max_len
    )

    balance = get_partition_balance(num_layers)

    pipe_model = fnn.Pipe(
        base_model,
        balance
    )

    opt = torch.optim.AdamW(pipe_model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tkn.pad_token_id)

    param_amount_b = count_parameters(pipe_model) * 1e-9
    logger.info('Model parameter amount: %.6f B', param_amount_b)

    if args.load_home is not None:
        load_model_chkpt(pipe_model, opt, args, logger)

    writer = SummaryWriter(log_dir=args.log_path)
    scaler = GradScaler()

    period_loss = 0
    stime = time()
    for ep in range(args.epochs):
        for bidx, batch in enumerate(data_loader()):
            if bidx < args.start_batch:
                continue

            x_encoded = tkn.batch_encode_plus(
                batch,
                max_length=args.max_len + 1,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            base_ids, x_attn_mask = x_encoded['input_ids'], x_encoded['attention_mask']

            ipt_device = pipe_model.devices[0]
            input_ids = base_ids[..., :-1].to(ipt_device)
            target_ids = base_ids[..., 1:].to(ipt_device)

            with autocast(dtype=torch.float16):
                y_pred = pipe_model.forward(input_ids)
                loss = loss_fn(
                    y_pred.to(ipt_device).contiguous(
                    ).view(-1, y_pred.size(-1)),
                    target_ids.contiguous().view(-1)
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            try:
                writer.add_scalar('Train Loss', loss, bidx)
            except Exception as e:
                logger.warn('batch: %d, tensorboard error: %s', bidx, e)

            period_loss += loss

            next_bidx = bidx + 1
            if next_bidx % args.batch_period == 0:
                time_period = time() - stime
                avg_ntokens = x_attn_mask.sum() / x_attn_mask.size(0)
                pad_token_len = x_attn_mask.size(-1)

                cur_lr = opt.param_groups[0]['lr']
                logger.info(
                    'ep: %d, batch: %d, time: %.2f, ntokens: %.2f/%d, loss: %f, lr: %f',
                    ep, next_bidx, time_period, avg_ntokens, pad_token_len, period_loss /
                    args.batch_period, cur_lr
                )

                try:
                    writer.add_scalar('Learning Rate', cur_lr, bidx)
                except Exception as e:
                    logger.warn('batch: %d, tensorboard error: %s', bidx, e)
                writer.flush()

                if next_bidx % args.save_period == 0:
                    os.makedirs(args.ckpt, exist_ok=True)
                    save_model_chkpt(
                        str(next_bidx),
                        pipe_model,
                        opt,
                        args.ckpt,
                        logger
                    )

                period_loss = 0
                stime = time()

        save_model_chkpt(
            f'ep-{ep}',
            pipe_model,
            opt,
            args.ckpt,
            logger
        )


if __name__ == '__main__':
    run(args=get_args())
