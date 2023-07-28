import deepspeed
import os
from time import time
from torch.utils.tensorboard import SummaryWriter

from rope_model import LLM
from datasets import load_from_disk
from deepspeed.utils import RepeatingLoader

from utils import build_logger, get_args, save_ds_chkpt, prepare_tokenizer, count_parameters, load_model_chkpt
from consts import *

import random
import numpy as np
import torch

import signal

class TimeoutException(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException("Function call timed out")

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

@timeout(9)
def train_step(
    model_eng,
    input_ids, 
    target_ids
):
    loss = model_eng.forward(input_ids, target_ids)
    model_eng.backward(loss)
    model_eng.step()
    return loss

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

    base_model = LLM(
        vocab=VOCAB_SIZE,
        pad_token_id=tkn.pad_token_id,
        d_model=args.d_model,
        num_head=args.n_head,
        num_blocks=args.n_block,
        max_len=args.max_len
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

    logger.info('Preparing dataset')
    dataset = load_from_disk(args.data_path)
    logger.info('Dataset ready, total_len: %d', len(dataset))
    
    model_eng, opt, base_data_loader, _ = deepspeed.initialize(
        model=base_model,
        config=args.ds_cfg,
        training_data=dataset
    )
    micro_batch_num = len(base_data_loader)
    logger.info('Rank: %d, batch_num: %d', args.local_rank, micro_batch_num)

    rep_data_loader = RepeatingLoader(base_data_loader)
    data_iter = iter(rep_data_loader)

    if args.load_home is None and args.ckpt is not None and os.path.exists(args.ckpt):
        model_eng.load_checkpoint(args.ckpt, args.model_name)

    writer = SummaryWriter(log_dir=args.log_path)

    period_prop_time_list = []
    period_load_time_list = []
    period_loss_list = []
    stime = time()
    for ep in range(args.epochs):
        for bidx in range(micro_batch_num):
            load_start = time()
            batch = next(data_iter)
            load_time = time() - load_start
            if bidx < args.start_batch:
                continue

            period_load_time_list.append(load_time)
            x_encoded = tkn.batch_encode_plus(
                batch['text'],
                max_length=args.max_len + 1,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            base_ids, x_attn_mask = x_encoded['input_ids'], x_encoded['attention_mask']

            input_ids = base_ids[..., :-1].to(model_eng.device)
            target_ids = base_ids[..., 1:].to(model_eng.device)

            next_bidx = bidx + 1
            try:
                prop_start = time()
                loss = train_step(model_eng, input_ids, target_ids)
                prop_time = time() - prop_start
                period_prop_time_list.append(prop_time)
            except TimeoutException:
                logger.warn('ep: %d, batch: %d, train step timeout, jump to next batch.',
                             ep, next_bidx)
                model_eng.zero_grad()
                continue

            try:
                writer.add_scalar('Train Loss', loss, bidx)
            except Exception as e:
                logger.warn('batch: %d, tensorboard error: %s', bidx, e)

            period_loss_list.append(loss.item())

            if next_bidx % args.batch_period == 0:
                time_period = time() - stime
                avg_ntokens = x_attn_mask.sum() / x_attn_mask.size(0)

                avg_loss = sum(period_loss_list) / len(period_loss_list)
                avg_load_time = sum(period_load_time_list) / len(period_load_time_list)
                avg_prop_time = sum(period_prop_time_list) / len(period_prop_time_list)
                cur_lr = opt.param_groups[0]['lr']
                logger.info(
                    'ep: %d, batch: %d, local_rank: %d, time: %.2f, load_data: %.2fs, model: %.2fs,' + \
                    ' ntokens: %.2f, loss: %f, lr: %f',
                    ep, next_bidx, args.local_rank, time_period, avg_load_time, avg_prop_time, 
                    avg_ntokens, avg_loss, cur_lr
                )

                try:
                    writer.add_scalar('Learning Rate', cur_lr, bidx)
                except Exception as e:
                    logger.warn('batch: %d, tensorboard error: %s', bidx, e)
                writer.flush()

                if next_bidx % args.save_period == 0:
                    os.makedirs(args.ckpt, exist_ok=True)
                    save_ds_chkpt(str(next_bidx), model_eng,
                                  args.ckpt, args.model_name)

                period_prop_time_list = []
                period_load_time_list = []
                period_loss_list = []
                stime = time()

        save_ds_chkpt(f'ep-{ep}', model_eng, args.ckpt, args.model_name)


if __name__ == '__main__':
    args = get_args()
    # import torch
    # deepspeed.init_distributed(dist_backend='nccl')
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    # torch.cuda.set_device(args.local_rank)

    run(args=args)

