import deepspeed
import os
from time import time
from torch.utils.tensorboard import SummaryWriter

from rope_model import LLM
from datasets import load_from_disk
from deepspeed.utils import RepeatingLoader
from torch.utils.data import DataLoader

from utils import (
    build_logger, 
    get_args, 
    save_ds_chkpt, 
    prepare_tokenizer, 
    count_parameters, 
    load_model_chkpt, 
    save_model_in_fp16,
    convert_batch_to_ids,
)

from consts import *

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

def validate(model_eng, tokenizer, val_loader, args):
    model_eng.eval()
    total_loss_list = []

    vali_start = time()
    with torch.no_grad():
        for bidx in range(args.valid_batch_num):
            batch = next(val_loader)
            input_ids, target_ids, batch_avg_ntokens = convert_batch_to_ids(
                tokenizer,
                batch['text'],
                args.max_len,
                args.ext_factor,
                model_eng.device
            )

            loss = model_eng(
                input_ids, 
                target_ids,
            )
            total_loss_list.append(loss.item())

    vali_time = time() - vali_start
    average_loss = sum(total_loss_list) / len(total_loss_list)
    return average_loss, vali_time

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
        max_len=args.max_len,
        ext_factor=args.ext_factor
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
    vali_dataset = load_from_disk(args.valid_data_path).shuffle(seed=1234)
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
    
    _, get_micro_batch_size, _ = model_eng.get_batch_info()
    vali_loader = iter(RepeatingLoader(DataLoader(
        vali_dataset, 
        batch_size=get_micro_batch_size(),
        shuffle=True
    )))

    if args.load_home is None and args.ckpt is not None and os.path.exists(args.ckpt):
        model_eng.load_checkpoint(args.ckpt, args.tag_name)

    writer = SummaryWriter(log_dir=args.log_path)

    period_prop_time = 0
    period_loss = 0
    period_ntokens = 0
    stime = time()
    for ep in range(args.epochs):
        for bidx in range(micro_batch_num):
            batch = next(data_iter)
            if bidx < args.start_batch:
                continue
                
            input_ids, target_ids, batch_avg_ntokens = convert_batch_to_ids(
                tkn,
                batch['text'],
                args.max_len,
                args.ext_factor,
                model_eng.device
            )

            prop_start = time()
            try:
                model_eng.train()
                loss = model_eng.forward(input_ids, target_ids)
                model_eng.backward(loss)
                model_eng.step()
            except Exception as ex:
                logger.warn('Model propagate failed: %s', ex)
                continue
            finally:
                prop_time = time() - prop_start
                period_prop_time += prop_time

            period_loss += loss.item()
            period_ntokens += batch_avg_ntokens

            next_bidx = bidx + 1
            if args.local_rank == 0:
                try:
                    writer.add_scalar('Train Loss', loss, bidx)
                except Exception as e:
                    logger.warn('batch: %d, tensorboard error: %s', bidx, e)

                if next_bidx % args.valid_period == 0:
                    avg_vali_loss, vali_time = validate(model_eng, tkn, vali_loader, args)
                    logger.info('ep: %d, batch: %d, time: %.2f, valid_loss: %f', 
                                    ep, next_bidx, vali_time, avg_vali_loss)
                    try:
                        writer.add_scalar('Validate Loss', avg_vali_loss, bidx)
                    except Exception as e:
                        logger.warn('batch: %d, tensorboard error: %s', bidx, e)


            if next_bidx % args.batch_period == 0:
                time_period = time() - stime

                avg_ntokens = period_ntokens / args.batch_period
                avg_loss = period_loss / args.batch_period
                avg_prop_time = period_prop_time / args.batch_period
                cur_lr = opt.param_groups[0]['lr']
                logger.info(
                    'ep: %d, batch: %d, local_rank: %d, time: %.2f, model: %.2fs,' + \
                    ' ntokens: %.2f, loss: %f, lr: %f',
                    ep, next_bidx, args.local_rank, time_period, avg_prop_time, 
                    avg_ntokens, avg_loss, cur_lr
                )

                if args.local_rank == 0:
                    try:
                        writer.add_scalar('Learning Rate', cur_lr, bidx)
                    except Exception as e:
                        logger.warn('batch: %d, tensorboard error: %s', bidx, e)
                    writer.flush()

                if next_bidx % args.save_period == 0:
                    os.makedirs(args.ckpt, exist_ok=True)
                    save_ds_chkpt(str(next_bidx), model_eng,
                                  args.ckpt, args.tag_name)
                    
                if args.local_rank == 0 and next_bidx % args.model_save_period == 0:
                    os.makedirs(args.save_home, exist_ok=True)
                    save_model_in_fp16(
                        model_eng,
                        args.save_home,
                        args.model_name,
                        next_bidx
                    )                    

                period_ntokens = 0
                period_prop_time = 0
                period_loss = 0
                stime = time()

        save_ds_chkpt(f'ep-{ep}', model_eng, args.ckpt, args.tag_name)


if __name__ == '__main__':
    args = get_args()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])

    run(args=args)

