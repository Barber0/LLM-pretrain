import torch
import deepspeed
import os
from time import time
from torch.utils.tensorboard import SummaryWriter

# from rope_model import LLM
from base_model import MyModel
from data_loader2 import DataLoader
from utils import build_logger, get_args, save_ds_chkpt, prepare_tokenizer, count_parameters, load_model_chkpt
from consts import *


def run(args):
    logger = build_logger(
        name='pretrain',
        log_filename=args.my_log
    )

    tkn, VOCAB_SIZE = prepare_tokenizer(args.tkn_path)
    data_loader = DataLoader(
        ds_name=args.data_name,
        ds_path=args.data_path,
        max_len=args.max_len,
        overlap_factor=args.overlap_factor,
        batch_size=args.batch_size
    )

    base_model = MyModel(
        vocab=VOCAB_SIZE,
        pad_token_id=tkn.pad_token_id,
        d_model=args.d_model,
        num_head=args.n_head,
        num_block=args.n_block,
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

    model_eng, opt = deepspeed.initialize(
        model=base_model,
        config=args.ds_cfg
    )[:2]

    if args.load_home is None and args.ckpt is not None and os.path.exists(args.ckpt):
        model_eng.load_checkpoint(args.ckpt, args.model_name)

    writer = SummaryWriter(log_dir=args.log_path)

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

            input_ids = base_ids[..., :-1].cuda()
            target_ids = base_ids[..., 1:].cuda()

            loss = model_eng.forward(input_ids, target_ids)
            model_eng.backward(loss)
            model_eng.step()

            try:
                writer.add_scalar('Train Loss', loss, bidx)
            except Exception as e:
                logger.warn('batch: %d, tensorboard error: %s', bidx, e)

            period_loss += loss.item()

            next_bidx = bidx + 1
            if next_bidx % args.batch_period == 0:
                time_period = time() - stime
                avg_ntokens = x_attn_mask.sum() / x_attn_mask.size(0)

                cur_lr = opt.param_groups[0]['lr']
                logger.info(
                    'ep: %d, batch: %d, time: %.2f, ntokens: %.2f, loss: %f, lr: %f',
                    ep, next_bidx, time_period, avg_ntokens, period_loss /
                    args.batch_period, cur_lr
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
    run(args=get_args())
