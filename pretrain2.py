import torch
import os
from time import time
from torch.utils.tensorboard import SummaryWriter

from model import LLM
from data_loader import DataLoader
from loss_scaler import LossScaler
from torch.optim import AdamW
from utils import build_logger, get_args, save_chkpt, prepare_tokenizer

from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
)
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp.wrap import default_auto_wrap_policy
from torch.distributed import init_process_group


def run(args):
    logger = build_logger(
        name='pretrain',
        log_filename=args.my_log
    )

    tkn, VOCAB_SIZE = prepare_tokenizer(args.tkn_path)
    data_loader = DataLoader(args.data_name, args.data_path, args.batch_size)

    model = LLM(
        vocab=VOCAB_SIZE,
        pad_token_id=tkn.pad_token_id,
        d_model=args.d_model,
        num_head=args.n_head,
        num_block=args.n_block,
        max_len=args.max_len
    )

    init_process_group(backend='nccl')
    model = DistributedDataParallel(model)
    model = FullyShardedDataParallel(
        model(),
        fsdp_auto_wrap_policy=default_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
    ).cuda()

    opt = AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_home, exist_ok=True)
    if args.load_home is not None:
        load_model_path = f'{args.load_home}/{args.model_name}'
        load_opt_path = f'{args.load_home}/opt-{args.model_name}'
        if os.path.exists(load_model_path):
            model.load_state_dict(torch.load(load_model_path))
        if os.path.exists(load_opt_path):
            opt.load_state_dict(torch.load(load_opt_path))

    writer = SummaryWriter(log_dir=args.log_path)
    scaler = LossScaler(logger=logger)

    period_loss = 0
    stime = time()
    for bidx, batch in enumerate(data_loader()):
        if bidx < args.start_batch:
            continue

        x_encoded = tkn.batch_encode_plus(
            batch,
            max_length=args.max_len,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        base_ids, x_attn_mask = x_encoded['input_ids'], x_encoded['attention_mask']

        input_ids = base_ids[..., :-1]
        target_ids = base_ids[..., 1:]

        micro_batch_start = 0
        real_batch_size = len(batch)

        batch_loss = 0

        opt.zero_grad()

        num_micro_batch = args.batch_size // args.micro_batch
        base_row_idx = bidx*args.batch_size
        while micro_batch_start < real_batch_size:
            micro_batch_end = micro_batch_start + args.micro_batch
            row_idx = base_row_idx + micro_batch_end

            loss = model.forward(
                x=input_ids[micro_batch_start:micro_batch_end].cuda(),
                y=target_ids[micro_batch_start:micro_batch_end].cuda(),
                num_micro_batch=num_micro_batch
            )
            if scaler.nan_or_inf(loss):
                scaler.update(True)
                continue

            loss.backward()
            batch_loss += loss
            micro_batch_start = micro_batch_end

        opt.step()
        if scaler.step(model, opt):
            continue

        try:
            writer.add_scalar('Train Loss', batch_loss, row_idx)
        except Exception as e:
            logger.warn('batch: %d, tensorboard error: %s', bidx, e)

        period_loss += batch_loss

        next_bidx = bidx + 1
        if next_bidx % args.batch_period == 0:
            time_period = time() - stime
            avg_ntokens = x_attn_mask.sum() / x_attn_mask.size(0)
            pad_token_len = x_attn_mask.size(-1)
            logger.info(
                'batch: %d, time: %.2f, ntokens: %.2f/%d, loss: %f',
                next_bidx, time_period, avg_ntokens, pad_token_len, period_loss/args.batch_period
            )
            writer.flush()

            if next_bidx % args.save_period == 0:
                save_chkpt(
                    next_bidx,
                    model,
                    opt,
                    args,
                    logger
                )

            period_loss = 0
            stime = time()


if __name__ == '__main__':
    run(args=get_args())