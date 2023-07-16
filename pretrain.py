import torch
import deepspeed
import os
from time import time
from torch.utils.tensorboard import SummaryWriter

from rope_model import LLM
from data_loader2 import DataLoader
from utils import build_logger, get_args, save_ds_chkpt, prepare_tokenizer


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
        overlap_factor=5,
        batch_size=args.batch_size
    )

    base_model = LLM(
        vocab=VOCAB_SIZE,
        pad_token_id=tkn.pad_token_id,
        d_model=args.d_model,
        num_head=args.n_head,
        num_block=args.n_block,
        max_len=args.max_len
    )

    if args.load_home is not None:
        load_model_path = f'{args.load_home}/{args.model_name}'
        if os.path.exists(load_model_path):
            base_model.load_state_dict(torch.load(load_model_path))

    model_eng = deepspeed.initialize(
        model=base_model,
        config=args.ds_cfg
    )[0]

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

            period_loss += loss

            next_bidx = bidx + 1
            if next_bidx % args.batch_period == 0:
                time_period = time() - stime
                avg_ntokens = x_attn_mask.sum() / x_attn_mask.size(0)
                pad_token_len = x_attn_mask.size(-1)
                logger.info(
                    'ep: %d, batch: %d, time: %.2f, ntokens: %.2f/%d, loss: %f',
                    ep, next_bidx, time_period, avg_ntokens, pad_token_len, period_loss/args.batch_period
                )
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
