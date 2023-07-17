import logging
from argparse import ArgumentParser
import os
import torch
from transformers import GPT2Tokenizer
from consts import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def build_logger(
    name,
    log_filename,
    level=logging.INFO,
    str_format='%(asctime)s [%(levelname)s] %(message)s'
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(str_format)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(logging.FileHandler(log_filename))
    return logger


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--model_name', default='myllm4-model')

    arg_parser.add_argument('--ds_cfg', default='./ds_cfg.json')

    arg_parser.add_argument('--load_home', default=None)
    arg_parser.add_argument('--save_home', default=None)
    arg_parser.add_argument('--ckpt', default=None)

    arg_parser.add_argument('--data_name', default='openwebtext')
    arg_parser.add_argument(
        '--data_path', default='/root/autodl-tmp/content/drive/MyDrive/webtext-datasets/arch/')

    arg_parser.add_argument('--tkn_path', default='./tokenizer')
    arg_parser.add_argument('--log_path', default='/root/tf-logs')
    arg_parser.add_argument('--my_log', default='./tmp/train.log')

    arg_parser.add_argument('--start_batch', default=0, type=int)
    arg_parser.add_argument('--batch_period', default=20, type=int)
    arg_parser.add_argument('--save_period', default=500, type=int)

    arg_parser.add_argument('--epochs', default=2, type=int)
    arg_parser.add_argument('--batch_size', default=10, type=int)

    arg_parser.add_argument('--max_len', default=512, type=int)
    arg_parser.add_argument('--d_model', default=512, type=int)
    arg_parser.add_argument('--n_head', default=32, type=int)
    arg_parser.add_argument('--n_block', default=12, type=int)
    arg_parser.add_argument('--local_rank', default=0)

    args = arg_parser.parse_args()
    return args


def save_ds_chkpt(
    name,
    model_eng,
    ckpt_path,
    md_tag
):
    model_eng.save_checkpoint(ckpt_path, tag=md_tag)
    with open(f'{ckpt_path}/progress.txt', 'w') as f:
        f.write(name)


def save_model_chkpt(
    bidx,
    model,
    opt,
    args,
    logger
):
    try:
        if args.save_home is not None and os.path.exists(args.save_home):
            torch.save(model.state_dict(),
                       f'{args.save_home}/{args.model_name}.pt')
            torch.save(opt.state_dict(),
                       f'{args.save_home}/opt-{args.model_name}.pt')

            with open(f'{args.save_home}/progress.txt', 'w') as f:
                f.write(str(bidx))
    except Exception as ex:
        logger.warn('batch: %d, save state failed: %s', bidx, ex)


def load_model_chkpt(
    model,
    opt,
    args,
    logger
):
    try:
        model_path = f'{args.load_home}/{args.model_name}.pt'
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            if CKPT_MODEL_KEY in ckpt:
                model.load_state_dict(ckpt[CKPT_MODEL_KEY])
            else:
                model.load_state_dict(ckpt)
            logger.info('Load model state: %s', model_path)
        else:
            logger.warn('Model state not found: %s', model_path)

        if opt is None:
            return
        opt_path = f'{args.load_home}/opt-{args.model_name}.pt'
        if os.path.exists(opt_path):
            ckpt = torch.load(opt_path)
            opt.load_state_dict(ckpt)
            logger.info('Load optimizer state: %s', opt_path)
        else:
            logger.warn('Optimizer state not found: %s', opt_path)
    except Exception as ex:
        logger.warn('Load state failed: %s', ex)


def prepare_tokenizer(tkn_path, added_tokens=[END_SIGN]):
    tkn = GPT2Tokenizer.from_pretrained(tkn_path)
    tkn.pad_token = '[PAD]'
    tkn.add_tokens(added_tokens)
    VOCAB_SIZE = tkn.vocab_size + len(added_tokens)
    return tkn, VOCAB_SIZE


def get_partition_balance(num_layers):
    partitions = torch.cuda.device_count()
    avg_num_layers = num_layers//partitions
    balance = [avg_num_layers for _ in range(partitions)]
    balance[-1] += num_layers % partitions
    return balance
