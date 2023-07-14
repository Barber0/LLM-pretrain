import logging
from argparse import ArgumentParser
import os
import torch
from transformers import GPT2Tokenizer


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
    arg_parser.add_argument('--model_name', default='myllm4-model.pt')

    arg_parser.add_argument('--load_home', default=None)
    arg_parser.add_argument('--save_home', default='../models')

    arg_parser.add_argument('--data_name', default='openwebtext')
    arg_parser.add_argument(
        '--data_path', default='/root/autodl-tmp/content/drive/MyDrive/webtext-datasets/arch/')

    arg_parser.add_argument('--tkn_path', default='./tokenizer')
    arg_parser.add_argument('--log_path', default='/root/tf-logs')
    arg_parser.add_argument('--my_log', default='./train.log')

    arg_parser.add_argument('--start_batch', default=0)
    arg_parser.add_argument('--batch_period', default=20)
    arg_parser.add_argument('--save_period', default=500)

    arg_parser.add_argument('--batch_size', default=10)
    arg_parser.add_argument('--micro_batch', default=2)
    arg_parser.add_argument('--lr', default=1e-4)

    arg_parser.add_argument('--max_len', default=1024)
    arg_parser.add_argument('--d_model', default=512)
    arg_parser.add_argument('--n_head', default=32)
    arg_parser.add_argument('--n_block', default=12)

    args = arg_parser.parse_args()
    assert args.batch_size % args.micro_batch == 0
    return args


def save_chkpt(
    bidx,
    model,
    opt,
    args,
    logger
):
    try:
        if os.path.exists(args.save_home):
            torch.save(model.state_dict(),
                       f'{args.save_home}/{args.model_name}.pt')
            torch.save(opt.state_dict(),
                       f'{args.save_home}/opt-{args.model_name}.pt')

            with open(f'./progress.txt', 'w') as f:
                f.write(str(bidx))
    except Exception as ex:
        logger.warn('batch: %d, save state failed: %s', bidx, ex)


def prepare_tokenizer(tkn_path, added_tokens=['[END]']):
    tkn = GPT2Tokenizer.from_pretrained(tkn_path)
    tkn.pad_token = '[PAD]'
    tkn.add_tokens(added_tokens)
    VOCAB_SIZE = tkn.vocab_size + len(added_tokens)
    return tkn, VOCAB_SIZE
