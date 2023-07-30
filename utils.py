import logging
from argparse import ArgumentParser
import os
import torch
from transformers import AutoTokenizer
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
    arg_parser.add_argument('--tag_name', default='main')

    arg_parser.add_argument('--ds_cfg', default='./ds_cfg.json')

    arg_parser.add_argument('--load_home', default=None)
    arg_parser.add_argument('--save_home', default='/root/autodl-tmp/model-arch')
    arg_parser.add_argument('--ckpt', default=None)

    arg_parser.add_argument('--data_name', default='openwebtext')
    arg_parser.add_argument(
        '--data_path', default='/root/autodl-tmp/pile02-parsed')
    arg_parser.add_argument('--valid_data_path', default='/root/autodl-tmp/pile00-parsed')
    arg_parser.add_argument('--data_vendor', default='hf')
    arg_parser.add_argument('--loader_workers', default=32, type=int)

    arg_parser.add_argument('--tkn_path', default='./tokenizer')
    arg_parser.add_argument('--log_path', default='/root/tf-logs')
    arg_parser.add_argument('--my_log', default='./tmp/train.log')

    arg_parser.add_argument('--valid_period', default=200, type=int)
    
    arg_parser.add_argument('--valid_batch_num', default=40, type=int)
    arg_parser.add_argument('--start_batch', default=0, type=int)
    arg_parser.add_argument('--batch_period', default=20, type=int)
    arg_parser.add_argument('--flush_period', default=20, type=int)
    arg_parser.add_argument('--save_period', default=500, type=int)
    arg_parser.add_argument('--model_save_period', default=3000, type=int)
    arg_parser.add_argument('--overlap_factor', default=6, type=int)

    arg_parser.add_argument('--epochs', default=2, type=int)
    arg_parser.add_argument('--batch_size', default=10, type=int)

    arg_parser.add_argument('--max_len', default=512, type=int)
    arg_parser.add_argument('--ext_factor', default=2, type=int)
    arg_parser.add_argument('--d_model', default=512, type=int)
    arg_parser.add_argument('--n_head', default=32, type=int)
    arg_parser.add_argument('--n_block', default=12, type=int)
    arg_parser.add_argument('--local_rank', default=0)
    arg_parser.add_argument('--world_size', default=1, type=int)

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


def prepare_tokenizer(tkn_path):
    tkn = AutoTokenizer.from_pretrained(tkn_path)
    VOCAB_SIZE = tkn.vocab_size
    return tkn, VOCAB_SIZE


def get_partition_balance(num_layers):
    partitions = torch.cuda.device_count()
    avg_num_layers = num_layers//partitions
    balance = [avg_num_layers for _ in range(partitions)]
    balance[-1] += num_layers % partitions
    return balance

def save_model_in_fp16(model_eng, save_home, model_name, bidx):
    state_dict = model_eng.module.state_dict()
    state_dict_fp16 = {k: v.half() for k, v in state_dict.items()}
    torch.save(state_dict_fp16, f'{save_home}/{model_name}-{bidx}.pt')
    
def convert_batch_to_ids(
    tokenizer, 
    pure_txt_list, 
    max_len, 
    ext_factor,
    device
):
    x_encoded = tokenizer.batch_encode_plus(
        pure_txt_list,
        max_length=max_len * ext_factor + 1,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    base_ids, x_attn_mask = x_encoded['input_ids'], x_encoded['attention_mask']
    batch_avg_ntokens = x_attn_mask.sum() / x_attn_mask.size(0)
    
    input_ids = base_ids[..., :-1]
    target_ids = base_ids[..., 1:]
    return input_ids.to(device), target_ids.to(device), batch_avg_ntokens

def remove_html_and_links(text):
    soup = BeautifulSoup(text, "html.parser")
    text_without_tags = soup.get_text()

    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text_without_links = url_pattern.sub('', text_without_tags)

    return text_without_links