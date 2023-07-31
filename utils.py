import logging
from argparse import ArgumentParser
from dataclasses import asdict, fields

from transformers import AutoTokenizer

from data_obj import ModelArgs, PositionEmbeddingType, ProgramArgs, TrainArgs


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
    
    fh = logging.FileHandler(log_filename)
    fh.setLevel(level)

    formatter = logging.Formatter(str_format)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


def add_arguments_from_dataclass(
    parser: ArgumentParser, 
    dataclass_instance
):
    dataclass_dict = asdict(dataclass_instance)
    for key, value in dataclass_dict.items():
        val_type = str if value is None else type(value)
        if val_type is PositionEmbeddingType:
            parser.add_argument(f'--{key}', type=val_type, choices=list(PositionEmbeddingType), default=value)
        else:
            parser.add_argument(f'--{key}', type=val_type, default=value)


def parse_to_dataclass(dataclass_type, args):
    args_dict = vars(args)
    dataclass_fields = {field.name: field.type for field in fields(dataclass_type)}
    filtered_args_dict = {key: value for key, value in args_dict.items() if key in dataclass_fields}
    return dataclass_type(**filtered_args_dict)


def get_args():
    arg_parser = ArgumentParser()
    add_arguments_from_dataclass(arg_parser, ModelArgs())
    add_arguments_from_dataclass(arg_parser, TrainArgs())
    add_arguments_from_dataclass(arg_parser, ProgramArgs())
    args = arg_parser.parse_args()

    prog_args = parse_to_dataclass(ProgramArgs, args)
    model_args = parse_to_dataclass(ModelArgs, args)
    train_args = parse_to_dataclass(TrainArgs, args)
    return prog_args, model_args, train_args


def prepare_tokenizer(tkn_path):
    tkn = AutoTokenizer.from_pretrained(tkn_path)
    VOCAB_SIZE = tkn.vocab_size
    return tkn, VOCAB_SIZE

    
def convert_batch_to_ids(
    tokenizer, 
    pure_txt_list, 
    max_len, 
    ext_factor,
    device
):
    base_ids = tokenizer.batch_encode_plus(
        pure_txt_list,
        max_length=max_len * ext_factor + 1,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).input_ids
    # base_ids, x_attn_mask = x_encoded['input_ids'], x_encoded['attention_mask']
    # batch_avg_ntokens = x_attn_mask.sum() / x_attn_mask.size(0)
    input_ids = base_ids[..., :-1]
    target_ids = base_ids[..., 1:]
    return input_ids.to(device), target_ids.to(device)