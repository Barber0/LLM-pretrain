import random

import deepspeed
import numpy as np
import torch
from datasets import load_from_disk
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_obj import ModelArgs, ProgramArgs, TrainArgs
# from models import SFLLM
from main_models.rope_model import LLM
from sf_trainer import ModelType, SFTrainer
from utils import (build_logger, convert_batch_to_ids, count_parameters,
                   get_args, prepare_tokenizer)

torch.autograd.set_detect_anomaly(True)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed = 168
set_random_seed(seed)


def compute_loss_fn(
    eng: ModelType,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_accum_period,
):
    return eng.forward(x, y)


def update_param_fn(
    bidx: int,
    loss: torch.Tensor,
    eng: ModelType,
    opt: Optimizer,
    grad_accum_period,
):
    print(f'--------{bidx}---{loss}')
    eng.backward(loss)
    eng.step()


def main(
    prog_args: ProgramArgs,
    model_args: ModelArgs,
    train_args: TrainArgs
):
    logger = build_logger(
        train_args.deepspeed_ckpt_tag,
        prog_args.log_path,
    )
    tkn, VOCAB_SIZE = prepare_tokenizer(prog_args.tokenizer_path)
    base_model = LLM(
        vocab=VOCAB_SIZE,
        pad_token_id=tkn.pad_token_id,
        d_model=model_args.hidden_states,
        num_head=model_args.n_heads,
        max_len=model_args.max_len,
        ext_factor=model_args.ext_factor,
        num_blocks=model_args.n_layers
    )
    # base_model = SFLLM(
    #     vocab_size=VOCAB_SIZE,
    #     pad_token_id=tkn.pad_token_id,
    #     args=model_args,
    # )
    param_num = count_parameters(base_model) * 1e-9
    logger.info('Model parameters: %f B', param_num)

    use_torch_ckpt = SFTrainer.validate_ckpt(
        train_args.torch_ckpt_home,
        train_args.torch_ckpt_tag
    )
    if use_torch_ckpt:
        SFTrainer.load_ckpt(
            train_args,
            base_model,
            None,
            logger
        )

    train_set = load_from_disk(prog_args.train_path)

    model_engine, opt, train_loader, _ = deepspeed.initialize(
        model=base_model,
        config=prog_args.deepspeed_cfg,
        training_data=train_set,
    )

    use_ds_ckpt = SFTrainer.validate_ckpt(
        train_args.deepspeed_ckpt_home,
        train_args.deepspeed_ckpt_tag
    )
    if not use_torch_ckpt and use_ds_ckpt:
        SFTrainer.load_ckpt(
            train_args,
            model_engine,
            None,
            logger
        )

    _, get_micro_batch_size, get_grad_accum_steps = model_engine.get_batch_info()
    train_args.batch_size = get_micro_batch_size()
    train_args.grad_accum_period = get_grad_accum_steps()

    validate_set = load_from_disk(prog_args.validate_path).shuffle()
    validate_loader = DataLoader(
        validate_set,
        batch_size=train_args.batch_size,
        shuffle=True
    )

    tb_writer = SummaryWriter(log_dir=prog_args.tensorboard_path)

    def batch_collate_fn(batch): return convert_batch_to_ids(
        tkn,
        batch['text'],
        model_args.max_len,
        model_args.ext_factor,
        model_engine.device
    )

    trainer = SFTrainer(
        train_args=train_args,
        model=model_engine,
        opt=opt,
        train_loader=train_loader,
        validate_loader=validate_loader,
        logger=logger,
        tb_writer=tb_writer,
        batch_collate_fn=batch_collate_fn,
        compute_loss_fn=compute_loss_fn,
        update_param_fn=update_param_fn
    )

    trainer.train()


if __name__ == '__main__':
    prog_args, model_args, train_args = get_args()
    main(
        prog_args=prog_args,
        model_args=model_args,
        train_args=train_args,
    )
