import random
from logging import Logger

import deepspeed
import numpy as np
import torch
from datasets import load_from_disk
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from data_obj import ModelArgs, ProgramArgs, TrainArgs
from data_obj.train_args import TrainArgs
from sf_trainer import ModelType, SFTrainer
from utils import (build_logger, convert_batch_to_ids, count_parameters,
                   get_args, prepare_tokenizer)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed = 168
set_random_seed(seed)


def fix_grad(model, logger):
    for param in model.parameters():
        param.requires_grad = False

    for i in range(-2, 0):
        for param in model.blocks[i].parameters():
            param.requires_grad = True

    for param in model.ln.parameters():
        param.requires_grad = True

    param_grad_info = ['']
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters():
            param_grad_info.append(
                f"Layer: {name} Parameter: {param_name} requires_grad: {param.requires_grad}")

    logger.info('\n'.join(param_grad_info))


class SFTrainerForDSDP(SFTrainer):
    def __init__(self, train_args: TrainArgs, model: ModelType, opt: Optimizer,
                 train_loader: DataLoader, validate_loader: DataLoader,
                 logger: Logger, tb_writer: SummaryWriter,
                 tokenizer: AutoTokenizer):
        super().__init__(train_args, model, opt, train_loader,
                         validate_loader, logger, tb_writer)
        self.tokenizer = tokenizer

    def process_batch(self, batch):
        return convert_batch_to_ids(
            self.tokenizer,
            batch['text'],
            model_args.max_len,
            model_args.ext_factor,
            self.model.device
        )

    def train_batch(self, bidx, batch):
        x, y = self.process_batch(batch)
        assert isinstance(self.model, deepspeed.DeepSpeedEngine)
        loss = self.model.forward(x, y)
        self.model.backward(loss)
        self.model.step()
        return loss.item()

    def validate_batch(self, batch):
        x, y = self.process_batch(batch)
        return self.model.forward(x, y).item()


def main(
    prog_args: ProgramArgs,
    model_args: ModelArgs,
    train_args: TrainArgs
):
    logger = build_logger(
        train_args.deepspeed_ckpt_tag,
        prog_args.log_path,
        local_rank=train_args.local_rank,
    )
    tkn, VOCAB_SIZE = prepare_tokenizer(prog_args.tokenizer_path)

    from models import SFLLM
    base_model = SFLLM(
        vocab_size=VOCAB_SIZE,
        pad_token_id=tkn.pad_token_id,
        args=model_args,
    )

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

    validate_set = load_from_disk(prog_args.validate_path).shuffle(
        seed=train_args.start_batch)
    validate_loader = DataLoader(
        validate_set,
        batch_size=train_args.batch_size,
        shuffle=True,
    )

    tb_writer = SummaryWriter(log_dir=prog_args.tensorboard_path)

    trainer = SFTrainerForDSDP(
        train_args=train_args,
        model=model_engine,
        opt=opt,
        train_loader=train_loader,
        validate_loader=validate_loader,
        logger=logger,
        tb_writer=tb_writer,
        tokenizer=tkn,
    )

    trainer.train()


if __name__ == '__main__':
    prog_args, model_args, train_args = get_args()
    main(
        prog_args=prog_args,
        model_args=model_args,
        train_args=train_args,
    )
