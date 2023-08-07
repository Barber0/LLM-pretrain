import os
from logging import Logger
from time import time
from typing import Callable, List, Tuple, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_obj.train_args import TrainArgs

try:
    from deepspeed import DeepSpeedEngine as DSEnginePlaceholder
except ImportError:
    class DSEnginePlaceholder:
        pass

ModelType = Union[DSEnginePlaceholder, Module]


class SFTrainer:
    def __init__(
        self,
        train_args: TrainArgs,
        model: ModelType,
        opt: Optimizer,
        train_loader: DataLoader,
        validate_loader: DataLoader,
        logger: Logger,
        tb_writer: SummaryWriter,
        batch_collate_fn: Callable[[List], Tuple[torch.Tensor, torch.Tensor]],
        compute_loss_fn: Callable[[ModelType, torch.Tensor, torch.Tensor, int], torch.Tensor],
        update_param_fn: Callable[[int, torch.Tensor, ModelType, Optimizer, int], None],
    ):
        self.args = train_args
        self.model = model
        self.opt = opt
        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.batch_collate_fn = batch_collate_fn
        self.compute_loss_fn = compute_loss_fn
        self.update_param_fn = update_param_fn
        self.logger = logger
        self.tb_writer = tb_writer

        self._validate_iter = None

    @staticmethod
    def validate_ckpt(ckpt_home, ckpt_tag):
        return ckpt_home is not None and ckpt_tag is not None and \
            os.path.exists(ckpt_home)

    @staticmethod
    def load_ckpt(
        args: TrainArgs,
        model: ModelType,
        opt: Optimizer,
        logger: Logger
    ):
        if isinstance(model, DSEnginePlaceholder):
            if not SFTrainer.validate_ckpt(args.deepspeed_ckpt_home, args.deepspeed_ckpt_tag):
                logger.warning('Checkpoint home not found: %s/%s',
                               args.deepspeed_ckpt_home,
                               args.deepspeed_ckpt_tag)
                if args.deepspeed_ckpt_home is not None:
                    os.makedirs(args.deepspeed_ckpt_home, exist_ok=True)
                return
            model.load_checkpoint(args.deepspeed_ckpt_home,
                                  args.deepspeed_ckpt_tag)
        elif isinstance(model, Module):
            if not SFTrainer.validate_ckpt(args.torch_ckpt_home, args.torch_ckpt_tag):
                logger.warning('Checkpoint home not found: %s/%s',
                               args.torch_ckpt_home,
                               args.torch_ckpt_tag)
                if args.torch_ckpt_home is not None:
                    os.makedirs(args.torch_ckpt_home, exist_ok=True)
                return

            if args.torch_ckpt_tag is not None:
                assert isinstance(model, Module)
                model_path = f'{args.torch_ckpt_home}/{args.torch_ckpt_tag}.pt'
                if os.path.exists(model_path):
                    ckpt = torch.load(model_path)
                    if args.deepspeed_module_key in ckpt:
                        result = model.load_state_dict(ckpt[args.deepspeed_module_key], strict=False)
                    else:
                        result = model.load_state_dict(ckpt, strict=False)

                    logger.info('Model loaded: %s; Missing: %s; Unexpected: %s', model_path, result.missing_keys, result.unexpected_keys)
                else:
                    logger.warn('Model not found: %s', model_path)

                if opt is not None:
                    opt_path = f'{args.torch_ckpt_home}/{args.torch_ckpt_opt_prefix}-{args.torch_ckpt_tag}.pt'
                    if os.path.exists(opt_path):
                        ckpt = torch.load(opt_path)
                        opt.load_state_dict(ckpt)
                        logger.info('Optimizer loaded: %s', opt_path)
                    else:
                        logger.warn('Optimizer not found: %s', model_path)

    def get_next_validate_batch(self):
        if self._validate_iter is None:
            self._validate_iter = iter(self.validate_loader)

        try:
            batch = next(self._validate_iter)
        except StopIteration:
            self._validate_iter = iter(self.validate_loader)
            batch = next(self._validate_iter)

        return batch

    @staticmethod
    def start_timer():
        stime = time()

        def _stop_timer():
            return time() - stime
        return _stop_timer

    @staticmethod
    def list_mean(
        data_list: List,
        need_clear: bool = True
    ):
        mean_val = sum(data_list) / len(data_list)
        if need_clear:
            data_list.clear()
        return mean_val

    def escape_from_exception(self, ep, bidx, callback):
        try:
            callback()
        except Exception as ex:
            self.logger.warn('ep: %d, batch: %d, err: %s', ep, bidx, ex)

    def train_one_batch(self, ep, bidx, batch):
        self.model.train()
        x, y = self.batch_collate_fn(batch)
        loss = self.compute_loss_fn(
            self.model,
            x,
            y,
            self.args.grad_accum_period,
        )

        if self.args.local_rank == 0:
            self.escape_from_exception(
                ep,
                bidx,
                lambda: self.tb_writer.add_scalar(
                    'Train Loss', loss.item(), bidx)
            )

        self.update_param_fn(
            bidx,
            loss,
            self.model,
            self.opt,
            self.args.grad_accum_period,
        )
        return loss.item()

    def validate(self, ep, bidx):
        stop_validate_timer = self.start_timer()
        self.model.eval()
        validate_loss_list = []
        with torch.no_grad():
            for _ in range(self.args.validate_batch_num):
                batch = self.get_next_validate_batch()
                x, y = self.batch_collate_fn(batch)
                loss = self.compute_loss_fn(
                    self.model,
                    x,
                    y,
                    self.args.grad_accum_period
                )
                validate_loss_list.append(loss.item())
        avg_validate_loss = self.list_mean(validate_loss_list, False)
        validate_time = stop_validate_timer()
        self.logger.info(
            '[V] ep: %d, batch: %d, calc_time: %.2f, loss: %f',
            ep, bidx, validate_time, avg_validate_loss
        )
        self.escape_from_exception(ep, bidx, lambda: self.tb_writer.add_scalar(
            'Validation Loss', avg_validate_loss, bidx))

    def period_log(
        self,
        ep,
        bidx,
        period_loss_list,
        period_calc_time_list,
    ):
        avg_loss = self.list_mean(period_loss_list)
        avg_calc_time = self.list_mean(period_calc_time_list)

        cur_lr = self.opt.param_groups[0]['lr']
        self.logger.info(
            '[T] ep: %d, batch: %d, calc_time: %.2f, '
            'loss: %f, lr: %f',
            ep, bidx, avg_calc_time,
            avg_loss, cur_lr
        )

        if self.args.local_rank == 0:
            self.escape_from_exception(
                ep,
                bidx,
                lambda: self.tb_writer.add_scalar(
                    'Learning Rate', cur_lr, bidx)
            )
            self.tb_writer.flush()

    def save_model_in_fp16(self, ep, bidx):
        if isinstance(self.model, DSEnginePlaceholder):
            state_dict = self.model.module.state_dict()
            ckpt_home = self.args.deepspeed_ckpt_home
        elif isinstance(self.model, Module):
            state_dict = self.model.state_dict()
            ckpt_home = self.args.torch_ckpt_home
        else:
            raise Exception(f'Unsupported model type: {type(self.model)}')

        state_dict_fp16 = {k: v.half() for k, v in state_dict.items()}
        os.makedirs(ckpt_home, exist_ok=True)
        save_path = f'{ckpt_home}/{self.args.deepspeed_ckpt_tag}-{ep}_{bidx}.pt'
        torch.save(state_dict_fp16, save_path)

    def save_optimizer(self, ep, bidx):
        state_dict = self.opt.state_dict()
        os.makedirs(self.args.torch_ckpt_home, exist_ok=True)
        save_path = f'{self.args.torch_ckpt_home}/opt-{self.args.torch_ckpt_tag}-{ep}_{bidx}.pt'
        torch.save(state_dict, save_path)

    def save_all_state(self, ep, bidx):
        if isinstance(self.model, DSEnginePlaceholder):
            self.escape_from_exception(
                ep,
                bidx,
                lambda: self.model.save_checkpoint(
                    self.args.deepspeed_ckpt_home,
                    tag=self.args.deepspeed_ckpt_tag
                )
            )
            with open(f'{self.args.deepspeed_ckpt_home}/progress.txt', 'w') as f:
                f.write(f'{ep}-{bidx}')

        elif isinstance(self.model, Module):
            self.escape_from_exception(ep, bidx,
                                       lambda: self.save_model_in_fp16(ep, bidx))
            self.escape_from_exception(ep, bidx,
                                       lambda: self.save_optimizer(ep, bidx))

    def train_one_epoch(self, ep=0):
        data_iter = iter(self.train_loader)

        period_loss_list = []
        period_calc_time_list = []

        real_bidx = -1
        for bidx, batch in enumerate(data_iter):
            if bidx < self.args.start_batch:
                continue
            real_bidx = bidx + 1

            stop_calc_timer = self.start_timer()
            try:
                loss_val = self.train_one_batch(ep, real_bidx, batch)
            except Exception as ex:
                logger.warning('[T] ep: %d, batch: %d, Calculating error: %s', ep, real_bidx, ex)
                if isinstance(self.model, DSEnginePlaceholder):
                    self.model.zero_grad()
                else:
                    self.opt.zero_grad()
                continue
                
            period_loss_list.append(loss_val)
            period_calc_time_list.append(stop_calc_timer())

            if real_bidx % self.args.log_period == 0:
                self.period_log(
                    ep,
                    real_bidx,
                    period_loss_list,
                    period_calc_time_list
                )

            if real_bidx % self.args.save_period == 0:
                self.save_all_state(ep, real_bidx)

            if self.args.local_rank == 0 and real_bidx % self.args.validate_period == 0:
                self.validate(ep, real_bidx)

            if self.args.local_rank == 0 and real_bidx % self.args.replicate_period == 0:
                self.escape_from_exception(
                    ep,
                    real_bidx,
                    lambda: self.save_model_in_fp16(ep, real_bidx)
                )

        return real_bidx

    def train(self):
        for ep in range(self.args.epochs):
            if ep < self.args.start_epoch:
                continue
            final_bidx = self.train_one_epoch(ep)
            self.save_all_state(ep, final_bidx)
