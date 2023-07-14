import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class LossScaler:
    def __init__(self,
                 init_scale=1,
                 min_scale=1,
                 scale_factor=2,
                 logger=None,
                 ):
        self._cur_scale = init_scale
        self._min_scale = min_scale
        self._scale_factor = scale_factor
        self._logger = logger

    @property
    def loss_scale(self):
        return self._cur_scale

    def _check_overflow(self, module: nn.Module):
        for p in module.parameters():
            if p.grad is not None and self.nan_or_inf(p.grad.data):
                return True
        return False

    def nan_or_inf(self, x: torch.Tensor):
        try:
            data_sum = float(x.float().sum())
        except RuntimeError as instance:
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if data_sum in [float('inf'), -float('inf')] or \
                    data_sum != data_sum:
                return True
            return False

    def update(self, overflow):
        if overflow:
            next_scale = max(
                self._cur_scale/self._scale_factor, self._min_scale)

            if self._logger is not None:
                self._logger.warn(
                    'OVERFLOW! Scale value %d -> %d',
                    self._cur_scale,
                    next_scale)

            self._cur_scale = next_scale

    def _scale_grads(self, module: nn.Module):
        for p in module.parameters():
            p.grad.data.mul_(1. / self._cur_scale)

    def step(self, module: nn.Module, opt: Optimizer):
        overflow = self._check_overflow(module)
        if not overflow:
            self._scale_grads(module)
            opt.step()
        self.update(overflow)
        return overflow


if __name__ == '__main__':
    scaler = LossScaler()

    from model import MyModel
    from torch.optim import SGD
    model = MyModel(
        vocab=100,
        pad_token_id=99,
        d_model=128,
        num_head=32,
        max_len=128,
        num_block=6
    )

    opt = SGD(model.parameters(), lr=7e-5)

    loss = model.forward(
        x=torch.randn((20, 32)),
        y=torch.randn((20, 32)),
    )

    opt.zero_grad()
    loss.backward()

    scaler.step(model, opt)
