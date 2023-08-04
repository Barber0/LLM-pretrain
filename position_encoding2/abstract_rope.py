import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


class RoPE(nn.Module):
    EmbTableGetterType = Callable[[int, int], Tensor]
    RoPEFunctionType = Callable[[Tensor, Tensor, int, int], Tensor],

    def __init__(
        self,
        hidden_states: int,
        interpolate_factor: float = 1.,
        theta: float = 10000.0,
        max_len: int = 1024,
        ext_factor: int = 1,
    ):
        super(RoPE, self).__init__()
        assert interpolate_factor >= 1.
        self._hidden_states = hidden_states
        self._interpolate_factor = interpolate_factor
        self._theta = theta
        self._max_len = max_len
        self._ext_factor = ext_factor
        self._ensure_emb_table()

    def _ensure_emb_table(self):
        freqs = 1.0 / (self._theta ** (
            torch.arange(0, self._hidden_states, 2)
            [: self._hidden_states // 2].float() / self._hidden_states))

        if self._interpolate_factor > 1:
            pos_ids = torch.arange(
                0, int(math.ceil(self._max_len*self._ext_factor *
                                 self._interpolate_factor))
            ) / self._interpolate_factor
        else:
            pos_ids = torch.arange(0, self._max_len*self._ext_factor)
        phase = torch.outer(pos_ids, freqs)
        self.emb_table = nn.parameter.Parameter(
            self._polar(torch.ones_like(phase), phase), requires_grad=False)

    def _polar(self, mag, phase):
        return torch.polar(mag, phase)

    def get_embedding_table(self, seq_len: int, start_idx: int):
        self._ensure_emb_table()
        return self.emb_table[start_idx:start_idx+seq_len]

    def apply_rotary(
        self,
        x: torch.Tensor,
        emb_table: torch.Tensor,
        seq_len_dim_idx: int = 2,
        head_dim_idx: int = 3
    ):
        raise Exception('Not implemented')
