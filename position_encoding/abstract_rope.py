from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


class RoPE(nn.Module):
    FreqCisBuilderType = Callable[[int, int], Tensor]
    RoPEFunctionType = Callable[[Tensor, Tensor, int, int], Tensor],

    def __init__(
        self,
        hidden_states: int,
        interpolate_factor: float = 1.,
        theta: float = 10000.0,
    ):
        super(RoPE, self).__init__()
        assert interpolate_factor >= 1.
        self._hidden_states = hidden_states
        self._interpolate_factor = interpolate_factor
        self._theta = theta
        self.freqs = self._build_basic_freqs()

    def _build_basic_freqs(self):
        return nn.parameter.Parameter(1.0 / (self._theta ** (
            torch.arange(0, self._hidden_states, 2)
            [: self._hidden_states // 2].float() / self._hidden_states)),
            requires_grad=False
        )

    def _get_pos_ids(self, seq_len: int, start_idx: int = 0):
        pos_ids = torch.arange(
            start_idx, start_idx + seq_len,
            device=self.freqs.device,
            dtype=self.freqs.dtype,
        )
        if self._interpolate_factor > 1:
            return pos_ids / self._interpolate_factor
        return pos_ids

    def build_freq_cis(self, seq_len: int, start_idx: int):
        raise Exception('Not implemented')

    def apply_rotary(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        seq_len_dim_idx: int = 2,
        head_dim_idx: int = 3
    ):
        raise Exception('Not implemented')
