import math

import torch
import torch.nn as nn
from torch import Tensor


class RoPE(nn.Module):
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
        self._ensure_phase_table()

    def _ensure_phase_table(self):
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

        self.phase = nn.parameter.Parameter(
            torch.outer(pos_ids, freqs), requires_grad=False)

    def get_phase_table(self, seq_len: int, start_idx: int):
        return self.phase[start_idx:start_idx+seq_len]

    def get_phase_table_for_qk(self, seq_len: int, start_idx: int):
        emb_table_k = self.get_phase_table(start_idx+seq_len, 0)
        emb_table_q = emb_table_k if start_idx == 0 else self.get_phase_table(
            seq_len, start_idx)
        return emb_table_q, emb_table_k

    def apply_rotary(
        self,
        x: torch.Tensor,
        phase: torch.Tensor,
        seq_len_dim_idx: int = 2,
        head_dim_idx: int = 3
    ):
        raise Exception('Not implemented')

    def apply_rotary_for_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len_dim_idx: int,
        head_dim_idx: int
    ):
        q_len = q.size(seq_len_dim_idx)
        k_len = k.size(seq_len_dim_idx)
        start_idx = k_len - q_len
        emb_table_q, emb_table_k = self.get_phase_table_for_qk(
            q_len, start_idx)
        q = self.apply_rotary(
            q, emb_table_q, seq_len_dim_idx, head_dim_idx)
        k = self.apply_rotary(
            k, emb_table_k, seq_len_dim_idx, head_dim_idx)
        return q, k
