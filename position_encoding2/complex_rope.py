import torch
from torch import Tensor

from .abstract_rope import RoPE


class ComplexRoPE(RoPE):
    def apply_rotary(
        self,
        x: Tensor,
        phase: Tensor,
        seq_len_dim_idx: int = 2,
        head_dim_idx: int = 3
    ):
        x_ = self._real_to_complex(x.float())
        phase_float = phase.float()
        emb_table = self._reshape_as_broadcast(
            torch.polar(torch.ones_like(phase_float), phase_float),
            x_,
            seq_len_dim_idx,
            head_dim_idx
        ).to(x_.device)
        return self._complex_to_real(emb_table * x_).type_as(x)

    @staticmethod
    def _real_to_complex(x):
        return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

    @staticmethod
    def _complex_to_real(x):
        return torch.view_as_real(x).flatten(x.ndim - 1)

    @staticmethod
    def _reshape_as_broadcast(emb_table, x, seq_len_dim_idx, head_dim_idx):
        assert emb_table.shape == (
            x.size(seq_len_dim_idx), x.size(head_dim_idx))
        target_dims = (seq_len_dim_idx, head_dim_idx)
        out_shape = [v if i in target_dims else 1 for i,
                     v in enumerate(x.shape)]
        return emb_table.view(*out_shape)
