import torch
from abstract_rope import RoPE

class ComplexRoPE(RoPE):
    def build_freq_cis(self, seq_len: int, start_idx: int = 0):
        pos_ids = self._get_pos_ids(seq_len, start_idx)
        phase = torch.outer(pos_ids, self.freqs).float()
        return torch.polar(torch.ones_like(phase), phase)

    def apply_rotary(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        seq_len_dim_idx: int = 2,
        head_dim_idx: int = 3
    ):
        x_ = self.real_to_complex(x.float())
        freq_cis = self.reshape_as_broadcast(
            freq_cis, x_, seq_len_dim_idx, head_dim_idx)
        return self.complex_to_real(freq_cis * x_).type_as(x)

    @staticmethod
    def real_to_complex(x):
        return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

    @staticmethod
    def complex_to_real(x):
        return torch.view_as_real(x).flatten(x.ndim - 1)

    @staticmethod
    def reshape_as_broadcast(freq_cis, x, seq_len_dim_idx, head_dim_idx):
        assert freq_cis.shape == (
            x.size(seq_len_dim_idx), x.size(head_dim_idx))
        target_dims = (seq_len_dim_idx, head_dim_idx)
        out_shape = [v if i in target_dims else 1 for i,
                     v in enumerate(x.shape)]
        return freq_cis.view(*out_shape)
