import torch

from .complex_rope import ComplexRoPE


class SimulatedRoPE(ComplexRoPE):
    def _polar(self, magnitude, phase):
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        real_tensor = torch.stack((real_part, imag_part), dim=-1)
        return real_tensor

    def apply_rotary(self, x: torch.Tensor, emb_table: torch.Tensor, seq_len_dim_idx: int = 2, head_dim_idx: int = 3):
        x_ = self._real_to_complex(x)
        emb_table = self._reshape_as_broadcast(
            emb_table, x_, seq_len_dim_idx, head_dim_idx)
        return self._complex_to_real(self._complex_multiply_on_2d_real(emb_table, x_))

    @staticmethod
    def _complex_multiply_on_2d_real(a, b):
        a_real, a_imag = a[..., 0], a[..., 1]
        b_real, b_imag = b[..., 0], b[..., 1]
        result_real = a_real * b_real - a_imag * b_imag
        result_imag = a_real * b_imag + a_imag * b_real
        return torch.stack([result_real, result_imag], dim=-1)

    @staticmethod
    def _reshape_as_broadcast(emb_table, x, seq_len_dim_idx, head_dim_idx):
        assert emb_table.shape == (
            x.size(seq_len_dim_idx), x.size(head_dim_idx), 2)
        target_dims = (seq_len_dim_idx, head_dim_idx)
        out_shape = [v if i in target_dims else 1 for i,
                     v in enumerate(x.shape)]
        out_shape[-1] = 2
        return emb_table.view(*out_shape)

    @staticmethod
    def _real_to_complex(x):
        return x.reshape(*x.shape[:-1], -1, 2)

    @staticmethod
    def _complex_to_real(x):
        flat_idx = x.ndim - 2
        return x.flatten(flat_idx)
