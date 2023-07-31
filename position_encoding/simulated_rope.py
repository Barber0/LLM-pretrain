import torch
from abstract_rope import RoPE


class SimulatedRoPE(RoPE):
    def build_freq_cis(self, seq_len: int, start_idx: int = 0):
        pos_ids = self._get_pos_ids(seq_len, start_idx)
        phase = torch.outer(pos_ids, self.freqs)
        return self.polar_in_real(torch.ones_like(phase), phase)

    def apply_rotary(self, x: torch.Tensor, freq_cis: torch.Tensor, seq_len_dim_idx: int = 2, head_dim_idx: int = 3):
        x_ = self.real_to_complex(x)
        freq_cis = self.reshape_as_broadcast(
            freq_cis, x_, seq_len_dim_idx, head_dim_idx)
        return self.complex_to_real(self.complex_multiply_on_2d_real(freq_cis, x_))

    @staticmethod
    def complex_multiply_on_2d_real(a, b):
        a_real, a_imag = a[..., 0], a[..., 1]
        b_real, b_imag = b[..., 0], b[..., 1]
        result_real = a_real * b_real - a_imag * b_imag
        result_imag = a_real * b_imag + a_imag * b_real
        return torch.stack([result_real, result_imag], dim=-1)

    @staticmethod
    def polar_in_real(magnitude, phase):
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        real_tensor = torch.stack((real_part, imag_part), dim=-1)
        return real_tensor

    @staticmethod
    def reshape_as_broadcast(freq_cis, x, seq_len_dim_idx, head_dim_idx):
        assert freq_cis.shape == (
            x.size(seq_len_dim_idx), x.size(head_dim_idx), 2)
        target_dims = (seq_len_dim_idx, head_dim_idx)
        out_shape = [v if i in target_dims else 1 for i,
                     v in enumerate(x.shape)]
        out_shape[-1] = 2
        return freq_cis.view(*out_shape)

    @staticmethod
    def real_to_complex(x):
        return x.reshape(*x.shape[:-1], -1, 2)

    @staticmethod
    def complex_to_real(x):
        flat_idx = x.ndim - 2
        return x.flatten(flat_idx)
