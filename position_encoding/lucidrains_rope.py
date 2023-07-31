import torch
from abstract_rope import RoPE
from einops import rearrange, repeat


class LucidrainsRoPE(RoPE):
    def build_freq_cis(self, seq_len: int, start_idx: int = 0):
        pos_ids = self._get_pos_ids(seq_len, start_idx)
        freq_cis = torch.einsum('..., f -> ...f', pos_ids, self.freqs)
        return repeat(freq_cis, '... n -> ... (n r)', r=2)

    def apply_rotary(self, x: torch.Tensor, freq_cis: torch.Tensor, seq_len_dim_idx: int = 2, head_dim_idx: int = 3):
        tmp = LucidrainsRoPE._rotate_half(x)
        assert x.shape == tmp.shape
        return (x * freq_cis.cos()) + (tmp * freq_cis.sin())

    @staticmethod
    def _rotate_half(x: torch.Tensor):
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d r -> ... (d r)')
