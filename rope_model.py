import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func

min_fp16 = torch.finfo(torch.float16).min


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    complex_dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: complex_dim].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    assert freqs_cis.shape == (end, complex_dim)
    return freqs_cis


def reshape_as_broadcast(freq_cis, x):
    assert freq_cis.shape[-2:] == x.shape[-2:]
    return freq_cis.view(1, 1, x.shape[2], x.shape[3])


def reshape_as_broadcast2(freq_cis, x, seq_len_dim_idx=2, head_dim_idx=3):
    assert freq_cis.size(seq_len_dim_idx) == x.size(seq_len_dim_idx)
    assert freq_cis.size(head_dim_idx) == x.size(head_dim_idx)
    target_dims = (seq_len_dim_idx, head_dim_idx)
    out_shape = [v if i in target_dims else 1 for i, v in enumerate(x.shape)]
    return freq_cis.view(*out_shape)

def real_to_complex(x):
    return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))


def complex_to_real(x):
    flat_idx = x.ndim - 1
    return torch.view_as_real(x).flatten(flat_idx)


def apply_rotary(x, freq_cis):
    x_ = real_to_complex(x)
    freq_cis = reshape_as_broadcast(freq_cis, x_)
    return complex_to_real(freq_cis * x_).type_as(x)


def create_mask(max_len):
    return 1 - torch.triu(torch.ones((1, 1, max_len, max_len)), diagonal=1)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_std = (x - x_mean).pow(2).mean(-1, keepdim=True)
        return self.w * (x - x_mean) / (x_std + self.eps) + self.b


class RoPE_MHA(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super(RoPE_MHA, self).__init__()
        assert d_model % num_head == 0
        self.num_head = num_head
        self.head_size = d_model // num_head
        self.head_scale = math.sqrt(self.head_size)

        self.dropout_val = dropout
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model * 3)
        self.ff = nn.Linear(d_model, d_model)

    def forward(
        self,
        x,
        freq_cis_q,
        freq_cis_k,
        mask=None,
        prefix_kv=None
    ):

        x_proj = self.proj(x)
        proj_shape = x_proj.shape[:-1] + (3, self.num_head, self.head_size)
        x_proj = x_proj.contiguous().view(proj_shape)
        assert x_proj.ndim == 5
        
        x_proj = x_proj.permute(2, 0, 3, 1, 4)
        assert x_proj.shape == (
            3,
            x.size(0),
            self.num_head,
            x.size(1),
            self.head_size
        )
        q, k, v = x_proj

        next_prefix_kv = None
        if prefix_kv is not None:
            pk, pv = prefix_kv
            k = torch.cat((pk, k), dim=-2)
            v = torch.cat((pv, v), dim=-2)

        if mask is None:
            q = apply_rotary(q,  freq_cis_q)
            k = apply_rotary(k,  freq_cis_k)
            attn_o = flash_attn_func(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                dropout_p=self.dropout_val, 
                causal=True
            )
        else:
            next_prefix_kv = torch.stack((k, v))
            q = apply_rotary(q,  freq_cis_q)
            k = apply_rotary(k,  freq_cis_k)
            attn_o = self.attn(q, k, v, mask).transpose(1, 2).contiguous()

        merged_shape = attn_o.shape[:-2] + (x.size(-1), )
        attn_o = attn_o.view(merged_shape)

        ff_o = self.ff(attn_o)
        return ff_o, next_prefix_kv

    def attn(self, q, k, v, mask):
        scores = q.matmul(k.transpose(-2, -1)) / self.head_scale
        scores = scores.masked_fill(mask == 0, min_fp16)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return scores.matmul(v)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super(MLP, self).__init__()
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.ff2(self.gelu(self.ff1(x)))


class Block(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super(Block, self).__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = RoPE_MHA(d_model, num_head, dropout)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 4)

    def forward(
            self,
            x,
            freq_cis_q,
            freq_cis_k,
            mask=None,
            prefix_kv=None
    ):
        attn_o, prefix_kv = self.attn(
            self.ln1(x), freq_cis_q, freq_cis_k, mask, prefix_kv)
        x = x + attn_o
        x = x + self.mlp(self.ln2(x))
        return x, prefix_kv


def prepare_freq_cis(
    x: torch.Tensor,
    base_freq_cis: torch.Tensor,
    seq_start: int,
    seq_end: int,
):
    seq_end = seq_start+x.size(-1)
    freq_cis_k = base_freq_cis[:seq_end].to(x.device)
    freq_cis_q = freq_cis_k if seq_start == 0 else base_freq_cis[
        seq_start:seq_end].to(x.device)
    return freq_cis_q, freq_cis_k


def prepare_mask(
    x: torch.Tensor,
    base_mask: torch.Tensor,
    seq_start: int,
    seq_end: int
):
    mask = base_mask[..., seq_start:seq_end, :seq_end]
    return mask.to(x.device)


def process_prefix_kv_list(x, num_blocks, prefix_kv_list=None):
    if prefix_kv_list is None:
        seq_start = 0
        prefix_kv_list = [None] * num_blocks
    else:
        print('--------------------------', len(prefix_kv_list), sum([1  if it is None else 0 for it in prefix_kv_list]))
        seq_start = prefix_kv_list[0][0].size(-2)

    seq_end = seq_start+x.size(-1)
    return seq_start, seq_end, prefix_kv_list


class LLM_Embeding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_blocks: int,
        base_freq_cis: torch.Tensor,
        base_mask: torch.Tensor,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self._base_freq_cis = base_freq_cis
        self._base_mask = base_mask
        self._num_blocks = num_blocks

    def forward_with_prefix(self, x, generate=False, prefix_kv_list=None):
        emb_out = super().forward(x)

        seq_start, seq_end, prefix_kv_list = process_prefix_kv_list(
            x,
            self._num_blocks,
            prefix_kv_list
        )

        freq_cis_q, freq_cis_k = prepare_freq_cis(
            x,
            self._base_freq_cis,
            seq_start,
            seq_end
        )

        mask = None
        if generate:
            mask = prepare_mask(
                x,
                self._base_mask,
                seq_start,
                seq_end
            )

        return emb_out, freq_cis_q, freq_cis_k, mask, prefix_kv_list

    def forward(self, x):
        return self.forward_with_prefix(x)[:-1]


class SequentialBlock(Block):
    def forward(self, ipt_tuple):
        x, freq_cis_q, freq_cis_k, mask = ipt_tuple
        out = super().forward(x, mask, freq_cis_q, freq_cis_k)[0]
        return out, mask, freq_cis_q, freq_cis_k


class SequentialLayerNorm(LayerNorm):
    def forward(self, x):
        return super().forward(x[0])


def create_sequential_model(
    vocab,
    d_model=768,
    num_head=12,
    max_len=512,
    ext_factor=2,
    dropout=0.1,
    num_blocks=12
):
    head_size = d_model//num_head
    ext_seq_len = max_len*ext_factor

    module_list = [
        LLM_Embeding(vocab, d_model, num_blocks, precompute_freqs_cis(
            head_size, ext_seq_len), create_mask(ext_seq_len))
    ] + [
        SequentialBlock(d_model, num_head, dropout)
        for _ in range(num_blocks)
    ] + [
        SequentialLayerNorm(d_model),
        nn.Linear(d_model, vocab, bias=False)
    ]
    return nn.Sequential(*module_list), len(module_list)


class LLM(nn.Module):
    def __init__(self, vocab, pad_token_id, d_model=768, num_head=12, max_len=512, ext_factor=2, dropout=0.1, num_blocks=12):
        super(LLM, self).__init__()

        head_size = d_model//num_head
        ext_seq_len = max_len*ext_factor

        self.emb = LLM_Embeding(
            vocab,
            d_model,
            num_blocks,
            precompute_freqs_cis(head_size, ext_seq_len),
            create_mask(ext_seq_len)
        )

        self.blocks = nn.ModuleList([
            Block(d_model, num_head, dropout)
            for _ in range(num_blocks)
        ])
        self.ln = LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab, bias=False)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def raw(self, x, generate=False, prefix_kv_list=None):
        x_rep, freq_cis_q, freq_cis_k, mask, prefix_kv_list = \
            self.emb.forward_with_prefix(x, generate, prefix_kv_list)

        next_prefix_kv_list = []
        for layer, prefix_kv in zip(self.blocks, prefix_kv_list):
            x_rep, next_prefix_kv = layer.forward(
                x_rep,
                freq_cis_q,
                freq_cis_k,
                mask,
                prefix_kv
            )
            next_prefix_kv_list.append(next_prefix_kv)

        x_rep = self.ln(x_rep)
        return x_rep, next_prefix_kv_list

    def forward(self, x, y=None, generate=False, prefix_kv_list=None, num_micro_batch=1):
        x_rep, next_prefix_kv_list = self.raw(x, generate, prefix_kv_list)
        y_pred = self.decoder(x_rep)
        if y is None:
            return y_pred, next_prefix_kv_list
        else:
            loss = self.loss_fn(
                y_pred.contiguous().view(-1, y_pred.size(-1)),
                y.contiguous().view(-1)
            )
            if num_micro_batch > 1:
                return loss / num_micro_batch
            return loss
