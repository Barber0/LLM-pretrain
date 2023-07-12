import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

min_fp16 = torch.finfo(torch.float16).min


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape[-1] == x.shape[-1]
    shape = [d if i == 1 or i == ndim -
             1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis[:x.shape[1]].view(shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


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


class MultiAttn(nn.Module):
    def __init__(self, d_model, num_head, max_len=512, dropout=0.1):
        super(MultiAttn, self).__init__()
        assert d_model % num_head == 0
        self.num_head = num_head
        self.head_size = d_model // num_head
        self.head_scale = math.sqrt(self.head_size)

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.ff = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis, mask, prefix_kv=None):
        x_proj = self.proj(x)
        proj_shape = x_proj.shape[:-1] + (self.num_head, self.head_size * 3)
        x_proj = x_proj.contiguous().view(proj_shape).transpose(1, 2)

        assert x_proj.size(1) == self.num_head
        assert x_proj.size(2) == x.size(1)
        assert x_proj.size(3) == self.head_size * 3

        base_q, base_k, v = x_proj.split(self.head_size, dim=-1)

        if prefix_kv is not None:
            pk, pv = prefix_kv
            base_k = torch.cat((pk, base_k), dim=-2)
            v = torch.cat((pv, v), dim=-2)

        next_prefix_kv = torch.stack((base_k, v))

        q, k = apply_rotary_emb(base_q, base_k, freqs_cis)

        attn_o = self.attn(q, k, v, mask)
        attn_o = attn_o.\
            transpose(1, 2).\
            contiguous()

        merged_shape = attn_o.shape[:-2] + (x.size(-1), )
        attn_o = attn_o.view(merged_shape)

        ff_o = self.ff(attn_o)
        return ff_o, next_prefix_kv

    def attn(self, q, k, v, mask):
        scores = q.matmul(k.transpose(-2, -1)) / self.head_scale

        tmp_len, seq_len = scores.shape[-2:]
        scores = scores.masked_fill(
            mask[..., seq_len-tmp_len:seq_len, :seq_len] == 0, min_fp16)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return scores.matmul(v)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super(MLP, self).__init__()
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.ff2(self.gelu(self.ff1(x)))


class Block(nn.Module):
    def __init__(self, d_model, num_head, max_len, dropout=0.1):
        super(Block, self).__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiAttn(d_model, num_head, max_len, dropout)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 4)

    def forward(self, x, freqs_cis, mask, prefix_kv=None):
        attn_o, prefix_kv = self.attn(
            self.ln1(x),
            freqs_cis,
            mask,
            prefix_kv
        )
        x = x + attn_o
        x = x + self.mlp(self.ln2(x))
        return x, prefix_kv


class MyModel(nn.Module):
    def __init__(self, vocab, pad_token_id, d_model=2560, num_head=32, max_len=2048, dropout=0.1, num_block=24):
        super(MyModel, self).__init__()
        self.vocab_emb = nn.Embedding(vocab, d_model)
        self.freqs_cis = precompute_freqs_cis(d_model // num_head, max_len * 2)

        self.blocks = nn.ModuleList([
            Block(d_model, num_head, max_len, dropout)
            for _ in range(num_block)
        ])
        self.ln = LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab, bias=False)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def raw(self, x, prefix_kv_list=None):
        if prefix_kv_list is None:
            seq_len = 0
            prefix_kv_list = [None] * len(self.blocks)
        else:
            seq_len = prefix_kv_list[0][0].size(-2)

        vemb_o = self.vocab_emb(x)
        self.freqs_cis = self.freqs_cis.to(vemb_o.device)
        seq_len_new = seq_len + x.size(-1)

        freqs_cis = self.freqs_cis[seq_len: seq_len_new]

        mask = 1 - torch.triu(
            torch.ones(
                (1, 1, seq_len_new, seq_len_new),
                device=x.device
            ),
            diagonal=1
        )

        x_rep = vemb_o

        next_prefix_kv_list = []
        for layer, prefix_kv in zip(self.blocks, prefix_kv_list):
            x_rep, next_prefix_kv = layer(x_rep, freqs_cis, mask, prefix_kv)
            next_prefix_kv_list.append(next_prefix_kv)

        x_rep = self.ln(x_rep)
        return x_rep, next_prefix_kv_list

    def forward(self, x, y=None, prefix_kv_list=None, num_micro_batch=1):
        x_rep, next_prefix_kv_list = self.raw(x, prefix_kv_list)
        y_pred = self.decoder(x_rep)
        if y is None:
            return y_pred, next_prefix_kv_list
        else:
            loss = self.loss_fn(
                y_pred.contiguous().view(-1, y_pred.size(-1)),
                y.contiguous().view(-1)
            )
            return loss / num_micro_batch
