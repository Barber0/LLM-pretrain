import math
import torch
import torch.nn as nn
import torch.nn.functional as F

min_fp16 = torch.finfo(torch.float16).min


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_as_broadcast(freq_cis, x):
    assert freq_cis.shape[-2:] == x.shape[-2:]
    return freq_cis.view(1, 1, x.shape[2], x.shape[3])


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

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model * 3)
        self.ff = nn.Linear(d_model, d_model)

    def forward(
        self,
            x,
            mask,
            freq_cis_q,
            freq_cis_k,
            prefix_kv,
    ):

        x_proj = self.proj(x)
        proj_shape = x_proj.shape[:-1] + (self.num_head, self.head_size * 3)
        x_proj = x_proj.contiguous().view(proj_shape).transpose(1, 2)

        assert x_proj.size(1) == self.num_head
        assert x_proj.size(2) == x.size(1)
        assert x_proj.size(3) == self.head_size * 3

        q, k, v = x_proj.split(self.head_size, dim=-1)

        if prefix_kv is not None:
            pk, pv = prefix_kv
            k = torch.cat((pk, k), dim=-2)
            v = torch.cat((pv, v), dim=-2)

        next_prefix_kv = torch.stack((k, v))

        q = apply_rotary(q,  freq_cis_q)
        k = apply_rotary(q,  freq_cis_k)
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
            mask,
            freq_cis_q,
            freq_cis_k,
            prefix_kv,
    ):
        attn_o, prefix_kv = self.attn(
            self.ln1(x), mask, freq_cis_q, freq_cis_k, prefix_kv)
        x = x + attn_o
        x = x + self.mlp(self.ln2(x))
        return x, prefix_kv


class LLM(nn.Module):
    def __init__(self, vocab, pad_token_id, d_model=768, num_head=12, max_len=512, ext_factor=2, dropout=0.1, num_block=12):
        super(LLM, self).__init__()

        head_size = d_model//num_head
        ext_seq_len = max_len*ext_factor
        self.freq_cis = precompute_freqs_cis(head_size, ext_seq_len)
        self.mask = create_mask(ext_seq_len)

        self.emb = nn.Embedding(vocab, d_model)

        self.blocks = nn.ModuleList([
            Block(d_model, num_head, dropout)
            for _ in range(num_block)
        ])
        self.ln = LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab, bias=False)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def raw(self, x, prefix_kv_list=None):
        x_rep = self.emb(x)

        if prefix_kv_list is None:
            seq_start = 0
            prefix_kv_list = [None] * len(self.blocks)
        else:
            seq_start = prefix_kv_list[0][0].size(-2)

        seq_end = seq_start+x.size(-1)

        freq_cis_k = self.freq_cis[:seq_end]
        freq_cis_q = freq_cis_k if seq_start == 0 else self.freq_cis[seq_start:seq_end]
        tmp_mask = self.mask[..., seq_start:seq_end, :seq_end]

        freq_cis_k = freq_cis_k.to(x.device)
        freq_cis_q = freq_cis_q.to(x.device)
        tmp_mask = tmp_mask.to(x.device)

        next_prefix_kv_list = []
        for layer, prefix_kv in zip(self.blocks, prefix_kv_list):
            x_rep, next_prefix_kv = layer.forward(
                x_rep,
                tmp_mask,
                freq_cis_q,
                freq_cis_k,
                prefix_kv
            )
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
            ) / num_micro_batch
            return loss