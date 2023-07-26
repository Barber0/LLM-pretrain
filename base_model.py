import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func

min_fp16 = torch.finfo(torch.float16).min


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

        self.dropout_val = dropout
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model * 3)
        self.ff = nn.Linear(d_model, d_model)

    def forward(self, x, prefix_kv=None, mask=None):
        x_proj = self.proj(x)

        next_prefix_kv = None
        if prefix_kv is None and mask is None:
            proj_shape = x_proj.shape[:-1] + (3, self.num_head, self.head_size)
            x_proj = x_proj.contiguous().view(proj_shape)
            attn_o = flash_attn_qkvpacked_func(
                qkv=x_proj,
                dropout_p=self.dropout_val,
                causal=True
            )
        else:
            # (batch_size, seq_len, 3, num_head, head_size)
            proj_shape = x_proj.shape[:-1] + (3, self.num_head, self.head_size)
            # (3, batch_size, num_head, seq_len, head_size)
            x_proj = x_proj.contiguous().view(proj_shape).permute(2, 0, 3, 1, 4)

            assert x_proj.ndim == 5
            assert x_proj.shape == (
                3,
                x.size(0),
                self.num_head,
                x.size(1),
                self.head_size
            )

            q, k, v = x_proj

            if prefix_kv is not None:
                pk, pv = prefix_kv
                k = torch.cat((pk, k), dim=-2)
                v = torch.cat((pv, v), dim=-2)

            next_prefix_kv = torch.stack((k, v))

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
    def __init__(self, d_model, num_head, max_len, dropout=0.1):
        super(Block, self).__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiAttn(d_model, num_head, max_len, dropout)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 4)

    def forward_with_prefix(self, x, prefix_kv=None, mask=None):
        attn_o, prefix_kv = self.attn(self.ln1(x), prefix_kv, mask)
        x = x + attn_o
        x = x + self.mlp(self.ln2(x))
        return x, prefix_kv

    def forward(self, x):
        return self.forward_with_prefix(x)[0]


class PositionEmbedding(nn.Embedding):
    def forward(self, x, seq_start=0, seq_end=None):
        assert x.ndim == 3
        if seq_end is None:
            seq_end = seq_start + x.size(1)
        pos_ids = torch.arange(seq_start, seq_end, device=x.device)
        pos_ids = pos_ids.unsqueeze(0).expand(x.size(0), -1)
        pos_o = super().forward(pos_ids)
        return x + pos_o


class LossWrapper(nn.CrossEntropyLoss):
    def forward(self, y_pred, target):
        return super().forward(
            y_pred.contiguous().view(-1, y_pred.size(-1)),
            target.contiguous().view(-1)
        )


class LLM(nn.Module):
    def __init__(self, vocab, pad_token_id, d_model=768, num_head=12, max_len=512, dropout=0.1, num_blocks=12):
        super(LLM, self).__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = PositionEmbedding(max_len, d_model)
        self.mask = self.create_mask(max_len)

        self.blocks = nn.ModuleList([
            Block(d_model, num_head, max_len, dropout)
            for _ in range(num_blocks)
        ])
        self.ln = LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab, bias=False)
        self.loss_fn = LossWrapper(ignore_index=pad_token_id)

    @staticmethod
    def create_mask(mask_size):
        return 1 - torch.triu(torch.ones((1, 1, mask_size, mask_size)), diagonal=1)

    def raw(self, x, prefix_kv_list=None, generate=False):
        emb_o = self.emb(x)

        if prefix_kv_list is None:
            seq_start = 0
            prefix_kv_list = [None] * len(self.blocks)
        else:
            seq_start = prefix_kv_list[0][0].size(-2)

        seq_end = seq_start + x.size(-1)

        x_rep = self.pos(emb_o, seq_start, seq_end)

        mask = None
        if generate:
            mask = self.mask[..., seq_start:seq_end, :seq_end].to(x_rep.device)

        next_prefix_kv_list = []
        for layer, prefix_kv in zip(self.blocks, prefix_kv_list):
            x_rep, next_prefix_kv = layer.forward_with_prefix(
                x_rep, prefix_kv, mask)
            next_prefix_kv_list.append(next_prefix_kv)

        x_rep = self.ln(x_rep)
        return x_rep, next_prefix_kv_list

    def forward(self, x, y=None, prefix_kv_list=None, generate=False):
        x_rep, next_prefix_kv_list = self.raw(x, prefix_kv_list, generate)
        y_pred = self.decoder(x_rep)
        if y is None:
            return y_pred, next_prefix_kv_list
        else:
            loss = self.loss_fn(y_pred, y)
            return loss

    def pipeline(self):
        module_list = [
            self.emb,
            self.pos
        ] + [block for block in self.blocks] + [
            self.ln,
            self.decoder
        ]
        return nn.Sequential(*module_list), self.loss_fn
