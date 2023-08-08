import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from data_obj.model_args import ModelArgs, PositionEmbeddingType
from position_encoding2 import RoPE

try:
    from flash_attn import flash_attn_func
    USE_FLASH_ATTN = True
except ImportError:
    USE_FLASH_ATTN = False


class LayerNorm(nn.Module):
    def __init__(self, hidden_states, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(hidden_states))
        self.b = nn.Parameter(torch.zeros(hidden_states))

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_std = (x - x_mean).pow(2).mean(-1, keepdim=True)
        return self.w * (x - x_mean) / (x_std + self.eps) + self.b


class MLP(nn.Module):
    def __init__(self, attn_hidden_states, ff_hidden_states):
        super(MLP, self).__init__()
        self.ff1 = nn.Linear(attn_hidden_states, ff_hidden_states)
        self.ff2 = nn.Linear(ff_hidden_states, attn_hidden_states)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.ff2(self.gelu(self.ff1(x)))


class Attention(nn.Module):
    def __init__(
        self,
        hidden_states: int,
        n_heads: int,
        rope: RoPE,
        dropout_prob: float = 0.1
    ):
        super(Attention, self).__init__()
        self.hidden_states = hidden_states
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob
        self.rope = rope

        self.head_states = hidden_states // n_heads
        self.head_scale = math.sqrt(self.head_states)

        self.dropout = nn.Dropout(dropout_prob)
        self.proj = nn.Linear(hidden_states, hidden_states*3)
        self.ff = nn.Linear(hidden_states, hidden_states)

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        prefix_kv: Tensor = None
    ):
        head_dim_idx = 3
        qkv = self.proj(x)
        next_prefix_kv = None
        if mask is None and USE_FLASH_ATTN:
            seq_len_dim_idx = 1
            q, k, v = rearrange(
                qkv, 'b n (c h d) -> c b n h d', c=3, h=self.n_heads)

            q, k = self.rope.apply_rotary_for_qk(
                q, k, seq_len_dim_idx, head_dim_idx)

            attn_o = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout_prob,
                causal=True
            )
            attn_o = rearrange(attn_o, 'b n h d -> b n (h d)')
        else:
            seq_len_dim_idx = 2
            q, k, v = rearrange(
                qkv, 'b n (c h d) -> c b h n d', c=3, h=self.n_heads)
            if prefix_kv is not None:
                pk, pv = prefix_kv
                k = torch.cat((pk, k), dim=seq_len_dim_idx)
                v = torch.cat((pv, v), dim=seq_len_dim_idx)
            next_prefix_kv = torch.stack((k, v))

            q, k = self.rope.apply_rotary_for_qk(
                q, k, seq_len_dim_idx, head_dim_idx)

            attn_o = self.attn(q, k, v, mask)
            attn_o = rearrange(attn_o, 'b h n d -> b n (h d)')

        return self.ff(attn_o), next_prefix_kv

    def attn(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        scores = torch.einsum('...qd, ...kd -> ...qk', q, k) / self.head_scale
        mask_value = torch.finfo(q.dtype).min
        scores = scores.masked_fill(mask == 0, mask_value)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, v)


class Block(nn.Module):
    def __init__(self, hidden_states: int, n_heads: int, rope: RoPE, dropout_prob: float = 0.1):
        super(Block, self).__init__()
        self.hidden_states = hidden_states
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob

        self.attn = Attention(
            self.hidden_states,
            self.n_heads,
            rope,
            self.dropout_prob,
        )
        self.ln1 = LayerNorm(self.hidden_states)
        self.ln2 = LayerNorm(self.hidden_states)
        self.mlp = MLP(self.hidden_states, self.hidden_states*4)

    def forward_with_prefix(
        self,
        x: Tensor,
        mask: Tensor = None,
        prefix_kv: Tensor = None
    ):
        attn_o, prefix_kv = self.attn.forward(
            self.ln1(x),
            mask,
            prefix_kv
        )
        x = x + attn_o
        x = x + self.mlp(self.ln2(x))
        return x, prefix_kv

    def forward(self, x: Tensor):
        return self.forward_with_prefix(x, None, None)[0]


class SFEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pad_token_id: int,
        n_layers: int,
        base_mask: torch.Tensor,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.pad_token_id = pad_token_id
        self.n_layers = n_layers
        self.base_mask = base_mask

    def forward_with_prefix(
        self,
        input: Tensor,
        prefix_kv_list: List[Tensor] = None,
        generate: bool = not USE_FLASH_ATTN,
    ):
        emb_out = super().forward(input)
        start_idx, prefix_kv_list = self._process_prefix_kv_list(
            prefix_kv_list)
        seq_len = input.size(-1)
        end_idx = start_idx+seq_len

        mask = None
        if generate:
            mask = (input != self.pad_token_id)
            if start_idx > 0:
                mask = torch.cat((
                    torch.ones(mask.size(0), start_idx,
                               device=mask.device).bool(),
                    mask,
                ), dim=-1)
            mask = repeat(mask, 'b n -> b 1 q n', q=seq_len)

            mask = mask & self.base_mask[..., start_idx:end_idx, :end_idx].to(
                input.device)

        return emb_out, mask, prefix_kv_list

    def forward(self, input: Tensor):
        return self.forward_with_prefix(input, None, False)[0]

    def _process_prefix_kv_list(self, prefix_kv_list: List[Tensor] = None):
        if prefix_kv_list is None:
            seq_start = 0
            prefix_kv_list = [None] * self.n_layers
        else:
            seq_start = prefix_kv_list[0][0].size(-2)
        return seq_start, prefix_kv_list


class CELoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor):
        return super().forward(
            input.contiguous().view(-1, input.size(-1)),
            target.contiguous().view(-1)
        )


class SFLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        args: ModelArgs
    ):
        super(SFLLM, self).__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.args = args
        self._build_model()

    def _init_rope_emb(self):
        assert self.args is not None
        args_pos_emb = self.args.position_encoding
        if args_pos_emb == PositionEmbeddingType.COMPLEX_ROPE:
            from position_encoding2 import ComplexRoPE
            rope_constructor = ComplexRoPE
        elif args_pos_emb == PositionEmbeddingType.SIMULATED_ROPE:
            from position_encoding2 import SimulatedRoPE
            rope_constructor = SimulatedRoPE
        else:
            raise Exception(
                f'Unsupported position embedding: {args_pos_emb}')

        assert self.args.hidden_states % self.args.n_heads == 0
        head_states = self.args.hidden_states // self.args.n_heads
        return rope_constructor(
            hidden_states=head_states,
            interpolate_factor=self.args.rope_interpolate_factor,
            theta=self.args.rope_theta,
            max_len=self.args.max_len,
            ext_factor=self.args.ext_factor
        )

    def _build_model(self):
        self.emb = SFEmbedding(
            self.vocab_size,
            self.args.hidden_states,
            self.pad_token_id,
            self.args.n_layers,
            self.create_mask(self.args.max_len * self.args.ext_factor),
        )

        base_rope_emb = self._init_rope_emb()

        self.blocks = nn.ModuleList([
            Block(
                self.args.hidden_states,
                self.args.n_heads,
                base_rope_emb,
            ) for _ in range(self.args.n_layers)
        ])

        self.ln = LayerNorm(self.args.hidden_states)

        self.decoder = None
        if not self.args.reuse_emb:
            self.decoder = nn.Linear(
                self.args.hidden_states, self.vocab_size, bias=False)

        self.loss_fn = CELoss(ignore_index=self.pad_token_id)

    @staticmethod
    def create_mask(max_len):
        return (1 - torch.triu(torch.ones((1, 1, max_len, max_len)), diagonal=1)).bool()

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor = None,
        prefix_kv_list: List[Tensor] = None,
        generate: bool = False
    ):
        x, mask, prefix_kv_list = self.emb.forward_with_prefix(
            input=input_ids,
            prefix_kv_list=prefix_kv_list,
            generate=generate,
        )

        next_prefix_kv_list = []
        for block, prefix_kv in zip(self.blocks, prefix_kv_list):
            x, next_prefix_kv = block.forward_with_prefix(
                x, mask, prefix_kv)
            next_prefix_kv_list.append(next_prefix_kv)

        ln_out = self.ln(x)

        if self.decoder is None:
            y_pred = torch.matmul(ln_out, self.emb.weight.transpose(0, 1))
        else:
            y_pred = self.decoder(ln_out)

        if target_ids is None:
            return y_pred, next_prefix_kv_list

        return self.loss_fn(
            y_pred,
            target_ids
        )

    def pipeline_and_loss_fn(self):
        if self.decoder is None:
            raise Exception("Decoder(Linear) not found.")

        module_list = [self.emb] + [block for block in self.blocks] + [
            self.ln,
            self.decoder
        ]
        return nn.Sequential(*module_list), self.loss_fn
