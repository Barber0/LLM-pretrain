import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from data_obj.model_args import ModelArgs, PositionEmbeddingType
from position_encoding import RoPE

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
        rope_fn: RoPE.RoPEFunctionType,
        dropout_prob: float = 0.1
    ):
        super(Attention, self).__init__()
        self.hidden_states = hidden_states
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob
        self.rope_fn = rope_fn

        self.head_states = hidden_states // n_heads
        self.head_scale = math.sqrt(self.head_states)

        self.dropout = nn.Dropout(dropout_prob)
        self.qkv_proj = nn.Linear(hidden_states, hidden_states*3)
        self.out_proj = nn.Linear(hidden_states, hidden_states)

    def forward(
        self,
        x: Tensor,
        freq_cis_q: Tensor,
        freq_cis_k: Tensor,
        expanded_attn_masks: Tensor,
        casual_mask=None,
        prefix_kv=None
    ):
        head_dim_idx = 3
        qkv = self.qkv_proj(x)
        next_prefix_kv = None
        if casual_mask is None and USE_FLASH_ATTN:
            seq_len_dim_idx = 1
            q, k, v = rearrange(
                qkv, 'b n (c h d) -> c b n h d', c=3, h=self.n_heads)
            q = self.rope_fn(q, freq_cis_q, seq_len_dim_idx, head_dim_idx)
            k = self.rope_fn(k, freq_cis_k, seq_len_dim_idx, head_dim_idx)
            attn_o = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout_val,
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

            q = self.rope_fn(q, freq_cis_q, seq_len_dim_idx, head_dim_idx)
            k = self.rope_fn(k, freq_cis_k, seq_len_dim_idx, head_dim_idx)
            attn_o = self.attn(q, k, v, expanded_attn_masks, casual_mask)
            attn_o = rearrange(attn_o, 'b h n d -> b n (h d)')

        return self.out_proj(attn_o), next_prefix_kv

    def attn(self, q: Tensor, k: Tensor, v: Tensor, attn_masks: Tensor, casual_mask: Tensor):
        scores = torch.einsum('...qd, ...kd -> ...qk', q, k) / self.head_scale
        mask_value = -torch.finfo(q.dtype).max
        scores = scores.masked_fill(casual_mask == 0, mask_value)
        scores = scores.masked_fill(attn_masks == 0, mask_value)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, v)


@dataclass
class BlockData:
    x: Tensor
    freq_cis_q: Tensor
    freq_cis_k: Tensor
    attn_mask: Tensor
    casual_mask: Tensor


class Block(Attention):
    ln1: LayerNorm = None
    ln2: LayerNorm = None
    mlp: MLP = None

    def ensure_ready(self):
        if self.ln1 is None:
            self.ln1 = LayerNorm(self.hidden_states)
        if self.ln2 is None:
            self.ln2 = LayerNorm(self.hidden_states)
        if self.mlp is None:
            self.mlp = MLP(self.hidden_states, self.hidden_states*4)

    def forward_with_prefix(
        self,
        data: BlockData,
        prefix_kv=None
    ):
        self.ensure_ready()
        attn_o, prefix_kv = super().forward(
            self.ln1(data.x),
            data.freq_cis_q,
            data.freq_cis_k,
            data.attn_mask,
            data.casual_mask,
            prefix_kv
        )
        x += attn_o
        x += self.mlp(self.ln2(x))
        return BlockData(
            x=data.x,
            freq_cis_q=data.freq_cis_q,
            freq_cis_k=data.freq_cis_k,
            attn_mask=data.attn_mask,
            casual_mask=data.casual_mask,
        ), prefix_kv

    def forward(self, data: BlockData):
        return self.forward_with_prefix(data=data)[0]


class SFEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pad_token_id: int,
        n_layers: int,
        base_mask: torch.Tensor,
        build_freq_cis_fn: RoPE.FreqCisBuilderType
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.pad_token_id = pad_token_id
        self.n_layers = n_layers
        self.base_mask = base_mask
        self.build_freq_cis_fn = build_freq_cis_fn

    def forward_with_prefix(
        self,
        input: Tensor,
        prefix_kv_list: List[Tensor] = None,
        generate: bool = False,
    ):
        emb_out = super().forward(input)
        start_idx, prefix_kv_list = self._process_prefix_kv_list(
            prefix_kv_list)
        seq_len = input.size(-1)
        end_idx = start_idx+seq_len
        freq_cis_q = self.build_freq_cis_fn(seq_len, start_idx)
        freq_cis_k = freq_cis_q if start_idx == 0 else \
            self.build_freq_cis_fn(end_idx, 0)

        attn_mask = (input != self.pad_token_id).int()
        if start_idx > 0:
            attn_mask = torch.cat((
                torch.ones(attn_mask.size(0), start_idx),
                attn_mask,
            ), dim=-1)
        attn_mask = repeat(attn_mask, 'b n -> b 1 q n', q=seq_len)

        casual_mask = None
        if generate:
            casual_mask = self.base_mask[..., ..., start_idx:end_idx, :end_idx].to(
                input.device)

        return BlockData(
            x=emb_out,
            freq_cis_q=freq_cis_q,
            freq_cis_k=freq_cis_k,
            attn_mask=attn_mask,
            casual_mask=casual_mask
        ), prefix_kv_list

    def forward(
        self,
        input: Tensor
    ):
        return self.forward_with_prefix(input, None, False)[0]

    def _process_prefix_kv_list(self, prefix_kv_list: List[Tensor] = None):
        if prefix_kv_list is None:
            seq_start = 0
            prefix_kv_list = [None] * self.n_layers
        else:
            seq_start = prefix_kv_list[0][0].size(-2)
        return seq_start, prefix_kv_list


class CELoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(
            input.contiguous().view(-1, input.size(-1)),
            target.contiguous().view(-1)
        )


class SFLLM(nn.Module):
    rope_emb: RoPE

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
        self._init_rope_emb()
        self._build_model()

    def _init_rope_emb(self):
        assert self.args is not None
        args_pos_emb = self.args.position_encoding
        if args_pos_emb == PositionEmbeddingType.COMPLEX_ROPE:
            from position_encoding import ComplexRoPE
            rope_constructor = ComplexRoPE
        elif args_pos_emb == PositionEmbeddingType.SIMULATED_ROPE:
            from position_encoding import SimulatedRoPE
            rope_constructor = SimulatedRoPE
        elif args_pos_emb == PositionEmbeddingType.LUCIDRAINS_ROPE:
            from position_encoding import LucidrainsRoPE
            rope_constructor = LucidrainsRoPE
        else:
            raise Exception(
                f'Unsupported position embedding: {args_pos_emb}')

        self.rope_emb = rope_constructor(
            self.args.hidden_states,
            self.args.rope_interpolate_factor,
            self.args.rope_theta,
        )

    def _build_model(self):
        assert self.rope_emb is not None
        self.vocab_emb = SFEmbedding(
            self.vocab_size,
            self.args.hidden_states,
            self.pad_token_id,
            self.args.n_layers,
            self.create_mask(self.args.max_len * self.args.ext_factor),
            self.rope_emb.build_freq_cis
        )

        self.blocks = nn.ModuleList([
            Block(
                self.args.hidden_states,
                self.args.n_heads,
                self.rope_emb.apply_rotary,
            ) for _ in range(self.args.n_layers)
        ])

        self.layerNorm = LayerNorm(self.args.hidden_states)
        self.decoder = nn.Linear(
            self.args.hidden_states, self.vocab_size, bias=False)
        self.loss_fn = CELoss(ignore_index=self.pad_token_id)

    @staticmethod
    def create_mask(max_len):
        return 1 - torch.triu(torch.ones((1, 1, max_len, max_len)), diagonal=1)

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor = None,
        prefix_kv_list: List[Tensor] = None,
        generate: bool = False
    ):
        data_block, prefix_kv_list = self.vocab_emb.forward_with_prefix(
            input=input_ids,
            prefix_kv_list=prefix_kv_list,
            generate=generate,
        )

        next_prefix_kv_list = []
        for block, prefix_kv in zip(self.blocks, prefix_kv_list):
            data_block, next_prefix_kv = block.forward(data_block, prefix_kv)
            next_prefix_kv_list.append(next_prefix_kv)

        ln_out = self.layerNorm(data_block.x)
        y_pred = self.decoder(ln_out)

        if target_ids is None:
            return y_pred, next_prefix_kv_list

        return self.loss_fn(y_pred, target_ids)

    def pipeline_and_loss_fn(self):
        return nn.Sequential(nn.ModuleList([self.vocab_emb] + [block for block in self.blocks] + [
            self.layerNorm,
            self.decoder
        ])), self.loss_fn