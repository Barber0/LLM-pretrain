from myllm_model import Block, LayerNorm
import torch.nn as nn
import torch


class PosEmbWrap(nn.Embedding):
    def forward(self, x):
        pos_ids = torch.arange(0, x.size(-2), device=x.device)
        pos_ids = pos_ids.unsqueeze(0).expand(x.shape[:-1])
        out = super().forward(pos_ids)
        out += x
        return out


class WordEmbWrap(nn.Embedding):
    def forward(self, x):
        out = super().forward(x)
        return out


class BlockWrap(Block):
    def forward(self, x):
        out = super().forward(x=x)
        return out[0]


class LayerNormWrap(LayerNorm):
    def forward(self, x):
        out = super().forward(x)
        return out


class LinearWrap(nn.Linear):
    def forward(self, x):
        out = super().forward(x)
        return out


# class LossWrap(nn.CrossEntropyLoss):
#     def forward(self, inputs):
#         y_pred, y = inputs
#         out = super().forward(
#             y_pred.contiguous().view(-1, y_pred.size(-1)),
#             y.contiguous().view(-1)
#         )
#         return (out, y)

def build_pipe(
    vocab,
    d_model=768,
    num_head=12,
    max_len=512,
    num_block=12
):
    return [
        WordEmbWrap(vocab, d_model),
        PosEmbWrap(max_len, d_model),
    ] + [
        BlockWrap(d_model, num_head, max_len)
        for _ in range(num_block)
    ] + [
        LayerNormWrap(d_model),
        LinearWrap(d_model, vocab)
    ]
