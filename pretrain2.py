from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer
from llama import TransformerBlock, ModelArgs, Transformer
from data_loader import DataLoader
import argparse
import torch
import lightning as L
import numpy as np
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import os
from time import time
import math
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
torch.set_float32_matmul_precision('high')

os.environ["OMP_NUM_THREADS"] = "1"
dim = 768
n_heads = 4
n_layers = 4
log_interval = 20
training_sample = 10000
num_epochs = 4

# Hyperparameters
learning_rate = 8e-4
batch_size = 10
weight_decay = 1e-1
beta1 = 0.965
beta2 = 0.99
grad_clip = 1.0
rho = 0.1

batch_period = 20

max_seq_len = 1024


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    torch.manual_seed(1)

    data_loader, num_rows = DataLoader(
        'openwebtext',
        '/root/autodl-tmp/content/drive/MyDrive/webtext-datasets/arch/',
        batch_size
    )

    tkn = GPT2Tokenizer.from_pretrained('./tokenizer')
    tkn.pad_token = '[PAD]'
    tkn.add_tokens(['[END]'])
    VOCAB_SIZE = tkn.vocab_size + 1

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy,
                            activation_checkpointing=TransformerBlock)

    fabric = L.Fabric(accelerator="cuda",
                      devices=args.num_nodes,
                      strategy=strategy)

    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    model_args = fabric.to_device(ModelArgs(dim=dim, n_layers=n_layers, n_heads=n_heads,
                                  vocab_size=VOCAB_SIZE,
                                  max_batch_size=batch_size,
                                  max_seq_len=max_seq_len))  # Update these parameters as needed

    model = fabric.to_device(Transformer(model_args))

    for param in model.parameters():
        param.requires_grad = True

    model = fabric.setup_module(model)
    opt = torch.optim.AdamW(model.parameters(
    ), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    opt = fabric.setup_optimizers(opt)

    total_steps = math.ceil(num_rows / batch_size)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=3e-4)

    loss_fn = CrossEntropyLoss(ignore_index=tkn.pad_token_id)

    model.train()

    period_loss = 0

    stime = time()
    for bidx, batch in enumerate(data_loader()):
        x_encoded = tkn.batch_encode_plus(
            batch,
            max_length=max_seq_len,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        base_ids, x_attn_mask = x_encoded['input_ids'], x_encoded['attention_mask']
        base_ids = fabric.to_device(base_ids)

        input_ids = base_ids[..., :-1]
        target_ids = base_ids[..., 1:]

        out = model.forward(input_ids, 0)
        loss = loss_fn(
            out.contiguous().view(-1, out.size(-1)),
            target_ids.contiguous().view(-1)
        )

        # if grad_clip != 0.0:
        #     fabric.clip_gradients(model, opt, max_norm=grad_clip)

        opt.step()
        sch.step()
        opt.zero_grad()

        period_loss += loss.item()

        next_bidx = bidx + 1
        if next_bidx % batch_period == 0:
            time_period = time() - stime
            avg_ntokens = x_attn_mask.sum() / x_attn_mask.size(0)
            pad_token_len = x_attn_mask.size(-1)
            print(
                'batch: %d, time: %.2f, ntokens: %.2f/%d, loss: %f' % (
                    next_bidx, time_period, avg_ntokens, pad_token_len, period_loss/batch_period)
            )

            period_loss = 0
            stime = time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_iter_path', type=str, default="")
    parser.add_argument('-o', '--optimizer', type=str, default="adamw")
    parser.add_argument('-n', '--num_nodes', type=int, default=2)

    args = parser.parse_args()
    main(args)
