import os
from time import time
import deepspeed
from transformers import GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter

from myllm_model import MyModel
from data_loader import SelfInstructLoader

mount_dir = '.'

saved_md_name = 'myllm-1B'
md_name = saved_md_name

md_home = 'gpt-model'
md_home = f'{mount_dir}/{md_home}'
os.makedirs(md_home, exist_ok=True)

saved_md_path = f'{md_home}/{saved_md_name}'
# saved_md_path = '/root/autodl-tmp/myllm-chkpt'
# md_path = '/root/autodl-tmp/myllm-chkpt'
md_path = f'{md_home}/{saved_md_name}'
md_tag = 'arch_latest'

ds_home = './self_instruct'
# ds_home = '/root/autodl-tmp/content/drive/MyDrive/webtext-datasets/arch/'

log_path = '/root/tf-logs'

tkn_path = f'{mount_dir}/gpt2-tkn'

ds_cfg_path = f'{mount_dir}/ds_cfg.json'

print(f'''
    saved_model_path: {saved_md_path}
    model_path: {md_path}
    log_path: {log_path}
    tkn_path: {tkn_path}
    ds_cfg_path: {ds_cfg_path}
''')

START_SIGN = '<start>'
END_SIGN = '<end>'

tkn = GPT2Tokenizer.from_pretrained(tkn_path)
tkn.pad_token = '[PAD]'
tkn.add_tokens([START_SIGN, END_SIGN])

START_ID, END_ID = tkn.convert_tokens_to_ids([START_SIGN, END_SIGN])
VOCAB_SIZE = tkn.vocab_size + 2

model = MyModel(
    vocab=VOCAB_SIZE,
    pad_token_id=tkn.pad_token_id,
    d_model=1280,
    num_head=32,
    num_block=60
)

# Temporarily load backup model
import torch
chkpt = torch.load('./gpt-model/myllm-1B-39000.pt')
model.load_state_dict(chkpt['module'])
# End of tmp load

model_eng, _, _, _ = deepspeed.initialize(
    model=model,
    config=ds_cfg_path
)

# Normaly load checkpoint
# load_tag = model_eng.load_checkpoint(
#     saved_md_path, md_tag)
# End of normal load


# start_batch = 39001
num_epochs = 5
start_batch = 0

# batch_size = 15
batch_size = 10
data_loader = SelfInstructLoader(ds_home, batch_size=batch_size)

batch_period = 20

stime = time()
total_loss = 0
period_loss = 0
period_tokens = 0

writer = SummaryWriter(log_dir=log_path)

def train_epoch(ep):
    for i, batch in enumerate(data_loader()):
        if i < start_batch:
            continue

        x_encoded = tkn.batch_encode_plus(
            [f'{START_SIGN} {line}' for line in batch],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids, x_attn_mask = x_encoded['input_ids'], x_encoded['attention_mask']

        target_ids = tkn.batch_encode_plus(
            [f'{line} {END_SIGN}' for line in batch],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )['input_ids']

        input_ids = input_ids.cuda()
        target_ids = target_ids.cuda()

        loss = model_eng(input_ids, target_ids)
        model_eng.backward(loss)
        model_eng.step()

        try:
            writer.add_scalar('Train Loss', loss, i)
        except Exception as e:
            print(f'\r{e}')

        total_loss += loss
        period_loss += loss

        if i > 0 and i % batch_period == 0:
            time_period = time() - stime
            avg_ntokens = x_attn_mask.sum() / x_attn_mask.size(0)
            pad_token_len = x_attn_mask.size(-1)
            print(f'\rbatch: {i}, time: {time_period}, avg_ntokens: {avg_ntokens},'
                f' loss: {period_loss}, per_loss: {period_loss / batch_period},'
                f' pad_len: {pad_token_len}')
            writer.flush()

            try:
                if i % 200 == 0:
                    model_eng.save_checkpoint(md_path, tag=md_tag)
                    with open(f'./progress.txt', 'w') as f:
                        f.write(str(i) + '\n')
            except Exception as e:
                print(e)

            period_loss = 0
            stime = time()

    model_eng.save_checkpoint(md_path, tag=md_tag)
    with open(f'./progress.txt', 'w') as f:
        f.write(f'epoch-{ep}\n')
