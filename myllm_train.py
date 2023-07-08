import torch
import os
from time import time
import deepspeed
from transformers import GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter

from myllm_model import MyModel
from data_loader import BcdsLoader
from train_epoch import train_epoch
from mytkn import get_tkn

mount_dir = '..'

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

# ds_home = f'{mount_dir}/self_instruct'
ds_home = '/root/autodl-tmp/content/drive/MyDrive/webtext-datasets/arch/'

log_path = '/root/tf-logs'

tkn_path = f'{mount_dir}/gpt2-tkn'

ds_cfg_path = f'./ds_cfg.json'

print(f'''
    saved_model_path: {saved_md_path}
    model_path: {md_path}
    log_path: {log_path}
    tkn_path: {tkn_path}
    ds_cfg_path: {ds_cfg_path}
''')

tkn, VOCAB_SIZE, START_SIGN, END_SIGN, START_ID, END_ID = get_tkn()

model = MyModel(
    vocab=VOCAB_SIZE,
    pad_token_id=tkn.pad_token_id,
    d_model=1024,
    num_head=32,
    num_block=18
)

# Temporarily load backup model
# chkpt = torch.load(f'{mount_dir}/gpt-model/myllm-1B-39000.pt')
# model.load_state_dict(chkpt['module'])
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
start_batch = 3601
start_epoch = 0

batch_size = 20
data_loader = BcdsLoader(ds_home, batch_size=batch_size)

batch_period = 20

writer = SummaryWriter(log_dir=log_path)

train_epoch(
    start_epoch,
    model_eng,
    start_batch,
    batch_period,
    md_path,
    md_tag,
    mount_dir,
    START_SIGN,
    END_SIGN,
    tkn,
    data_loader,
    writer
)

for ep in range(start_epoch+1, num_epochs):
    train_epoch(
        ep,
        model_eng,
        0,
        batch_period,
        md_path,
        md_tag,
        mount_dir,
        START_SIGN,
        END_SIGN,
        tkn,
        data_loader,
        writer
    )
