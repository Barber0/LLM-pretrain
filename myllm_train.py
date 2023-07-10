import torch
import os
import deepspeed
from transformers import GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter

from myllm_model import MyModel
from data_loader import SelfInstructLoader
from train_epoch import train_epoch

mount_dir = '..'

saved_md_name = 'myllm2-1B'
md_name = saved_md_name

md_home = 'llm-model'
md_home = f'{mount_dir}/{md_home}'
os.makedirs(md_home, exist_ok=True)

saved_md_path = f'{md_home}/{saved_md_name}'
# saved_md_path = '/root/autodl-tmp/myllm-22000/myllm2-1B/'
md_path = saved_md_path
md_tag = 'main'

ds_name = 'self_instruct'
# ds_name = 'webtext'
ds_home = f'/root/autodl-tmp/content/drive/MyDrive/datasets/{ds_name}/'

log_path = '/root/tf-logs'

tkn_path = f'./tokenizer'

ds_cfg_path = f'./ds_cfg_backup.json'

print(f'''
    saved_model_path: {saved_md_path}
    model_path: {md_path}
    log_path: {log_path}
    tkn_path: {tkn_path}
    ds_cfg_path: {ds_cfg_path}
''')

tkn = GPT2Tokenizer.from_pretrained(tkn_path)
tkn.pad_token = '[PAD]'
VOCAB_SIZE = tkn.vocab_size

model = MyModel(
    vocab=VOCAB_SIZE,
    pad_token_id=tkn.pad_token_id,
    d_model=1280,
    num_head=32,
    num_block=60
)

# Temporarily load backup model
chkpt = torch.load('/root/autodl-tmp/myllm2-1B-finetune.pt')
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


start_batch = 0
num_epochs = 1
start_epoch = 0

batch_size = 10
data_loader = SelfInstructLoader(ds_home, batch_size=batch_size)

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
    tkn,
    data_loader,
    writer
)

# for ep in range(start_epoch+1, num_epochs):
#     train_epoch(
#         ep,
#         model_eng,
#         0,
#         batch_period,
#         md_path,
#         md_tag,
#         mount_dir,
#         START_SIGN,
#         END_SIGN,
#         tkn,
#         data_loader,
#         writer
#     )
