import torch
import os
import deepspeed

from transformers import GPT2Tokenizer
from myllm_model import MyModel

from data_loader import BcdsLoader
from train_epoch import train_epoch

from torch.utils.tensorboard import SummaryWriter

mount_dir = '..'

saved_md_name = 'myllm2-1B'
md_name = saved_md_name

md_home = 'llm-model'
md_home = f'{mount_dir}/{md_home}'
os.makedirs(md_home, exist_ok=True)

saved_md_path = f'/root/autodl-tmp/myllm3-2B'
md_path = saved_md_path
md_tag = 'main'

# ds_home = f'{mount_dir}/self_instruct'
ds_home = '/root/autodl-tmp/content/drive/MyDrive/webtext-datasets/arch/'

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
tkn.add_tokens(['[END]'])
VOCAB_SIZE = tkn.vocab_size

max_len = 512

model = MyModel(
    vocab=VOCAB_SIZE,
    pad_token_id=tkn.pad_token_id,
    d_model=2560,
    num_head=32,
    num_block=24,
    max_len=max_len
)

# Temporarily load backup model
chkpt = torch.load('/root/autodl-tmp/backup/myllm3-2B-94800.pt')
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


# start_batch = 81601
start_batch = 94801
num_epochs = 1
start_epoch = 0

batch_size = 12
data_loader = BcdsLoader(ds_home, batch_size=batch_size)

batch_period = 50

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
    writer,
    max_len
)
