import os
from time import time
from torch.utils.tensorboard import SummaryWriter

from myllm_model import MyModel
from data_loader import BcdsDataset
from mytkn import get_tkn
import deepspeed


def save_chkpt(
    name,
    model_eng,
    mount_dir,
    md_path,
    md_tag
):
    model_eng.save_checkpoint(md_path, tag=md_tag)
    with open(f'{mount_dir}/progress.txt', 'w') as f:
        f.write(name)


deepspeed.init_distributed(dist_backend='nccl')

mount_dir = '..'

saved_md_name = 'myllm-1B'
md_name = saved_md_name

md_home = 'gpt-model'
md_home = f'{mount_dir}/{md_home}'
os.makedirs(md_home, exist_ok=True)

saved_md_path = f'{md_home}/{saved_md_name}'
md_path = f'{md_home}/{saved_md_name}'
md_tag = 'arch_latest'

ds_home = '/root/autodl-tmp/content/drive/MyDrive/webtext-datasets/arch/'

log_path = '/root/tf-logs'

ds_cfg_path = f'./ds_cfg.json'

print(f'''
    saved_model_path: {saved_md_path}
    model_path: {md_path}
    log_path: {log_path}
    ds_cfg_path: {ds_cfg_path}
''')

tkn, VOCAB_SIZE, START_SIGN, END_SIGN, START_ID, END_ID = get_tkn()

max_len = 512

base_model = MyModel(
    vocab=VOCAB_SIZE,
    d_model=1024,
    num_head=32,
    max_len=max_len,
    num_block=12,
    pad_token_id=tkn.pad_token_id
)

model = deepspeed.PipelineModule(
    layers=base_model.pipeline(),
    loss_fn=base_model.loss_fn,
    partition_method='parameters',
    num_stages=2
)

batch_period = 20

batch_size = 10

bcds_ds = BcdsDataset(ds_home, tkn, max_len, batch_size)

model_eng, _, _, _ = deepspeed.initialize(
    model=model,
    config=ds_cfg_path,
    model_parameters=[p for p in model.parameters() if p.requires_grad],
    training_data=bcds_ds
)

writer = SummaryWriter(log_dir=log_path)

stime = time()
period_loss = 0
i = 1

while True:
    loss = model_eng.train_batch()

    try:
        writer.add_scalar('Train Loss', loss, i)
    except Exception as e:
        print(f'\r{e}')

    period_loss += loss

    if i % batch_period == 0:
        time_period = time() - stime
        print(
            f'batch: {i}, time: {time_period}, loss: {period_loss / batch_period}')
        writer.flush()

        try:
            if i % 200 == 0:
                save_chkpt(str(i), model_eng, mount_dir, md_path, md_tag)
        except Exception as e:
            print(e)

        period_loss = 0
        stime = time()

    i += 1
