from datasets import load_from_disk
import random
from consts import *
from torch.utils.data import Dataset

def prepare_conversation(data, tkn):
    conversation = data['conversation']
    utterances = []
    for x in conversation:
        utterances.append(x['human'])
        utterances.append(x['assistant'])

    utterances.append('')
    tmp_txt = f' {tkn.eos_token} '.join(utterances)
    return tmp_txt

ds_handlers_map = {
    'openwebtext': lambda line, tkn: line['text'],
    'self_instruct': lambda line, tkn: f'''{line['prompt']} {line['completion']} {tkn.eos_token}''',
    'ultrachat': prepare_conversation,
}

def generate_random_sequence(n):
    seq = list(range(n))
    random.shuffle(seq)
    return seq


def split_and_join(txt, max_len=1024, overlap_factor=4):
    segs = txt.split(' ')
    if len(segs) > max_len:
        start_idx = 0
        out_list = []
        overlap_len = max_len // overlap_factor
        while start_idx < len(segs):
            remain_len = len(segs) - start_idx
            if start_idx > 0 and remain_len <= overlap_len:
                break
            next_idx = start_idx + max_len
            rejoin_txt = ' '.join(segs[start_idx: next_idx])
            out_list.append(rejoin_txt)
            next_idx -= overlap_len
            start_idx = next_idx
        return out_list
    return [txt]


def DataLoader(ds_name, ds_path, max_len, tokenizer, overlap_factor=0, batch_size=5):
    data = load_from_disk(ds_path)['train']
    idx_list = generate_random_sequence(data.num_rows)
    preprocess = ds_handlers_map[ds_name]

    def _iter():
        batch = []
        for idx in idx_list:
            v = data[idx]
            
            processed = preprocess(v, tokenizer)
            if overlap_factor > 0:
                chunk = split_and_join(
                    processed,
                    max_len=max_len,
                    overlap_factor=overlap_factor
                )
            else:
                chunk = [processed]
                
            for line in chunk:
                batch.append(line)
                if len(batch) >= batch_size:
                    out = batch
                    batch = []
                    yield out

    return _iter


class BaseDataset(Dataset):
    def __init__(self, ds_name, ds_path, max_len):
        super().__init__()
