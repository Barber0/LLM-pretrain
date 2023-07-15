from datasets import load_from_disk
import random
import json

main_seed = 888
random.seed(main_seed)

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


def SelfInstructLoader(ds_path, max_len, overlap_factor, batch_size=5):
    data = load_from_disk(ds_path)['train']
    idx_list = generate_random_sequence(data.num_rows)

    def _iter():
        batch = []
        for idx in idx_list:
            v = data[idx]
            chunk = split_and_join(f'''{v['prompt']} {v['completion']}''', max_len=max_len, overlap_factor=overlap_factor)
            for line in chunk:
                batch.append(line)
                if len(batch) >= batch_size:
                    out = batch
                    batch = []
                    yield out
    
    return _iter


def BcdsLoader(ds_path, batch_size=5):
    data = load_from_disk(ds_path)

    def _iter():
        batch = []
        for i, v in enumerate(data['train']):
            batch.append(v['text'])
            if (i + 1) % batch_size == 0:
                out = batch
                batch = []
                yield out
    return _iter

def WikiLoader(ds_path, max_len, overlap_factor, batch_size=5):
    data = load_from_disk(ds_path)['train']
    idx_list = generate_random_sequence(data.num_rows)

    def _iter():
        batch = []
        for idx in idx_list:
            v = data[idx]
            chunk = split_and_join(v['text'], max_len=max_len, overlap_factor=overlap_factor)
            for line in chunk:
                batch.append(line)
                if len(batch) >= batch_size:
                    out = batch
                    batch = []
                    yield out
    
    return _iter