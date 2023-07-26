import random
from consts import *

line_sep_max_limit = 5

def prepare_conversation_for_ultrachat(data, tkn):
    conversation = data['conversation']
    utterances = []
    for x in conversation:
        utterances.append(x['human'])
        utterances.append(x['assistant'])

    utterances.append('')
    tmp_txt = f' {tkn.eos_token} '.join(utterances)
    return tmp_txt

def get_rand_rep_str(s, start=0, end=line_sep_max_limit):
    return s * random.randint(start, end)

def prepare_qa_for_alpaca(line, tkn):
    txt_input = line['input']
    if txt_input is None:
        txt_input = line['instruction']
    else:
        txt_input = line['instruction'] + get_rand_rep_str('\n', 1) + txt_input
    txt_output = line['output']
    return f'{txt_input} {tkn.eos_token} {txt_output} {tkn.eos_token}'

ds_handlers_map = {
    'openwebtext': lambda line, tkn: line['text'],
    'self_instruct': lambda line, tkn: f'''{line['prompt']} {line['completion']} {tkn.eos_token}''',
    'ultrachat': prepare_conversation_for_ultrachat,
    'alpaca': prepare_qa_for_alpaca
}

def load_from_ms(ds_path):
    from modelscope.msdatasets import MsDataset
    data = MsDataset.load(ds_path)
    idx_list = generate_random_sequence(len(data))
    return data, idx_list

def load_from_hf(ds_path):
    from datasets import load_from_disk
    data = load_from_disk(ds_path)['train']
    idx_list = generate_random_sequence(data.num_rows)
    return data, idx_list

data_vendor_map = {
    'hf': load_from_hf,
    'ms': load_from_ms
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


def DataLoader(ds_name, ds_path, max_len, tokenizer, data_vendor='hf', overlap_factor=0, batch_size=5):
    data, idx_list = data_vendor_map[data_vendor](ds_path)
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


def DataTensorLoader(ds_name, ds_path, max_len, tokenizer, data_vendor='hf', overlap_factor=0, batch_size=5):
    data, idx_list = data_vendor_map[data_vendor](ds_path)
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

                    base_ids = tokenizer.batch_encode_plus(
                        out,
                        max_length=max_len + 1,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).input_ids
                    input_ids = base_ids[..., :-1]
                    target_ids = base_ids[..., 1:]
                    yield input_ids, target_ids

    return _iter