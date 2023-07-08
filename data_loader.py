from datasets import load_from_disk
from torch.utils.data import Dataset
import math


def SelfInstructLoader(ds_path, batch_size=5):
    data = load_from_disk(ds_path)

    def _iter():
        batch = []
        for i, v in enumerate(data['train']):
            batch.append(f'''{v['prompt']} {v['completion']}''')
            if i > 0 and i % batch_size == 0:
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
            if i > 0 and i % batch_size == 0:
                out = batch
                batch = []
                yield out
    return _iter


class BcdsDataset(Dataset):
    def __init__(self, ds_path, tkn, max_len, batch_size):
        self.data = load_from_disk(ds_path)['train']
        self.tkn = tkn
        self.max_len = max_len
        self.batch_size = batch_size
        self.length = math.ceil(self.data.num_rows / self.batch_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # start_idx = idx*self.batch_size
        # lines = self.data[start_idx:start_idx+self.batch_size]['text']

        input_ids = self.tkn.batch_encode_plus(
            self.data[idx]['text'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )['input_ids']
        return input_ids[0, :-1], input_ids[0, 1:]


if __name__ == '__main__':
    from mytkn import get_tkn
    tkn, VOCAB_SIZE, START_SIGN, END_SIGN, START_ID, END_ID = get_tkn()
    ds = BcdsDataset(
        '/root/autodl-tmp/content/drive/MyDrive/webtext-datasets/arch/',
        tkn,
        512
    )
