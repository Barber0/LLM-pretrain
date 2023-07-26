from torch.utils.data import Dataset
import pickle
import os


class EfficientTextDataset(Dataset):
    def __init__(self, file_path, line_handler=lambda x: x):
        super(EfficientTextDataset, self).__init__()
        self.line_handler = line_handler

        self.file_path = file_path
        offsets_file = file_path + '.offsets.pkl'

        if os.path.exists(offsets_file):
            # Load the precomputed offsets
            with open(offsets_file, 'rb') as f:
                self.offsets = pickle.load(f)
        else:
            # Compute offsets and save to file
            self.offsets = self.get_offsets(file_path)
            with open(offsets_file, 'wb') as f:
                pickle.dump(self.offsets, f)

        self.length = len(self.offsets) - 1

    def get_offsets(self, file_path):
        offsets = [0]  # Start with the first byte
        with open(file_path, 'r') as file:
            for line in file:
                offsets.append(offsets[-1] + len(line.encode('utf-8')))
        return offsets

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with open(self.file_path, 'r') as file:
            file.seek(self.offsets[index])  # Jump directly to the position
            line = file.readline().strip()
        return self.line_handler(line)


def get_batch_collater(tkn, max_len):
    tkn_len = max_len+1

    def batch_collator(batch):
        base_ids = tkn.batch_encode_plus(
            batch,
            max_length=tkn_len,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).input_ids
        input_ids = base_ids[..., :-1]
        target_ids = base_ids[..., 1:]
        return input_ids, target_ids
    return batch_collator
