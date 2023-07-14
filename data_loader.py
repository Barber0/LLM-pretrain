from datasets import load_from_disk

ds_handlers_map = {
    'openwebtext': lambda line: line['text'],
    'self_instruct': lambda line: f'''{line['prompt']} {line['completion']}'''
}


def DataLoader(ds_name, ds_path, batch_size=5):
    data = load_from_disk(ds_path)
    ds_handler = ds_handlers_map[ds_name]

    def _iter():
        batch = []
        for i, v in enumerate(data['train']):
            batch.append(ds_handler(v))
            if (i+1) % batch_size == 0:
                out = batch
                batch = []
                yield out
    return _iter, data['train'].num_rows
