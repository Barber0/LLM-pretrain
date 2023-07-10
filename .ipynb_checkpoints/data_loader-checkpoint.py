from datasets import load_from_disk


def SelfInstructLoader(ds_path, batch_size=5):
    data = load_from_disk(ds_path)

    def _iter():
        batch = []
        for i, v in enumerate(data['train']):
            batch.append(f'''{v['prompt']} {v['completion']} EndEndEnd''')
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
