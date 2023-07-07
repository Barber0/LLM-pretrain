from time import time


def train_epoch(
    ep,
    model_eng,
    start_batch,
    batch_period,
    md_path,
    md_tag,
    mount_dir,
    START_SIGN,
    END_SIGN,
    tkn,
    data_loader,
    writer,
):
    stime = time()
    total_loss = 0
    period_loss = 0
    
    for i, batch in enumerate(data_loader()):
        if i < start_batch:
            continue

        x_encoded = tkn.batch_encode_plus(
            [f'{START_SIGN} {line}' for line in batch],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids, x_attn_mask = x_encoded['input_ids'], x_encoded['attention_mask']

        target_ids = tkn.batch_encode_plus(
            [f'{line} {END_SIGN}' for line in batch],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )['input_ids']

        input_ids = input_ids.cuda()
        target_ids = target_ids.cuda()

        loss = model_eng(input_ids, target_ids)
        model_eng.backward(loss)
        model_eng.step()

        try:
            writer.add_scalar('Train Loss', loss, i)
        except Exception as e:
            print(f'\r{e}')

        total_loss += loss
        period_loss += loss

        if i > 0 and i % batch_period == 0:
            time_period = time() - stime
            avg_ntokens = x_attn_mask.sum() / x_attn_mask.size(0)
            pad_token_len = x_attn_mask.size(-1)
            print(f'\rep: {ep} batch: {i}, time: {time_period}, ntokens: {avg_ntokens}/{pad_token_len},'
                  f' loss: {period_loss / batch_period}')
            writer.flush()

            try:
                if i % 200 == 0:
                    save_chkpt(str(i), model_eng, mount_dir, md_path, md_tag)
            except Exception as e:
                print(e)

            period_loss = 0
            stime = time()

    save_chkpt(f'epoch-{ep}', model_eng, mount_dir, md_path, md_tag)


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
