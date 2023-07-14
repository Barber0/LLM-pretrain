from time import time


def train_epoch(
    ep,
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
):
    stime = time()
    total_loss = 0
    period_loss = 0
    
    # ext_max_len = max_len + 1
    for i, batch in enumerate(data_loader()):
        if i < start_batch:
            continue

        x_encoded = tkn.batch_encode_plus(
            batch,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        base_ids, x_attn_mask = x_encoded['input_ids'], x_encoded['attention_mask']

        input_ids = base_ids[..., :-1].cuda()
        target_ids = base_ids[..., 1:].cuda()

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
            print('\rep: {} batch: {}, time: {:.2f}, ntokens: {:.2f}/{}, loss: {}'.format(
                ep, i, time_period, avg_ntokens, pad_token_len, period_loss/batch_period
            ))
            writer.flush()

            try:
                if i % 1200 == 0:
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
