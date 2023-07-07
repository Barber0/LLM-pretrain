from transformers import GPT2Tokenizer

def get_tkn(tkn_path):
    START_SIGN = '<start>'
    END_SIGN = '<end>'

    tkn = GPT2Tokenizer.from_pretrained(tkn_path)
    tkn.pad_token = '[PAD]'
    tkn.add_tokens([START_SIGN, END_SIGN])

    START_ID, END_ID = tkn.convert_tokens_to_ids([START_SIGN, END_SIGN])
    VOCAB_SIZE = tkn.vocab_size + 2

    added_tkn_map = {
        START_SIGN: START_ID,
        END_SIGN: END_ID
    }
    return tkn, VOCAB_SIZE, added_tkn_map