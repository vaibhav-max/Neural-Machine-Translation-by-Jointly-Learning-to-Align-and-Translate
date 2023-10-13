import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_length = seq_len
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        
    def __len__(self) :
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        encoder_input_token = self.tokenizer_src.encode(src_text).ids
        decoder_input_token = self.tokenizer_tgt.encode(tgt_text).ids
        
        encode_num_padding_zeros = self.seq_length - len(encoder_input_token) - 2
        decode_num_padding_zeros = self.seq_length - len(decoder_input_token) - 1
        
        if encode_num_padding_zeros < 0 or decode_num_padding_zeros < 0:
            raise ValueError('Sentence is too long')
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_token, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encode_num_padding_zeros, dtype=torch.int64)
            ]
        )
        
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_token, dtype=torch.int64),
                torch.tensor([self.pad_token] * decode_num_padding_zeros, dtype=torch.int64)
            ]
        )
        
        label = torch.cat(
            [
                torch.tensor(decoder_input_token, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decode_num_padding_zeros)
            ]
        )
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }