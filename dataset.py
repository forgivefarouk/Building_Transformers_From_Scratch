from torch.utils.data import Dataset
from tokenizers import Tokenizer
import torch

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer,lang_src,lang_tgt,seq_len):
        self.ds = ds
        self.seq_len = seq_len
        self.pad_token = torch.tensor(tokenizer.token_to_id("[PAD]"))
        self.sos_token = torch.tensor(tokenizer.token_to_id("[SOS]"))
        self.eos_token = torch.tensor(tokenizer.token_to_id("[EOS]"))
        self.tokenizer = tokenizer
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,idx):
        item = self.ds[idx]
        
        encoder_input = self.tokenizer.encode(item["translation"][self.lang_src]).ids
        decoder_input = self.tokenizer.encode(item["translation"][self.lang_tgt]).ids
        label = decoder_input[1:]
        
        encoder_pad_len  = len(encoder_input) - self.seq_len - 2
        decoder_pad_len  = len(decoder_input) - self.seq_len - 1
        
        if encoder_pad_len < 0 or decoder_pad_len < 0:
            raise ValueError("Encoder input is too long")
        
        encoder_input= torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input[:self.seq_len]),
                self.eos_token,
                torch.tensor([self.pad_token]*encoder_pad_len)
            ]
        )
        
        decoder_input= torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input[:self.seq_len]),
                torch.tensor([self.pad_token]*decoder_pad_len)
            ]
        )
        
        label= torch.cat(
            [
                torch.tensor(label[:self.seq_len]),
                self.eos_token,
                torch.tensor([self.pad_token]*decoder_pad_len)
            ]
        )
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0))
        }
        
def casual_mask(size):
    mask = torch.triu(torch.ones(1,size,size),diagonal=1)
    return mask == 0
