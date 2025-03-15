import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer
from dataset import BilingualDataset , casual_mask
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from model import build_transformer

def get_all_sentences(ds , lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_train_tokenizer(config , ds , lang):
    try:
        tokenizer = Tokenizer.from_file(config["tokenizer_file"].format(lang))
    except:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]",])
        tokenizer.train(get_all_sentences(ds,lang), trainer)
        tokenizer.save("tokenizer.json")
    return tokenizer


def get_ds(config):
    ds_row = load_dataset("majedk01/english-arabic-text",split=["train"])
    
    tokenizer_src= get_or_train_tokenizer(config,ds_row,config["lang_src"])
    tokenizer_tgt= get_or_train_tokenizer(config,ds_row,config["lang_tgt"])
    
    # keep 90% of the data for training and 10% for validation
    ds_row = ds_row.train_test_split(test_size=0.1)
    ds_train = ds_row["train"]
    ds_val = ds_row["test"]
    
    train_ds = BilingualDataset(ds_train,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    val_ds = BilingualDataset(ds_val,tokenizer_src,tokenizer_tgt,config["lang_src"],config["lang_tgt"],config["seq_len"])
    
    train_ds = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_ds = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    
    return train_ds,val_ds,tokenizer_src,tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

    
    