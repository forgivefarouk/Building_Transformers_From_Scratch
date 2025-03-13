import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


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
    ds_row = load_dataset("majedk01/english-arabic-text", f'{config['lang_src']}-{config['lang_tgt']}',splits=["train"])
    
    tokenizer_src= get_or_train_tokenizer(config,ds_row,config["lang_src"])
    tokenizer_tgt= get_or_train_tokenizer(config,ds_row,config["lang_tgt"])
    
    # keep 90% of the data for training and 10% for validation
    ds_row = ds_row.train_test_split(test_size=0.1)
    ds_train = ds_row["train"]
    ds_val = ds_row["test"]
    return ds_train, ds_val, tokenizer_src, tokenizer_tgt
    
    