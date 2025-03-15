from pathlib import Path

def get_config():
    return {
        "lang_src": "en",
        "lang_tgt": "ar",
        "seq_len": 350,
        "batch_size": 8,
        "d_model": 512,
        "tokenizer_file": "tokenizer_{0}.json",
        "model_folder":"weights",
        "model_basename": "tmodel_",
        "epochs": 20,
        "lr": 0.0001,
        "preload":None,
        "experiment_name":"runs/tmodel",
    }
    

def get_weights_file_path(config,epoch):
    return Path(config["model_folder"]) / (config["model_basename"] + str(epoch) + ".pt")