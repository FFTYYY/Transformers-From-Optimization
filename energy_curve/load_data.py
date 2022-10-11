from model import Transformer
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.io import IMDBLoader , SST2Loader
from fastNLP.io import IMDBPipe , SST2Pipe
from fastNLP.embeddings import StaticEmbedding
import pdb
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


def load_data(data_name = "imdb"):

    if data_name == "imdb":
        save_path = Path("./chach_datas.pkl")

        if not save_path.exists():
            loader = IMDBLoader()
            pipe = IMDBPipe()
            data_bundle = pipe.process( loader.load(loader.download()) )

            word_vocab = data_bundle.get_vocab("words")
            word2vec_embed = StaticEmbedding(word_vocab, model_dir_or_name = "en")

            with open(save_path , "wb")  as fil:
                pickle.dump([data_bundle , word2vec_embed] , fil)

        else:
            with open(save_path , "rb")  as fil:
                data_bundle , word2vec_embed = pickle.load(fil)

        return data_bundle , word2vec_embed

    save_path = Path("./chach_datas_sst2.pkl")

    if not save_path.exists():
        loader = SST2Loader()
        pipe = SST2Pipe()
        data_bundle = pipe.process( loader.load(loader.download()) )

        word_vocab = data_bundle.get_vocab("words")
        word2vec_embed = StaticEmbedding(word_vocab, model_dir_or_name = "en")

        with open(save_path , "wb")  as fil:
            pickle.dump([data_bundle , word2vec_embed] , fil)

    else:
        with open(save_path , "rb")  as fil:
            data_bundle , word2vec_embed = pickle.load(fil)

    return data_bundle , word2vec_embed
