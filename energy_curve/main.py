from model import Transformer
import torch as tc
import pdb
import torch.backends.cudnn
from load_data import load_data
from paint import paint
from tqdm import tqdm

import numpy as np
import torch as tc
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tc.manual_seed(seed)
    tc.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(2333)

dataset_names = ["imdb" , "sst2"]

for dataset_name in dataset_names:
    norelu_name = ""
    relu_name = ""
    if dataset_name == "imdb":
        norelu_name = "norelu"
        relu_name = "relu"
    else:
        norelu_name = "norelu_%s" % dataset_name
        relu_name = "relu_%s" % dataset_name


    data_bundle , word2vec_embed = load_data(dataset_name)

    train_data = data_bundle.get_dataset("train")
    d = word2vec_embed.embedding_dim  
    num_layers = 12

    model_1 = Transformer(d , num_layers , 2)
    model_1.normalize_weight()
    paint(model_1 , "../generated_figures/%s.png" % norelu_name)

    model_2 = Transformer(d , num_layers , 2)
    model_2.normalize_weight()
    paint(model_2 , "../generated_figures/%s.png" % relu_name)
