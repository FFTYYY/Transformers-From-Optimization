from model import Transformer
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.io import IMDBLoader
from fastNLP.io import IMDBPipe
from fastNLP.embeddings import StaticEmbedding
import pdb
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from load_data import load_data
from tqdm import tqdm
import random

num_test_epoch = 200

def paint(model , savepath = None):
    model = model.eval()
    data_bundle , word2vec_embed = load_data()
    datas = data_bundle.get_dataset("test")

    eners_tot = [  ]

    for data_idx in tqdm( random.sample( list(range(len(datas))) , num_test_epoch) ):
        Y = word2vec_embed(datas[data_idx]["words"])

        eners = model(Y)["ener"]
        eners = tc.Tensor(eners).view(-1)

        eners_tot.append(eners)
    
    eners_tot = tc.cat( [e.view(-1 , 1) for e in eners_tot] , dim = -1 )
    eners_tot = eners_tot - eners_tot.min()
    # eners_tot = eners_tot / eners_tot.max()

    fig = plt.figure(figsize=(8,4))
    plt.plot( range(model.num_layers + 1) , eners_tot.mean(dim = -1) )
    pre_bp = plt.boxplot( 
        eners_tot , 
        positions = list(range(model.num_layers + 1)), 
        showfliers = False 
    )
    res = {key : [v.get_data() for v in value] for key, value in pre_bp.items()}
    whiskers = res["whiskers"]
    whisker_min = min( [ whiskers[i][1].min() for i in range(len(whiskers))] )
    # plt.cla()
    plt.close()

    fig = plt.figure(figsize=(8,4))
    eners_tot = eners_tot - float( whisker_min )
    plt.plot( range(model.num_layers + 1) , eners_tot.mean(dim = -1) )
    plt.boxplot( 
        eners_tot , 
        positions = list(range(model.num_layers + 1)), 
        showfliers = False 
    )


    plt.xlabel("$t$ (layer index)" , fontsize = 15)

    plt.ylabel("$E\\left(Y^{(t)}\\right) - E_{\\min}$" , fontsize = 15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    fig.tight_layout()

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)
