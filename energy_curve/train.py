import torch as tc
import torch.nn.functional as F
import torch.backends.cudnn
import numpy as np
import pdb
from load_data import load_data
from paint import paint 
from tqdm import tqdm
import random
from model import Transformer

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tc.manual_seed(seed)
    tc.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def train(model , data_bundle , word2vec_embed):
    # model = model.train()
    train_data = data_bundle.get_dataset("train")
    
    optimizer = tc.optim.SGD( model.parameters() , lr = 1e-2)
    
    pbar = tqdm( random.sample( list(range(len(train_data))) , num_steps) )
    for i , data_idx in enumerate(pbar):
        x = word2vec_embed(train_data["words"][data_idx])
        targ = tc.LongTensor( [train_data["target"][data_idx]] ).view(1)

        x = x.cuda()
        targ = targ.cuda()

        pred = model(x, no_energy = True)["pred"].view(1,-1)

        loss = F.cross_entropy(pred , targ)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # can remove this, but will cause NaN loss in some cases
        model.normalize_weight() 

        pbar.set_postfix_str("loss = %.4f" % (float(loss)))

    return model

set_random_seed(2333)

num_layers = 12
num_steps = 2000

for dataset_name in ["imdb" , "sst2"]:

    norelu_name = "trained_norelu_" + dataset_name
    relu_name = "trained_relu_" + dataset_name

    data_bundle , word2vec_embed = load_data(dataset_name)
    d = word2vec_embed.embedding_dim  

    model_1 = Transformer(d , num_layers , 2 , False).cuda()
    model_1.normalize_weight()

    model_2 = Transformer(d , num_layers , 2 , True ).cuda()
    model_2.normalize_weight()

    model_1 = train(model_1, data_bundle , word2vec_embed)
    model_1.normalize_weight()
    model_2 = train(model_2, data_bundle , word2vec_embed)
    model_2.normalize_weight()

    paint(model_1 , "../generated_figures/%s.png" % norelu_name)
    paint(model_2 , "../generated_figures/%s.png" % relu_name)
