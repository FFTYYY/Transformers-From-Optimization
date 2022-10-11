import torch as tc
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import set_random_seed
import pdb
import sklearn
import sklearn.decomposition
import scipy.spatial as spt
import itertools

set_random_seed(23333)

def norm(x):
    return (x ** 2).sum(-1) ** 0.5

def paint(xf,xh,C):
    fig = plt.figure(figsize = (6,6) , dpi=512)

    X = tc.arange(-3.5,3.5 , 0.02)
    Y = tc.arange(-3.5,3.5 , 0.02)
    xys = tc.Tensor( list(itertools.product(X.numpy(),Y.numpy())) )

    idx = norm( xys - xf ) / norm( xys - xh ) <= C
    xys = xys[idx]

    plt.scatter(xys[:,0] , xys[:,1] , s = 4)
    plt.scatter(xf[0,0] , xf[0,1] , s = 122 , color = (0.4,0,0) , marker = "^")
    plt.scatter(xh[0,0] , xh[0,1] , s = 122 , color = (0,0.4,0) , marker = "^")

    plt.xlim(-3.5,3.5)
    plt.ylim(-3.5,3.5)

    # plt.xlabel("",fontsize = 15)
    # plt.ylabel("",fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)


    plt.text( xf[0,0]-0.3 , xf[0,1]-0.5 , s = "$\mathbf{y}_f^*$" , fontsize = 18)
    plt.text( xh[0,0]-0.3 , xh[0,1]-0.5 , s = "$\mathbf{y}_h^*$" , fontsize = 18)

    fig.tight_layout()

    plt.savefig("generated_figures/apollo_C={0}.png".format(C))
    # plt.show()

xf = tc.randn(2).view(1,-1)
xh = tc.randn(2).view(1,-1)
paint(xf,xh,0.7)
paint(xf,xh,1.5)
print ("xf = {0}, xh = {1}".format(xf,xh))