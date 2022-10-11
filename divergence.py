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

W = tc.randn(2,2)
W = W @ W.t()
B = tc.randn(1,2)
def h(x):
    return 0.5 * x.t() @ W @ x + 0.5 * norm(x)**2

def alphah(X):
    alpha = 0.25
    return alpha * X @ W + alpha * X - alpha * B

def div(X):
    '''X: (n,2)'''
    xi1 = alphah(X) # (n,2)
    xi2 = X # (n,2)

    r = xi2**2 - xi1**2
    r [r >= 0] = 0

    return r.sum(-1) / norm(xi1)**2


fig = plt.figure(figsize = (24,6) , dpi = 512 )
p1 = plt.subplot(141)
p2 = plt.subplot(142)
p3 = plt.subplot(143)
p4 = plt.subplot(144)

for thres , pl in zip([0.2 , 0.4 , 0.6, 0.8] , [p1,p2,p3,p4]):
    X = tc.arange(-5,5 , 0.02)
    Y = tc.arange(-5,5 , 0.02)
    xys = tc.Tensor( list(itertools.product(X.numpy(),Y.numpy())) )

    idx = div(xys) >= -thres
    xys = xys[idx]

    pl.scatter(xys[:,0] , xys[:,1] , s = 4)

    pl.set_xlim(-5,5)
    pl.set_ylim(-5,5)

    for tick in pl.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in pl.yaxis.get_major_ticks():
        tick.label.set_fontsize(15) 

    # pl.set_xticks(fontsize=15)
    # pl.set_yticks(fontsize=15)
    pl.set_title("$\kappa={0}$".format(thres) , fontsize = 18)

fig.tight_layout()

# plt.show()


plt.savefig("generated_figures/divergence.png")
