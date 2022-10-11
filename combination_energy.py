import torch as tc
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import set_random_seed
import pdb
import sklearn
import sklearn.decomposition
from matplotlib.patches import ConnectionPatch

set_random_seed(233)

def F_norm(Y):
    return (Y ** 2).sum()

n = 500
d = 128
W1 = tc.randn(d,d) * 0.1
W2 = tc.randn(n,n) * 0.1
num_epochs = 200

YB1 = tc.rand(n,d)
YB2 = tc.rand(n,d)

class Model(nn.Module):
    def __init__(self , Y0):
        super().__init__()
        self.Y = nn.Parameter( Y0 )

Y0 = tc.rand(n,d) * 2
model = Model( Y0 )
model_h = Model( Y0.detach().clone() )
optimizer_1 = tc.optim.SGD(model.parameters() , lr = 0.05)
optimizer_2 = tc.optim.SGD(model.parameters() , lr = 0.05)
optimizer_h = tc.optim.SGD(model_h.parameters() , lr = 0.01) # 用来寻找h的最优点


def ener_1(Y):
    return F_norm(Y @ W1) + F_norm( Y - YB1 )
def ener_2(Y):
    return F_norm(W2 @ Y) + F_norm( Y - YB2 )

mapW = tc.randn(2 , n*d)
def mapsto(Y):
    Y = Y.view(-1,1)
    Y = mapW @ Y
    return Y


tot_losses = []
Y_trace = []
for epoch_id in tqdm( range(num_epochs) ):
    
    Y = model.Y
    
    if epoch_id % 2 == 0:
        loss = ener_1(Y)
        optimizer = optimizer_1
    elif epoch_id % 2 == 1:
        loss = ener_2(Y)
        optimizer = optimizer_2
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with tc.no_grad():
        tot_loss = ener_1(Y) + ener_2(Y)
        
    tot_losses.append( tot_loss )
    Y_trace.append( Y.data.clone().view(1,-1) )


# 求h的最优解
_pbar = tqdm(range(600))
for _ in _pbar:
    loss_h = ener_1(model_h.Y) + ener_2(model_h.Y)
    optimizer_h.zero_grad()
    loss_h.backward()
    optimizer_h.step()
    _pbar.set_description("loss = %.4f" % loss_h)


Y_trace = tc.cat( Y_trace , dim = 0 ) # (num_epoch , n*d)
pca = sklearn.decomposition.PCA(2)
mapedYs = pca.fit_transform(Y_trace)
mapedh = tc.Tensor( pca.transform(model_h.Y.detach().view(1,-1)) ).view(-1)

tot_losses = tc.Tensor(tot_losses)
tot_losses = tc.log(tot_losses)
tot_losses = tot_losses - tot_losses.min()

# ---- 画轨迹 ----

# 小方框
xl , xr = mapedh[0]-4,mapedh[0]+4
yb , yt = mapedh[1]-4,mapedh[1]+4

fig = plt.figure(figsize=(12,5) , dpi=512)
p1 = plt.subplot(121)
p2 = plt.subplot(122)

p1.plot( mapedYs[:,0] , mapedYs[:,1] , zorder = 1 , label = "trace of $Y^{(t)}$")
p1.scatter( mapedh[0] , mapedh[1] , color = (1,0.4,0.1) , s = 40 , zorder = 2 , marker = "^" , label = "$Y_h^*$")
p1.scatter( mapedYs[0][0] , mapedYs[0][1] , color = (0.7,0.4,0.4) , s = 40 , zorder = 3 , marker = "*" , label = "$Y^{(0)}$")
p1.plot( [xl,xr,xr,xl,xl] , [yt,yt,yb,yb,yt] , color = (0.2,0.0,0.2,0.7))
p1.legend()
# p1.set_xlabel("x[0]")
#   p1.set_ylabel("x[1]")

p2.plot( mapedYs[:,0] , mapedYs[:,1] , zorder = 1 , label = "trace of $Y^{(t)}$")
p2.scatter( mapedh[0] , mapedh[1] , color = (1,0.4,0.1) , s = 40 , zorder = 2 , marker = "^" , label = "$Y_h^*$")
p2.set_xlim(xl , xr)
p2.set_ylim(yb , yt)
p2.legend()
# p2.set_xlabel("x[0]")
# p2.set_ylabel("x[1]")

# 连接p1和p2
con1 = ConnectionPatch(
    xyA = [xr,yt] , xyB = [xl,yt] , 
    coordsA = "data" , coordsB = "data" , 
    axesA = p1 , axesB = p2 , 
    color = (0.2,0.0,0.2,0.7) , linestyle = "dashed"
)
con2 = ConnectionPatch(
    xyA = [xr,yb] , xyB = [xl,yb] , 
    coordsA = "data" , coordsB = "data" , 
    axesA = p1 , axesB = p2 , 
    color = (0.2,0.0,0.2,0.7) , linestyle = "dashed"
)
p1.add_artist(con1)
p1.add_artist(con2)

fig.tight_layout()

plt.savefig("generated_figures/alternate_trace.png")


# ---- 画能量函数图 ----
fig = plt.figure(figsize=(6,4) , dpi=512)
plt.plot( range(num_epochs) , tot_losses )
plt.xlabel("$t$" , fontsize = 15)

plt.ylabel("$\log E\\left(Y^{(t)}\\right) - \log E_{\\min}$" , fontsize = 15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

fig.tight_layout()

plt.savefig("generated_figures/alternate_energy.png")
