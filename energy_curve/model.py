import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pdb

alpha_1 = 1
alpha_2 = 1

def norm2(X):
    return (X ** 2).sum()

def inner(x,y):
    return (x.view(-1) * y.view(-1)).sum()

idxs_cache = {}

class Attention(nn.Module):
    def __init__(self , d):
        super().__init__()

        self.d = d

        self._W = Parameter( tc.zeros(d,d) )
        self.reset_params()

    def reset_params(self):
        with tc.no_grad():
            nn.init.xavier_normal_(self._W.data)

    def normalize_weight(self):
        pass

    @property
    def W(self):
        return self._W.t() @ self._W

    def get_energy(self , Y):
        Y = Y @ self._W

        n = Y.size(0)
        rho = lambda x: - tc.exp( - x )


        with tc.no_grad():
            ener = 0

            # 这他妈太慢了
            # for i in range(n):
            #     for j in range(i):
            #         ener = ener + rho( 0.5 * norm2(Y[i] - Y[j]) ) 

            if idxs_cache.get(n) is None:
                idxs_cache[n] = {
                    "idxs_i": tc.LongTensor( [ i for i in range(n) for j in range(i)] ) , 
                    "idxs_j": tc.LongTensor( [ j for i in range(n) for j in range(i)] ) , 
                }

            idxs_i = idxs_cache[n]["idxs_i"]
            idxs_j = idxs_cache[n]["idxs_j"]

            ener_rho = rho( 0.5 * ((Y[idxs_i] - Y[idxs_j])**2).sum(-1) ).sum()

            ener = ener_rho + 0.5 * norm2(Y)

        return ener


    def forward(self , Y):
        '''
            Y: (n,d)
        '''

        n , d = Y.size(0) , Y.size(1)

        beta = -0.5 * ((Y @ self._W) ** 2).sum(-1)

        A = tc.softmax( Y @ self.W @ Y.t()  , -1 )
        Z = (1-alpha_1) * Y + alpha_1 * A @ Y

        return Z

            
class FFN(nn.Module):
    def __init__(self , d):
        super().__init__()

        self.d = d

        self._Wf = Parameter( tc.zeros(d,d) )
        self.B  = Parameter( tc.zeros(1,d) )

        self.reset_params()

    def reset_params(self):
        with tc.no_grad():
            nn.init.xavier_normal_(self._Wf.data)
            nn.init.xavier_normal_(self.B.data)

    def normalize_weight(self):
        with tc.no_grad():
            W = self._Wf.data
            W = W @ W.t()

            L , U = tc.linalg.eigh(W) # W = U @ L.diag() @ U.t()
            L[L >  0.95] = 0.95
            L[L < -0.95] = -0.95
            W = U @ L.diag() @ U.t()

            self._Wf.data = W
 
    @property
    def Wf(self):
        return - 0.5 * alpha_2 * (self._Wf + self._Wf.t()) + (1-alpha_2) * tc.eye(self.d, device = self._Wf.device)


    def get_energy(self , Y):
        with tc.no_grad():
            return 0.5 * tc.trace(Y @ self._Wf @ Y.t()) + 0.5 * norm2(Y - self.B)


    def forward(self , Y):
        '''
            Y: (n,d)
        '''
        Y = Y @ self.Wf + self.B
        return Y

class TransformerLayer(nn.Module):
    def __init__(self , d , relu):
        super().__init__()
        self.d = d
        self.relu = relu

        self.attn = Attention(self.d)
        self.ffn  = FFN(self.d)

    def get_energy(self , Y):
        ener = 0
        ener = ener + self.attn.get_energy(Y)
        ener = ener + self.ffn.get_energy(Y)
        return ener

    def normalize_weight(self):
        self.attn.normalize_weight()
        self.ffn.normalize_weight()

    def forward(self , Y):
        Y = self.attn(Y)
        Y = self.ffn(Y)
        if self.relu:
            Y = F.relu(Y)
        return Y

class Transformer(nn.Module):
    def __init__(self , d , num_layers , output_size = 2 , relu = False):
        super().__init__()

        self.d = d
        self.num_layers = num_layers
        self.relu = relu

        self.rec_layer = TransformerLayer(self.d , self.relu)

        self.output = nn.Linear(d , output_size)

    def normalize_weight(self):
        self.rec_layer.normalize_weight()

    def get_energy(self , Y ):
        return self.rec_layer.get_energy(Y)

    def forward(self , Y, no_energy = False, no_X = False):

        energies = [] if no_energy else [ self.get_energy(Y) ]
        for layer_idx in range(self.num_layers):
            Y = self.rec_layer(Y)
            if not no_energy:
                energies.append( self.get_energy(Y) )

        output = self.output(Y)

        return {
            "repr": Y , 
            "ener": energies , 
            "pred": output , 
        }
        
        
        