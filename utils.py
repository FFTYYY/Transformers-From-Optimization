import random
import torch as tc
import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tc.manual_seed(seed)
    tc.cuda.manual_seed_all(seed)
    tc.backends.cudnn.deterministic = True
    tc.backends.cudnn.benchmark = False
