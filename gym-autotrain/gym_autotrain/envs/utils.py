
import torch
import torch.nn

import numpy as np

INIT_METHODS = [nn.init.xavier_uniform_, nn.init.xavier_normal_, \
                nn.init.kaiming_uniform_, nn.init.kaiming_normal_]

def init_weights(m, init=nn.init.xavier_uniform):
    if type(m) == nn.Linear:
        init(m.weight)
        m.bias.data.fill_(0.01)
        
def init_model(model):
    func = np.random.choice(INIT_METHODS)
    model.apply(partial(init_weights, init=func))