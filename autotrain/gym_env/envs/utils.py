import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import numpy as np

from functools import partial

INIT_METHODS = [nn.init.xavier_uniform_, nn.init.xavier_normal_, \
                nn.init.kaiming_uniform_, nn.init.kaiming_normal_]

def init_weights(m, init=nn.init.xavier_uniform):
    if type(m) == nn.Linear:
        init(m.weight)
        m.bias.data.fill_(0.01)
        
def init_params(model): # TODO
    func = np.random.choice(INIT_METHODS)
    model.apply(partial(init_weights, init=func))

def create_dls(trnds, valds, bs=16, num_workers=4):
    trndl = torchdata.DataLoader(trnds, batch_size=bs, shuffle=True, num_workers=num_workers)

    fixdl = torchdata.DataLoader(trnds, batch_size=bs, shuffle=False, num_workers=num_workers)

    valdl = torchdata.DataLoader(valds, batch_size=bs, shuffle=False, num_workers=num_workers)

    return trndl, fixdl, valdl


def create_dss(data, data_split, dist):
    """create datasets"""
    X, Y = data

    trn, val = splitds(X, Y, data_split)

    # pack it into datasets
    trnds = SimpleDataset(*trn)
    valds = SimpleDataset(*val)

    return trnds, valds

def splitds(X, Y, data_split):
    splitidx = int(len(X) * data_split)

    Xtrn, Xval = X[:splitidx], X[splitidx:]
    Ytrn, Yval = Y[:splitidx], Y[splitidx:]
    return (Xtrn, Ytrn), (Xval, Yval)


class SimpleDataset(torchdata.Dataset):

    def __init__(self, data, targets, tfms=None):
        super().__init__()
        self.data = data
        self.targets = targets
        self.tfms = tfms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = self.data[i], self.targets[i]

        if self.tfms:
            x = self.tfms(x)
        
        return x, y