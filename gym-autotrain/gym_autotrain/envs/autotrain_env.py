import gym
from gym import error, spaces, utils
from gym.utils import seeding

import gym_autotrain.envs.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import pandas as pd

class AutoTrainEnvironment(gym.Env):
      metadata = {'render.modes': ['human']}

      def __init__(self):
            pass

      def init(self, model: nn.Module , all_data, phi: callable, 
            cls_dist=None, tfms=None, lr_init=3e-4, opt_func='Adam', 
            data_split=0.6, num_workers=4, bs=16):

            self.log = pd.DataFrame(columns=[])

            self.phi = phi # function to be optimised

            # model init first
            self.model = model
            utils.init_model(self.arch)

            self.criterion = nn.CrossEntropyLoss()
            self.lr_init = self.lr_curr = lr_init
            self.opt = optim.Adam(self.model.parameters(), lr=self.lr_init)

            # package data
            self.trnds, self.valds = utils.create_dss(all_data, data_split, cls_dist)

            self.trndl, self.fixdl, self.valdl =  utils.create_dls(self.trnds, self.valds, bs=bs, num_workers=num_workers)


      def train_one_cycle(self):
            pass
      
      def visualise_data(self):
            pass 

      def step(self, action): 
            lr_scale, go_back = action

      def reset(self): 
            pass

      def render(self, mode='human', close=False):
            pass


class Learner:
      """use fastai learner but for later prototypes"""
      def __init__(self):
            pass