import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_autotrain.envs import ReplayBuffer
from gym_autotrain.envs.utils import init_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AutoTrainEnvironment(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
        pass

  def init(self, model: nn.Module , all_data, num_cls=-1, num_examples=-1, tfms=None, lr_init=3e-4, opt_func='Adam', data_split=0.6):

      self.replbuff = ReplayBuffer()

      self.log = pd.DataFrame(columns=[])

      # model init first
      self.model = model
      init_model(self.arch)
      self.criterion = nn.CrossEntropyLoss()
      self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
      self.lr_init = self.lr_curr = lr_init

      # create databunch
      self.create_dss(all_data)

      # create learner object     
      def create_dss(self, data):

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