import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_autotrain.envs import ReplayBuffer

import torch.utils


class AutoTrainEnvironment(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
        pass

  def init(self, arch, dataset, opt_func='Adam', data_split=0.6):

        self.arch = arch
        self.dataset = dataset
        
        
  def step(self, action): 
        lr_scale, go_back = action

        pass

  def reset(self): 
        pass

  def render(self, mode='human', close=False): 
        pass


class Learner:
      """maybe a good idea to use fastai.Learner for this?"""
      def __init__(self):
            pass