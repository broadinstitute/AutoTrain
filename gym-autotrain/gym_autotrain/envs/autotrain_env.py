import gym
from gym import error, spaces, utils
from gym.utils import seeding

import gym_autotrain.envs.utils as utils

from gym_autotrain.envs.thresholdout import Thresholdout

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as  torchdata

import pandas as pd
import numpy as np

from pathlib import Path
from functools import partial

import logging

logger = logging.getLogger(__file__)


def make_o(loss_vec:np.array, lr:float, phi_val:float):
      return np.concat((loss_vec, [lr, phi_val]), dim=0)


class AutoTrainEnvironment(gym.Env):
      metadata = {'render.modes': ['human']}

      def __init__(self):
            pass

      def __repr__(self):
            return f"""AutoTrainEnvironment with the following parameters:
                        lr_init={self.lr_init}, inter_reward={self._inter_reward}, H={self.H}, K={self.K}, T={self.T}"""

      def init(self, backbone: nn.Module,  phi: callable, savedir:Path,
             trnds:torchdata.Dataset, valds:torchdata.Dataset, 
             T=3, H=5, K=256, lr_init=3e-4, inter_reward=0.05,
             num_workers=4, bs=16, log_file=None):
            """
            params:
                  - backbone: nn.Module, neural network architecture for training
                  - trainds, valds: Dataset, train and validation datasets
                  - phi: callable, function to be optimised  

                  - T: num epochs that constitutes one time step
                  - H: length of rewind vector
                  - K: length of the training loss vector in the observation
            """
            # experiment paramter setup
            self.T, self.H, self.K = T, H, K
            self._inter_reward = inter_reward
            # rewind actions * lr_scale actions [decrease  10%, keep, increase 10%] + reinit + stop 
            self.action_space_dim = self.H*3 + 2 
            # loss_vec of size K + lr + phi_val
            self.observation_space_dim = self.K + 2
            # model 
            self.ll = StateLinkedList(savedir=savedir, dim=self.observation_space_dim)
            self.backbone = backbone


            self.criterion = nn.CrossEntropyLoss()
            self.lr_init = self._curr_lr = lr_init

            self._init_backbone()

            # package data
            self.trnds, self.valds = trnds, valds

            self.trndl, self.fixdl, self.valdl =  utils.create_dls(self.trnds, self.valds, bs=bs, num_workers=num_workers)

            self.time_step = 0

            # Thresholdout & statistic of interest
            self.thresholdout = Thresholdout(self.trnds,self.valds)
            self.phi = phi 
            self._phi_func = partial(self.phi, model=self.backbone) #  does partial work on this

            self._cur_phi_val = self.thresholdout.verify(self._phi_func)
            self._last_phi_update = self.time_step

            # calculate the sampling interval
            self._sampling_interval = len(self.trndl) * self.T // self.K

            self.log = pd.DataFrame(columns=['t', 'reward', 'is_stop', 'action_id'])

            self.ll.append(self._curr_observation) 

            logger.info(f"env initialised: {str(self)}")

      def _init_backbone(self):
            utils.init_params(self.backbone)
            self.opt = optim.Adam(self.backbone.parameters(), lr=self.lr_init)

      def _get_phi_val(self) -> float:
            if self._last_phi_update != self.time_step:
                  self._prev_phi_val = self._cur_phi_val
                  self._cur_phi_val = self.thresholdout.verify(self._phi_func)
                  self._last_phi_update = self.time_step

            return self._cur_phi_val

      def _add_observation(self,  loss_vec: np.array,  phi_val: float):
            o_state = ObservationAndState(
                  param_dict=self.backbone.state_dict(),
                  o=make_o(loss_vec, self._curr_lr, phi_val)
            )
            self.ll.append(o_state)

      def visualise_data(self):
            """
            two plots: 
                  - class distribution of training data
                  - class distribution of validation data
            """
            pass 

      def step(self, action_vec): 
            """
            a step in an environment consitiutes of:
                   - check for stop; if stop  then calculate final reward else
                   - scale learning rate
                   - rewind/keep weights 
                   - do training step
                   - calculate intermediate reward
            
            """

            action = torch.argmax(action_vec, dim=-1).item()

            is_stop = action == self.action_space_dim
            is_reinit = action == self.action_space_dim - 1

            if is_stop:
                  final_reward = self._compute_final_reward()
                  return None, final_reward, True, {}

            if is_reinit:
                  self._init_backbone()
                  self.ll = StateLinkedList(savedir=self.savedir)

            # lr and rewind steps
            if action < 5:
                  self._scale_lr(0.9)
                  rewind_steps = action
            elif action >= 5 and action < 10:
                  rewind_step = action - 5
            else:
                  self._scale_lr(1.1)
                  rewind_step = action - 10
            
            # rewind
            if rewind_step != 0 and not is_reinit:
                  self.ll.rewind(rewind_steps)
            
            # do training 
            loss_vec = self._train_one_cycle()
            
            # set current observation
            self._add_observation(loss_vec, self._get_phi_val) # whats this for then

            # get last H observations
            o_history = self.ll.get_observations(self.H)

            # compute intermediate reward
            step_reward = self._compute_intermediate_reward()

            self.time_step += 1

            return o_history, step_reward, False, {}


      def _scale_lr(self, scale_factor):

            for g in self.opt.param_groups:
                  g['lr'] *= scale_factor

            self._curr_lr *= scale_factor


      def _train_one_cycle(self):
            """ train for T epochs, record loss """
            pass

      def _compute_final_reward(self):
            return self._get_phi_val()

      def _compute_intermediate_reward(self):
            delta = self._get_phi_val() - self._prev_phi_val
            if  delta > 0:
                  return self._inter_reward
            else:
                  return -self._inter_reward


      def reset(self):
            # TODO
            # T=3, H=5, K=256, lr_init=3e-4, inter_reward=0.05,
            # num_workers=4, bs=16, log_file=None
            self.init(self.backbone, self.phi, self.savedir, self.trnds, self.valds)
            logger.info('environment re-initialized')
            return self.ll.get_observations(self.H)

      def render(self, mode='human', close=False):
            """render observation """
            pass



class ObservationAndState:
      def __init__(self, param_dict: dict, o: np.array):
            self.param_dict = param_dict
            self.o = o

            self.dim = self.o.size

      def __dict__(self): # param & observation?
            return {
                  'param_dict': self.param_dict,
                  'o': self.o
            }

      def __repr__(self):
            return str(dict(self))

      
            

class StateLinkedList:

      def __init__(self, savedir: Path, dim: int):

            if type(savedir) == str:
                  savedir = Path(savedir) 

            assert savedir.exists() and savedir.is_dir(), "please make sure save path exists and is directory"
            assert dim > 0

            self.savedir = savedir
            self.dim = dim # observation dimension 

            self.len = 0 # the id of the next node

      def get_observations(self, size):
            os = []
            if size > self.len:
                  for _ in range(size - self.len):
                        os.append(np.zeros(self.dim))
                  size = self.len

            os += [self[i]['o'] for i in range(size)]
            return np.vstack(os)
            

      def append(self, state: ObservationState):

            new_node_path = self.node_path(self.len)
            torch.save(dict(state), new_node_path)
            self.len += 1

      def node_path(self, id) -> Path:
            return self.savedir / f'state_{self.len}.ckpt'


      def __len__(self):
            return self.len

      def __getitem__(self, idx):

            if idx > self.len:
                  raise ValueError('idx too large')

            return torch.load(self.node_path(idx))
            

      def rewind(self, steps: int):

            steps = min(steps, self.len)  # remove all its on the caller to check

            for i in range(steps):
                  nodepath = self.node_path(self.len - i)
                  nodepath.unlink()

            self.len -= steps



