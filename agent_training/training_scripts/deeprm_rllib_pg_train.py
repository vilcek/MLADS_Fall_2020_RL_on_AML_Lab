"""
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.
"""

import ray
import ray.rllib.agents.pg as pg
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.tune.registry import register_env

import gym
from gym import spaces
from environment.environment import Parameters, Env

import torch
import torch.nn as nn

import numpy as np
import time
import argparse

from azureml.core import Run

# define the custom class that exposes the RL environment through a Gym interface
# details can be found in the RLLib documentation here: https://docs.ray.io/en/master/rllib-env.html#configuring-environments

class CustomEnv(gym.Env):
    def __init__(self, env_config):
        simu_len = env_config['simu_len']
        num_ex = env_config['num_ex']
        
        pa = Parameters()
        pa.simu_len = simu_len
        pa.num_ex = num_ex
        pa.compute_dependent_parameters()
        
        self.env = Env(pa, render=False, repre='image')
        self.action_space = spaces.Discrete(n=pa.num_nw + 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.env.observe().shape, dtype=np.float)
    
    def reset(self):
        self.env.reset()
        obs = self.env.observe()
        return obs
    
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        info = {}
        return next_obs, reward, done, info

# define the custom class that implements the custom model for the agent policy
# here we are using PyTorch
# details can be found in the RLLib documentation here: https://docs.ray.io/en/master/rllib-models.html#pytorch-models
    
class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.hidden_layers = nn.Sequential(nn.Linear(20*124, 32), nn.ReLU(),
                                           nn.Linear(32, 16), nn.ReLU())
        
        self.logits = nn.Sequential(nn.Linear(16, 6))
        
        self.value_branch = nn.Sequential(nn.Linear(16, 1))

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs'].float()
        obs = obs.view(obs.shape[0], 1, obs.shape[1], obs.shape[2])
        obs = obs.view(obs.shape[0], obs.shape[1] * obs.shape[2] * obs.shape[3])
        self.features = self.hidden_layers(obs)
        actions = self.logits(self.features)
        
        return actions, state
    
    @override(TorchModelV2)
    def value_function(self):
        return self.value_branch(self.features).squeeze(1)

# define the RL environment constructor and register it for use in RLLib

def env_creator(env_config):
    return CustomEnv(env_config)

register_env('CustomEnv', env_creator)

# register the custom policy model for use in RLLib
    
ModelCatalog.register_custom_model('CustomModel', CustomModel)

# parse all input arguments to this training script

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, dest='gamma', default=0.99)
parser.add_argument('--num_gpus', type=int, dest='num_gpus', default=1)
parser.add_argument('--num_workers', type=int, dest='num_workers', default=12)
parser.add_argument('--num_envs_per_worker', type=int, dest='num_envs_per_worker', default=5)
parser.add_argument('--num_cpus_per_worker', type=int, dest='num_cpus_per_worker', default=1)
parser.add_argument('--use_pytorch', type=int, dest='use_pytorch', default=1)
parser.add_argument('--timesteps_per_iteration', type=int, dest='timesteps_per_iteration', default=100000)
parser.add_argument('--rollout_fragment_length', type=int, dest='rollout_fragment_length', default=50)
parser.add_argument('--train_batch_size', type=int, dest='train_batch_size', default=500)
parser.add_argument('--lr', type=float, dest='lr', default=0.00025)
parser.add_argument('--num_iterations', type=int, dest='num_iterations', default=500)
parser.add_argument('--simu_len', type=int, dest='simu_len', default=50)
parser.add_argument('--num_ex', type=int, dest='num_ex', default=1)
parser.add_argument('--default_ray_address', type=str, dest='default_ray_address', default='localhost:6379')

args = parser.parse_args()

# create a copy of the default Policy Gradient configuration in RLLib

config = pg.DEFAULT_CONFIG
my_config = config.copy()

# set the relevent parameters for this training

my_params = {
    'gamma': args.gamma,
    'num_gpus': args.num_gpus,
    'num_workers': args.num_workers,
    'num_envs_per_worker': args.num_envs_per_worker,
    'num_cpus_per_worker': args.num_cpus_per_worker,
    'use_pytorch': bool(args.use_pytorch),
    'timesteps_per_iteration': args.timesteps_per_iteration,
    'rollout_fragment_length': args.rollout_fragment_length,
    'train_batch_size': args.train_batch_size,
    'lr': args.lr,
    'model': {'custom_model': 'CustomModel'},
    'env': 'CustomEnv',
    'env_config': {'simu_len': args.simu_len, 'num_ex': args.num_ex}
}

for key, value in my_params.items():
    my_config[key] = value

# initialize the Ray backend

ray.init(address=args.default_ray_address)

# create the RLLib trainer object

trainer = pg.PGTrainer(config=my_config)

# get a reference to Azure ML Run object, to be used to log training metrics

run = Run.get_context()

# execute the RLLib training loop

for i in range(args.num_iterations):
    start_time = time.time()
    result = trainer.train()
    end_time = time.time()
    
    print('Iteration: {0} - Mean Score: {1} - Min Score: {2} - Max Score: {3} - Elapsed time: {4} s.'.format(result['training_iteration'], round(result['episode_reward_mean']), round(result['episode_reward_min']), round(result['episode_reward_max']), round(end_time - start_time)))

    run.log('Mean Reward', round(result['episode_reward_mean']))
    run.log('Min Reward', round(result['episode_reward_min']))
    run.log('Max Reward', round(result['episode_reward_max']))
    run.log('Duration', round(end_time - start_time))

# save the model after training

trainer.save(checkpoint_dir='./outputs')