import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import gym

from collections import namedtuple

from ..problems.env import MinatarEnv, CartPoleSwingUp, CustomMountainCarEnv
from .impala import *

NETWORKS={}

def register(name):
    def wrapped(func):
        NETWORKS[name]=func
        return func
    return wrapped

def get_genome_size(Net):
    model = Net()
    with torch.no_grad():
        params = model.parameters()
        vec = torch.nn.utils.parameters_to_vector(params)
    return len(vec.cpu().numpy())

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


# Flat net for Atari RAM and gym envs
class FFNet(nn.Module):
    def __init__(self, n_in, h_size, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_out)

        self.n_out=n_out

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        return x

@register("flat")
def gym_flat_net(env_name, h_size=64):
    if env_name.lower() == "swingup": 
        env = CartPoleSwingUp()
    elif env_name.lower() == "custommc": 
        env = CustomMountainCarEnv()
    else:
        env=gym.make(env_name)
    n_out = env.action_space.n
    n_in = env.observation_space.shape[0]
    env.close()	
    def wrapped():
        return FFNet(n_in, h_size, n_out)
    return wrapped

## Procgen DQN

class ConvNet(nn.Module):
    def __init__(self, h_size, n_out, stacks=3, norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(stacks, 32, 8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        if norm: 
            self.norm = nn.LayerNorm([64, 4, 4])
        else:
            self.norm = lambda x: x
        self.conv_output_size = 1024
        self.fc1 = nn.Linear(self.conv_output_size, h_size)
        self.fc2 = nn.Linear(h_size, n_out)

        self.n_out=n_out

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.norm(x)
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@register("conv")
def gym_conv(env_name, h_size=512, norm=True):
    env_name = env_name.split("-")[0]
    env=gym.make(f"procgen:procgen-{env_name}-v0")
    n_out = env.action_space.n
    env.close()
    n_channels=3
    def wrapped():
        return ConvNet(h_size, n_out, norm=norm)
    return wrapped

@register("cont_conv")
def cont_conv(env_name, h_size=512, norm=True):
    env=gym.make(env_name)
    n_out = env.action_space.shape[0]
    env.close()
    n_channels=3
    def wrapped():
        return ConvNet(h_size, n_out, norm=norm)
    return wrapped

## Procgen: Data efficient from DQN

class DataEfficientConvNet(nn.Module):
    def __init__(self, h_size, n_out, stacks=3, norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(stacks, 32, 5, stride=5, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=5, padding=0)
        if norm: 
            self.norm = nn.LayerNorm([64, 2, 2])
        else:
            self.norm = lambda x: x
        self.conv_output_size = 256
        self.fc1 = nn.Linear(self.conv_output_size, h_size)
        self.fc2 = nn.Linear(h_size, n_out)

        self.n_out=n_out

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.norm(x)
        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@register("efficientconv")
def gym_conv_efficient(env_name, h_size=256, norm=True):
    env_name = env_name.split("-")[0]
    env=gym.make(f"procgen:procgen-{env_name}-v0")
    n_out = env.action_space.n
    env.close()
    n_channels=3
    def wrapped():
        return DataEfficientConvNet(h_size, n_out, norm=norm)
    return wrapped

@register("impala")
def impala(env_name, h_size=512, norm=False):
    if norm:
        print("WARNING: norm is not implemented for impala")
    env_name = env_name.split("-")[0]
    env=gym.make(f"procgen:procgen-{env_name}-v0")
    n_out = env.action_space.n
    env.close()
    n_channels=3
    def wrapped():
        return ImpalaModel(n_out=n_out, stacks=n_channels)
    return wrapped