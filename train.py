import os
import yaml
import argparse
from datetime import datetime

from sacd.env import make_pytorch_env
from sacd.agent import SacdAgent, SharedSacdAgent


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd
import gym_cancer


patientNo = 11
os.chdir("/Users/michael/OneDrive - City University of Hong Kong/Project/Net Embedding_RL_Cancer/CancerNN/Model_creation")
pars = pd.read_csv("pars_patient11.csv")
pars = pars.to_numpy().reshape(-1)
pars = torch.from_numpy(pars).float().requires_grad_()
K = 5e+8 # 5cm^3 = 5000mm^3
K1 = 5e+8
K2 = 2.5e+8
K = torch.tensor([K1,K2])
r = torch.tensor([2., 1.], dtype = torch.float) # AD is 2 times more responsive compared to AI cell
e = torch.tensor([[1], [0.5]], dtype = torch.float) # AD is 5 times more competitive effects exert than AI cells
A = r * e#torch.tensor([1.,0.9,0.9,1.], dtype = torch.float).view(2,2)
alpha = torch.tensor([0.462], dtype = torch.float)  

patient = {"A": A, "alpha": alpha, "K": K, "pars": pars}

# env_dict = gym.envs.registration.registry.env_specs.copy()
# for env in env_dict:
#     if 'CancerControl-v0' in env:
#         print("Remove {} from registry".format(env))
#         del gym.envs.registration.registry.env_specs[env]




def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = gym.make(args.env_id, patient = args.patient).unwrapped
    test_env = gym.make(args.env_id, patient = args.patient).unwrapped

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    # if args.shared:
    #     name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    Agent = SacdAgent #if not args.shared else SharedSacdAgent
    agent = Agent(
        env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda,
        seed=args.seed, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='gym_cancer:CancerControl-v0')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patient', type = dict, default = patient)
    args = parser.parse_args()
    run(args)


