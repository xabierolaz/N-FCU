import sys

import torch
import torch.nn as nn
import numpy as np
import rospy

from scipy.spatial.transform import Rotation as R

from models.ppo.ppo_model import agent
from models.ppo.memory_v2 import memory, upper_memory
from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter


## HYPERPARAMETERS - CHANGE IF NECESSARY ##
actor_lr = 0.0003
critic_lr = 0.001

#action_std = 0.1 # normal landing
action_std = 0.25 # platform landing
eps_clip = 0.2
betas = (0.9, 0.999)

H0 = 64
H1 = 64


def plot_returns(returns, values, terminals, start, end):
    f, ax = plt.subplots(figsize=(12,12))
    # delete_indexes = np.where(terminals)[0] + 1
    # values = np.delete(values, delete_indexes)
    # print(returns, values)
    ax.plot(returns[start:end])
    ax.plot(values[start:end])
    for i, value in enumerate(terminals[start:end]):
        if value:
            ax.vlines(i, min(returns), max(returns))
    return f


class PPO:
    def __init__(self,
        input_size, 
        output_size, 
        problem_type, 
        batch_size, 
        device):
        
        self.device = torch.device("cuda:0")
        self.critic_loss_memory = []
        self.actor_loss_memory = []
        self.critic_epoch_loss = []
        self.actor_epoch_loss = []
        self.batch_size = batch_size

        self.epoch = 0
        self.train_step = 0

        self.eps_clip = eps_clip

        self.policy = agent(input_size, H0, H1, output_size, action_std, self.device)

        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': 0.0003},
                {'params': self.policy.critic.parameters(), 'lr': 0.001}
            ])

        self.path = 'models/checkpoints/'+problem_type+'/'+'ppo_checkpoint.pt'
        try:
            checkpoint = torch.load(self.path, map_location=self.device)

            self.policy.load_state_dict(checkpoint['policy'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']
            self.train_step = checkpoint['train_step']
            self.policy.std = checkpoint['std']
            # print('Saved Landing Policy loaded')
        except:
            checkpoint = {
                'policy': self.policy.state_dict(), 
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'train_step': self.train_step,
                'std': self.policy.std
            }
            torch.save(checkpoint, self.path)
            print('New Landing Policy generated')
            pass

        self.MseLoss = nn.MSELoss()

    def eval_action(self, state):
        state = torch.Tensor(state).to(self.device).detach().unsqueeze(dim=0)
        out, state_value = self.policy.noiseless_act(state)
        out = out.reshape(-1)
        return out, state_value

    def select_action(self, state, memory):
        state = torch.Tensor(state).to(self.device).detach().unsqueeze(dim=0)
        out, state_value = self.policy.act(state, memory)
        out = out.reshape(-1)
        return out, state_value

    def update(self, np_dict):
        self.epoch += 1
        self.train_step += len(np_dict['states'])

        for _ in range(80):

            random_index = torch.randperm(len(np_dict['rewards']))
            old_states = torch.tensor(np_dict['states']).detach().to(self.device, dtype=torch.float)[random_index]
            old_actions = torch.tensor(np_dict['actions']).detach().to(self.device, dtype=torch.float)[random_index]
            old_logprobs = torch.tensor(np_dict['logprobs']).detach().to(self.device, dtype=torch.float)[random_index]
            returns = torch.tensor(np_dict['returns']).detach().to(self.device, dtype=torch.float)[random_index]
            advantages = torch.tensor(np_dict['advantages']).detach().to(self.device, dtype=torch.float)[random_index]

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

            surr1 = ratios * advantages.unsqueeze(dim=1)
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.unsqueeze(dim=1)
            # print(logprobs.shape, state_values.shape, dist_entropy.shape, surr1.shape, surr2.shape, advantages.shape)

            # final loss of clipped objective PPO
            loss_ac = torch.min(surr1, surr2).sum(axis=1)
            loss_ct = 0.5*nn.MSELoss()(state_values, returns)
            ent_loss = 0.01*dist_entropy.sum(axis=1)
            loss = -loss_ac + loss_ct - ent_loss
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        std = self.policy.std

        random_init = np.random.randint(0, len(np_dict['returns'])-min(self.batch_size, 600))
        returns_figure = plot_returns(
            np_dict['returns'], 
            np_dict['state_values'], 
            np_dict['is_terminals'],
            start = random_init,
            end = random_init + min(self.batch_size, 600) - 10)

        checkpoint = {
            'policy': self.policy.state_dict(), 
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'train_step': self.train_step,
            'std': self.policy.std
        }
        
        torch.save(checkpoint, self.path)

        return loss_ac.detach().cpu().numpy(), loss_ct.detach().cpu().numpy(), list(np_dict['returns']), returns_figure, std, self.epoch