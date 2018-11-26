import random
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_scores(filename, scores, label='Score'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel(label)
    plt.xlabel('Episode #')
    plt.savefig(filename, transparent=False)
    plt.close()

def torch_reverse(tensor, axis=0):
    idx = [i for i in range(tensor.size(axis)-1, -1, -1)]
    idx = torch.LongTensor(idx).to(device)
    return tensor.index_select(axis, idx)

def future_rewards(rewards, axis=0):
    return torch_reverse(torch_reverse(rewards, axis=axis).cumsum(axis), axis=axis)

def normalize_rewards(rewards, axis=0):
    return (rewards - rewards.mean().float()) / (rewards.std().float() + 1.0e-10)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
