import numpy as np
import random
import copy
from collections import namedtuple, deque

import utils
from model import Actor, Critic
from OUNoise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, seed, memory, batch_size, lr_actor, lr_critic, clip_critic, gamma, tau, weight_decay, update_network_steps, sgd_epoch, checkpoint_prefix):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            memory (ReplayBuffer): The replay buffer for storing xperiences
            batch_size (int): Number of experiences to sample from the memory
            lr_actor (float): The learning rate for the actor
            lr_critic (float): The learning rate critic
            clip_critic (float): The clip value for updating grads
            gamma (float): The reward discount factor
            tau (float): For soft update of target parameters
            weight_decay (float): The weight decay
            update_network_steps (int): How often to update the network
            sgd_epoch (int): Number of iterations for each network update
            checkpoint_prefix (string): The string prefix for saving checkpoint files
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.memory = memory
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.clip_critic = clip_critic
        self.gamma = gamma
        self.tau = tau
        self.weight_decay = weight_decay
        self.update_network_steps = update_network_steps
        self.sgd_epoch = sgd_epoch
        self.n_step = 0
        
        # checkpoint
        self.checkpoint_prefix = checkpoint_prefix
        self.actor_loss_episodes = []
        self.critic_loss_episodes = []
        self.actor_loss = 0
        self.critic_loss = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(len(state)):
            self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])
         
        # learn every n steps
        self.n_step = (self.n_step + 1) % self.update_network_steps
        if self.n_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.batch_size:
                for i in range(self.sgd_epoch):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # normalize rewards
        #rewards = utils.normalize_rewards(rewards)
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_loss = critic_loss
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_critic > 0:
            torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), self.clip_critic)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_loss = actor_loss
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        utils.soft_update(self.critic_target, self.critic_local, self.tau)
        utils.soft_update(self.actor_target, self.actor_local, self.tau)                     

    def checkpoint(self):
        """Save internal information in memory for later checkpointing"""
        self.actor_loss_episodes.append(self.actor_loss)
        self.critic_loss_episodes.append(self.critic_loss)

    def save_checkpoint(self):
        """Persist checkpoint information"""
        # the history loss
        utils.plot_scores(self.checkpoint_prefix + "_actor_loss.png", self.actor_loss_episodes, label="loss")
        utils.plot_scores(self.checkpoint_prefix + "_critic_loss.png", self.critic_loss_episodes, label="loss")
        
        # network
        torch.save(self.actor_local.state_dict(), self.checkpoint_prefix + "_actor.pth")
        torch.save(self.critic_local.state_dict(), self.checkpoint_prefix + "_critic.pth")

    def load_checkpoint(self):
        """Restore checkpoint information"""
        self.actor_local.load_state_dict(torch.load(self.checkpoint_prefix + "_actor.pth"))
        self.critic_local.load_state_dict(torch.load(self.checkpoint_prefix + "_critic.pth"))
        
