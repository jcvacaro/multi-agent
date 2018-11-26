import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

from utils import *
from network import Network
from OUNoise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AgentNetwork:
    """Individual network settings for each actor + critic pair."""

    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2):
        super(AgentNetwork, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)
        
        # checkpoint
        self.actor_loss_episodes = []
        self.critic_loss_episodes = []
        self.actor_loss = 0
        self.critic_loss = 0

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, num_agents, state_size, action_size, noise, noise_decay, seed, memory, batch_size, lr_actor, lr_critic, clip_critic, gamma, tau, weight_decay, update_network_steps, sgd_epoch, checkpoint_prefix):
        """Initialize an Agent object.
        
        Params
        ======
            num_agents (int): The number of agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            noise (float): The amplitude of OU noise for action exploration
            noise_decay (float): The noise reduction value
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
        self.num_agents = num_agents
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

        # Agents
        critic_size = num_agents * (action_size + state_size)
        self.agents = [AgentNetwork(state_size, 3*state_size, 2*state_size, action_size, 
                                    critic_size, 3*critic_size, 2*critic_size, 
                                    lr_actor=lr_actor, lr_critic=lr_critic) for i in range(num_agents)]

        # Noise process
        self.noise = torch.tensor(noise, dtype=torch.float).to(device)
        self.noise_decay = torch.tensor(noise_decay, dtype=torch.float).to(device)
        self.ounoise = OUNoise(action_size, scale=1.0)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
         
        # learn every n steps
        self.n_step = (self.n_step + 1) % self.update_network_steps
        if self.n_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                for i in range(self.sgd_epoch):
                    for a_i in range(self.num_agents):
                        self.learn(experiences, a_i)
                        
                # noise update
                self.noise *= self.noise_decay
                # soft update the target network towards the actual networks
                self.update_targets()

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        return [agent.actor for agent in self.agents]

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        return [agent.target_actor for agent in self.agents]

    def _act(self, actors, state, add_noise=True, to_numpy=True):
        """get actions from all agents in the MADDPG object"""
        if to_numpy:
            state = torch.from_numpy(state).float().to(device)
        action_list = [actor(state[:,i]) for i,actor in enumerate(actors)]
        # convert from list([batch, data], ...)  to tensor([batch, agent, data])
        actions = torch.stack(action_list, dim=-1).to(device)
        if add_noise:
            actions += self.noise * torch.tensor(self.ounoise.noise(), dtype=torch.float).to(device)
            actions = torch.clamp(actions, -1, 1)
        if to_numpy:
            actions = actions.cpu().data.numpy()
        return actions
        
    def act(self, state, add_noise=True, to_numpy=True):
        """get actions from all agents in the MADDPG object"""
        return self._act(self.get_actors(), state, add_noise, to_numpy)

    def target_act(self, state, add_noise=True, to_numpy=True):
        """get actions from all agents in the MADDPG object"""
        return self._act(self.get_target_actors(), state, add_noise, to_numpy)

    def learn(self, experiences, agent_number):
        """update the critics and actors of all the agents """
        states, actions, rewards, next_states, dones = experiences
        states_i, actions_i, rewards_i, next_states_i, dones_i = map(lambda x: x[:,agent_number], experiences)
        agent = self.agents[agent_number]
        
        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        agent.critic_optimizer.zero_grad()

        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_states, to_numpy=False)
        target_critic_input = torch.cat([next_states.view(self.batch_size, -1), target_actions.view(self.batch_size, -1)], dim=1).to(device)
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        y = rewards_i + self.gamma * q_next * (1 - dones_i)

        # Q(s,a)
        critic_input = torch.cat([states.view(self.batch_size, -1), actions.view(self.batch_size, -1)], dim=1).to(device)
        q = agent.critic(critic_input)

        # critic loss
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        agent.critic_loss = critic_loss
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [self.agents[i].actor(states[:,i]) if i == agent_number
                   else self.agents[i].actor(states[:,i]).detach()
                   for i in range(self.num_agents)]
        q_input = torch.stack(q_input, dim=-1).to(device)
        
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat([states.view(self.batch_size, -1), q_input.view(self.batch_size, -1)], dim=1).to(device)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        agent.actor_loss = actor_loss
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

    def update_targets(self):
        """soft update targets"""
        for agent in self.agents:
            soft_update(agent.target_actor, agent.actor, self.tau)
            soft_update(agent.target_critic, agent.critic, self.tau)

    def checkpoint(self):
        """Save internal information in memory for later checkpointing"""
        for agent in self.agents:
            agent.actor_loss_episodes.append(agent.actor_loss)
            agent.critic_loss_episodes.append(agent.critic_loss)

    def save_checkpoint(self):
        """Persist checkpoint information"""
        for i,agent in enumerate(self.agents):
            # the history loss
            plot_scores(self.checkpoint_prefix + "_agent_" + str(i) + "_actor_loss.png", agent.actor_loss_episodes, label="loss")
            plot_scores(self.checkpoint_prefix + "_agent_" + str(i) + "_critic_loss.png", agent.critic_loss_episodes, label="loss")
            
            # network
            torch.save(agent.actor.state_dict(), self.checkpoint_prefix + "_agent_" + str(i) + "_actor.pth")
            torch.save(agent.critic.state_dict(), self.checkpoint_prefix + "_agent_" + str(i) + "_critic.pth")

    def load_checkpoint(self):
        """Restore checkpoint information"""
        for i,agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(self.checkpoint_prefix + "_agent_" + str(i)  + "_actor.pth"))
            agent.critic.load_state_dict(torch.load(self.checkpoint_prefix + "_agent_" + str(i)  + "_critic.pth"))
