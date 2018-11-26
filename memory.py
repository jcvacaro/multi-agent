import math
import heapq
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class experience:
    """Represents an experience.
    
    It provides the __lt__ method so that it can be added to a heapq.
    """
    
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, state_size, buffer_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            state_size (int): dimension of each state
            buffer_size (int): maximum size of buffer
        """
        self.action_size = action_size
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)  
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = self.sample_experiences()
        batch_size = len(experiences)
        
        states = np.zeros((batch_size, *experiences[0].state.shape), dtype=np.float)
        actions = np.zeros((batch_size, *experiences[0].action.shape), dtype=np.float)
        rewards = np.zeros((batch_size, *experiences[0].reward.shape), dtype=np.float)
        next_states = np.zeros((batch_size, *experiences[0].state.shape), dtype=np.float)
        dones = np.zeros((batch_size, *experiences[0].done.shape), dtype=np.float)
        
        for i, e in enumerate(experiences):
            states[i] = e.state
            actions[i] = e.action
            rewards[i] = e.reward
            next_states[i] = e.next_state
            dones[i] = e.done

        # place tensors in GPU for faster calculations
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def sample_experiences(self):
        pass

class UniformReplayBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, state_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            state_size (int): dimension of each state
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        super(UniformReplayBuffer, self).__init__(action_size, state_size, buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def sample_experiences(self):
        return random.sample(self.memory, k=self.batch_size)

