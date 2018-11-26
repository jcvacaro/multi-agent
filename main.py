from unityagents import UnityEnvironment
import random
import torch
import numpy as np
from collections import deque
import os
import argparse
import matplotlib.pyplot as plt

import utils
from memory import UniformReplayBuffer
import maddpg_agent as maddpg

# Arguments Parsing Settings
parser = argparse.ArgumentParser(description="DQN Reinforcement Learning Agent")

parser.add_argument('--seed', help="Seed for random number generation", type=int, default=1)
parser.add_argument('--env', help="The environment path", default="Tennis/Tennis.app")
parser.add_argument('--checkpoint_prefix', help="The string prefix for saving checkpoint files", default="Tennis")

# training/testing flags
parser.add_argument('--train', help="train or test (flag)", action="store_true")
parser.add_argument('--algorithm', choices=["maddpg"], help="The algorithm", default="maddpg")
parser.add_argument('--test_episodes', help="The number of episodes for testing", type=int, default=3)
parser.add_argument('--train_episodes', help="The number of episodes for training", type=int, default=500)
parser.add_argument('--batch_size', help="The mini batch size", type=int, default=128)
parser.add_argument('--noise', help="The amplitude of OU noise for action exploration", type=float, default=2)
parser.add_argument('--noise_decay', help="The noise coefficient decay", type=float, default=0.9999)
parser.add_argument('--gamma', help="The reward discount factor", type=float, default=0.99)
parser.add_argument('--lr_actor', help="The learning rate for the actor", type=float, default=1e-4)
parser.add_argument('--lr_critic', help="The learning rate for the critic", type=float, default=1e-4)
parser.add_argument('--clip_critic', help="The clip value for updating grads", type=float, default=1)
parser.add_argument('--tau', help="For soft update of target parameters", type=float, default=1e-2)
parser.add_argument('--weight_decay', help="The weight decay", type=float, default=1e-7)
parser.add_argument('--update_network_steps', help="How often to update the network", type=int, default=4)
parser.add_argument('--sgd_epoch', help="Number of iterations for each network update", type=int, default=1)

# replay memory 
parser.add_argument('--buffer_type', choices=["uniform"], help="The replay buffer type", default="uniform")
parser.add_argument('--buffer_size', help="The replay buffer size", type=int, default=int(1e5))

def create_environment(no_graphics=False):
    env = UnityEnvironment(file_name=args.env, no_graphics=no_graphics)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    return env, brain, brain_name, num_agents, action_size, state_size

def test(agent, env, brain, brain_name, num_agents, n_episodes):
    scores_episodes = deque(maxlen=n_episodes)                  # The score history over all episodes
    agent.load_checkpoint()                                     # loads a pth checkpoint 

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]      # reset the environment
        states = env_info.vector_observations                   # get initial observation
        scores = np.zeros(num_agents)                           # initialize the score (for each agent)

        while True:
            # agent chooses an action
            actions, action_probs = agent.act(states)
            
            # interact with the environment
            env_info = env.step(actions)[brain_name]            # send all actions to tne environment
            next_states = env_info.vector_observations          # get next state (for each agent)
            rewards = env_info.rewards                          # get reward (for each agent)
            dones = env_info.local_done                         # see if episode finished
            
            scores += rewards                                   # update the score (for each agent)
            states = next_states                                # roll over states to next time step
            if np.any(dones):                                   # exit loop if episode finished
                break

        # checkpoint 
        score = np.max(scores)                                  # max over agents
        scores_episodes.append(score)                           # save most recent score in the history
        score_mean = np.mean(scores_episodes)                   # the mean of all episodes so far
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score_mean))

def train(agent, env, brain, brain_name, num_agents, n_episodes):
    scores_episodes = deque(maxlen=n_episodes)                  # The score history over all episodes
    scores_window = deque(maxlen=100)                           # last 100 scores

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        states = env_info.vector_observations                   # get initial observation
        scores = np.zeros(num_agents)                           # initialize the score (for each agent)

        while True:
            # agent chooses an action
            states_batch = states[np.newaxis,:]                 # add batch dim to states
            actions = agent.act(states_batch)[0]                      # remove batch dim from actions
            
            # interact with the environment
            env_info = env.step(actions)[brain_name]            # send all actions to tne environment
            next_states = env_info.vector_observations          # get next state (for each agent)
            rewards = env_info.rewards                          # get reward (for each agent)
            dones = env_info.local_done                         # see if episode finished
            
            # agent learns with the new experience
            agent.step(np.asarray(states), 
                       np.asarray(actions), 
                       np.asarray(rewards)[:, np.newaxis],
                       np.asarray(next_states), 
                       np.asarray(dones)[:, np.newaxis])

            scores += rewards                                   # update the score (for each agent)
            states = next_states                                # roll over states to next time step
            if np.any(dones):                                   # exit loop if episode finished
                break

        # checkpoint 
        score = np.max(scores)                                  # max score over the agents
        scores_episodes.append(score)                           # save most recent score in the history
        scores_window.append(score)                             # save most recent score
        agent.checkpoint()                                      # agent checkpoint
        
        # verify if the goal has been achieved
        score_window = np.mean(scores_window)                   # the mean of the last 100 episodes
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score_window), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score_window))
            save_checkpoint(agent, scores_episodes, scores_window)
        if score_window >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, score_window))
            break

    # plot score history
    save_checkpoint(agent, scores_episodes, scores_window)
  
def save_checkpoint(agent, scores_episodes, scores_window):
    utils.plot_scores(args.checkpoint_prefix + "_reward_history_plot.png", scores_episodes)
    utils.plot_scores(args.checkpoint_prefix + "_reward_plot.png", scores_window)
    agent.save_checkpoint()

if __name__ == '__main__':
    args = parser.parse_args()
    
    # seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # environment
    env, brain, brain_name, num_agents, action_size, state_size = create_environment(no_graphics=args.train)

    # replay memory
    if args.buffer_type == "uniform":
        memory = UniformReplayBuffer(action_size=action_size, 
                                     state_size=state_size, 
                                     buffer_size=args.buffer_size,
                                     batch_size=args.batch_size,
                                     seed=args.seed)

    # agent
    if args.algorithm == "maddpg":
        agent = maddpg.Agent(num_agents=num_agents,
                             state_size=state_size, 
                             action_size=action_size, 
                             noise=args.noise,
                             noise_decay=args.noise_decay,
                             seed=args.seed,
                             batch_size=args.batch_size,
                             memory=memory,
                             lr_actor=args.lr_actor,
                             lr_critic=args.lr_critic,
                             clip_critic=args.clip_critic,
                             gamma=args.gamma,
                             tau=args.tau,
                             weight_decay=args.weight_decay,
                             update_network_steps=args.update_network_steps,
                             sgd_epoch=args.sgd_epoch,
                             checkpoint_prefix=args.checkpoint_prefix)

    if args.train:
        train(agent, env, brain, brain_name, num_agents, n_episodes=args.train_episodes)
    else:
        test(agent, env, brain, brain_name, num_agents, n_episodes=args.test_episodes)

    env.close()
