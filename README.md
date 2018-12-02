# Multi-Agent

This repository contains implementations in Pytorch of DDPG (Deep Deterministic Policy Gradients) and MADDPG (Multi-Agent Deep Deterministic Policy Gradients) for solving multi-agent tasks. The environment is defined in the Unity engine, and the communication is based on the Unity ML Agents API. See the Report.md file for in depth details about the algorithms used and the organization of the source code.

## Goal

The target environment is the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, the rewards that each agent receives are added up (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. Then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Getting Started

The source code is implemented in Python 3x, and uses PyTorch as the Machine Learning framework. 

1. Install PyTorch
    - Windows: [Anaconda](https://conda.io/docs/user-guide/install/windows.html), [PyTorch](https://pytorch.org/get-started/locally/)
    - Linux: [Anaconda](https://conda.io/docs/user-guide/install/linux.html), [PyTorch](https://pytorch.org/get-started/locally/)
    - Docker: See the Dockerfile for instructions about how to generate the image
2. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - [Linux](https://drive.google.com/uc?id=1UF_rA9HkMF3TnEB4IDkFJ30DRDO-yuAB), [Windows 64-bit](https://drive.google.com/uc?id=1TgV2f1Fqj2UJnqad3S5r6LbmZdBG4RJt)
3. To test the agent with a pre-trained DDPG network, download the model checkpoint:
    - [Actor](https://drive.google.com/uc?id=1qUK_Xax6M92Y2J-071uKAnE0H-cvmMPt), [Critic](https://drive.google.com/uc?id=1IiZtLqG5ZLJW1H5VbJiWvaT7A7vmk02T)
4. Place the file(s) in the repository folder, and unzip (or decompress) the file(s).

## Instructions

The main.py is the application entry point. To show all available options:

```bash
python main.py --help
```

To train the agent:

```bash
python main.py --train \
    --algorithm=ddpg \
     --checkpoint_prefix=Tennis_ddpg \
     --train_episodes=10000 \
     --gamma=0.95 \
     --tau=0.01 \
    --lr_actor=1e-4 \
    --lr_critic=1e-3 \
    --weight_decay=25e-5 \
    --clip_critic=0.2 \
    --batch_size=1024 \
    --update_network_steps=2 \
    --sgd_epoch=1
```

This command configures the agent to use the DDPG algorithm. The training runs 10000 episodes, and saves the model checkpoint in the current directory if the goal is achieved.

To use the MADDPG algorithm instead (experimental):

```bash
python main.py --train \
    --algorithm=maddpg \
     --checkpoint_prefix=Tennis_maddpg \
     --train_episodes=10000 \
     --gamma=0.95 \
     --tau=0.01 \
    --lr_actor=1e-4 \
    --lr_critic=1e-3 \
    --weight_decay=25e-5 \
    --clip_critic=0.2 \
    --batch_norm \
    --batch_size=1024 \
    --update_network_steps=8 \
    --sgd_epoch=8
```

To test the agent using the DDPG model checkpoint:

```bash
python main.py \
    --algorithm=ddpg \
    --test_episodes=1 \
    --checkpoint_prefix=Tennis_ddpg
```

In addition, many hyper parameters can be customized such as the learning rate, the reward discount factor gamma. Check the --help for all available options.
