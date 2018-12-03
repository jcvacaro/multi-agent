[//]: # (Image References)

[image1]: https://drive.google.com/uc?id=1JSrqS-ZUN9k4reNSdBxSxAlGt04jkA8h "Agent Training"
[image2]: https://drive.google.com/uc?id=1PK-b9DjV-wmPhbqTl_zoVbj-9vBXIdEZ "Agent Training 100 Episodes"
[image3]: https://drive.google.com/open?id=1XrRuwh7trawIMMiKR_uSvPNnyQ5r6Euf "MADDPG Agent Training"
[image4]: https://drive.google.com/open?id=1GLNXKCWVjYLHw4HB4ErDPDnp5Ijc7oX_  "MADDPG Agent Training 100 Episodes"

# Report

## Learning Algorithm

It is possible to identify four distinct components in DDPG-based algorithms: (1) the training loop, agent--environment interaction; (2) the DDPG algorithm itself; (3) the particular replay buffer strategy; and (4) the neural network used to generate the policy and the critic Q-values. Although DDPG is presented as an actor critic reinforcement learning method, it has many similarities with the value-base DQN algorithm.

### The training loop

`main.py` contains the traditional training loop:

- initialize the environment
- for each episode
    - get the initial state from the environment
    - for each step in the current episode
        - get the next action from the agent
        - get the next state and reward from the environment
        - the agent observes the new experience and improve its model
        - verifiy if the agent achieved the goal

### The DDPG algorithm
    
`ddpg_agent.py` contains theDDPG algorithm. Each time the training loop observes a new experience (state, action, reward, next state, done) the method `step` is invoked. The experience is stored into the replay buffer, and if the buffer contains at least batch size entries, a new batch of experience is sampled.

To adapt DDPG for multi-agent environments, the experiences of both agents are stored in the buffer as two independent entries. That's because each agent has its own observation. Since both tennis players perform the same action, the same network can be trained and use for inference. 

There are 4 networks involved in DDPG: the local actor, the local critic, the target actor, and the target critic. The `learn` method  trains such networks in three steps:

1. The critic loss: First, predicted next-state actions and Q values from the target models. The next action is obtained from the target actor, and the next Q-value from the target critic model. Then, the loss is obtained by the TD (temporal difference) between the just calculated next Q-value, and the current Q-value from the local critic network. 

2. The actor loss: We predict the action from the local actor model based on the current experience state, and then the Q-value from the local critic. Then, the critic Q-value is the actual loss for the actor, that is, the critic guides the direction of the actor gradient updates.

3. Soft updates: Instead of performing a huge update every n steps, DDPG slowly updates the target networks based on the local networks every step. Remember that the local networks are the most updated because those are the ones being trained. The target networks are slowly updated to maintain stability during learning.

Training the agent on the target environment was really involving. The following techniques contributed to the stability of the algorithm significantly:

- Increasing the batch size: Going from 128 to 1024 reduced the noise in the actor and critic loss. That is probably because a more significant sampling is performed from the total entries in the replay memory.
- Clipping the critic gradients: It is very clear the relationship between the critic and the actor loss, and of course that impacts the rewards. So, since the critic guides the actor gradient updates, reducing the variance of the critic improves the whole system. The goal is to use a clip function in the gradients to eliminate extreme updates in the critic network.
- Choosing the weight decay value for the optimizer: Here the actor loss plot helped a lot. The loss started very well decreasing over time, but after a certain number of episodes it diverged completely by increasing its value. The weight decay was criticaly important for solving this problem. By decreasing the learning rate during the optimization step probably contributed for the network to reach a better local minimum. 

### The MADDPG algorithm

MADDPG (Multi-Agent Deep Deterministic Policy Gradients) adopts the framework of centralized training with decentralized execution, allowing the policies to use extra information to ease training, so long as this information is not used at test time. Basically, the critic network can use additional information such as the state of other agents during training to guide the graident updates of its actor. Since the critic has a better view of the environment state, the stability of the training increases.

`maddpg_agent.py` contains theDDPG algorithm. The structure is very similar to DDPG, but now two separate agents are created, one for each tennis player. The `learn` method is also similar to the DDPG implementation, but now the critic has acess to the whole observation space, not only the local observation of each player.

### The replay Buffer

`memory.py` holds the implementation for the memory buffers strategies. The DDPG algorithm uses a uniform sampling buffer with the objective of training the model by first storing experiences in the buffer, and then replaying a batch of experiences from it in a subsequent step. The expectation is to reduce the correlation of such observations, which leads to a more stable training procedure. The implementation of this strategy is defined in the ReplayBuffer class.

### The neural network

`model.py` implements the neural network architecture for the actor and the critic. Both models are very similar, consisting of 3 Multilayer perceptron (MLP) layers. Each layer uses the RELU action function, except the last one, which has dimension equivalent to the number of actions and applies the `tanh` activation function. That is because `tanh` outputs values between -1 and 1, exactly the continuous action space needed for solving the target environment. The table below shows the complete network model configuration:

#### Actor

The actor represents the policy.

| Layer |  type  | Input | Output | Activation |
| ----- | ------ | ----- | ------ | ---------- |
| 1     | linear | 33    | 400    | RELU       |
| 2     | linear | 400   | 300    | RELU       |
| 3     | linear | 300   | 2      | TANH       |

#### Critic

The critic represents the Q-value function Q(s, a). The action is incorporated into the second layer of the network.

| Layer |  type  | Input | Output | Activation |
| ----- | ------ | ----- | ------ | ---------- |
| 1     | linear | 33    | 400    | RELU       |
| 2     | linear | 404   | 300    | RELU       |
| 3     | linear | 300   | 2      | TANH       |

## Results

### DDPG

By using DDPG, the goal was achieve in 1291 episodes. The following graphs show the reward progression of the entire training, as well as the last 100 episodes where the agent achieved +0.51 points.

![Agent Training][image1]
![Agent Training 100 Episodes][image2]

The complete configuration is shown in the table below.

| Parameter       | Description                                        | Value    |
| --------------- | -------------------------------------------------- | -------- |
| algorithm       | The actor-critic algorithm                         | DDPG     |
| replay_buffer   | The replay buffer strategy                         | uniform  |
| buffer_size     | The replay buffer size                             | 1e5      |
| batch_size      | The batch size                                     | 1024     |
| gamma           | The reward discount factor                         | 0.95     |
| tau             | For soft update of target parameters               | 1e-2     |
| lr_actor        | The learning rate                                  | 1e-4     |
| lr_critic       | The learning rate                                  | 1e-3     |
| clip_critic     | Clip the critic gradients during training          | 0.2      |
| weight_decay    | The decay value for the Adam optmizer algorithm    | 25e-5    |
| update_steps    | Number of steps to apply the learning procedure    | 2        |
| sgd_epoch       | Number of training iterations for each learning    | 1        |

### MADDPG

By using MADDPG, the goal was achieve in 7332 episodes. The following graphs show the reward progression of the entire training, as well as the last 100 episodes where the agent achieved +0.51 points.

![MADDPG Agent Training][image3]
![MADDPG Agent Training 100 Episodes][image4]

The complete configuration is shown in the table below.

| Parameter       | Description                                        | Value    |
| --------------- | -------------------------------------------------- | -------- |
| algorithm       | The actor-critic algorithm                         | MADDPG   |
| replay_buffer   | The replay buffer strategy                         | uniform  |
| buffer_size     | The replay buffer size                             | 1e5      |
| batch_size      | The batch size                                     | 1024     |
| gamma           | The reward discount factor                         | 0.95     |
| tau             | For soft update of target parameters               | 1e-2     |
| lr_actor        | The learning rate                                  | 1e-4     |
| lr_critic       | The learning rate                                  | 1e-3     |
| clip_critic     | Clip the critic gradients during training          | 0.2      |
| weight_decay    | The decay value for the Adam optmizer algorithm    | 1e-4     |
| update_steps    | Number of steps to apply the learning procedure    | 8        |
| sgd_epoch       | Number of training iterations for each learning    | 8        |
| seed            | Seed for random number generation                  | 1        |

## Ideas for future work

- Apply techniques for reducing variance: Note that the MADDPG training took much longer than DDPG. That's probably because two isolated neural networks are trained, instead of a single one in DDPG.
- Use Self Playing techniques for training: A possibility is to train both agents in separated steps, so that they will learn incrementally with each other.
- Increase the Critic neural network units: Since the critics have acess to the whole observation space, it may be the case that it requires a more complex architecture to learn.

## References

- [DDPG](https://arxiv.org/abs/1509.02971)
- [MADDPG](https://arxiv.org/abs/1706.02275)

