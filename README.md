# Step into the AI Era: Deep Reinforcement Learning Workshop
In this workshop, through exercises, we will learn about (Deep) Reinforcement Learning and how to implement different strategies and train an agent to solve different tasks (or play games) in [OpenAI Gym](https://gym.openai.com/). For the consistency of the environment and make use of a free GPU, we will use [Google Colaboratory](http://colab.research.google.com/) (Google Account needed)

----

## Table of Contents

* **Part 1**
  * [What is Reinforcement Learning](#what-is-reinforcement-learning)
  * [101 of Reinforcement Learning](#101-of-reinforcement-learning)
  * [Crossentropy Method](#crossentropy-method)
    * [Exercise - Crossentropy Method](https://colab.research.google.com/github/Cheukting/rl_workshop/blob/master/exercises/rl_workshop_crossentropy_method.ipynb)
    * [Exercise - Deep Crossentropy Method](https://colab.research.google.com/github/Cheukting/rl_workshop/blob/master/exercises/rl_workshop_deep_crossentropy_method.ipynb)
* **Part 2**
  * [Model-free Model](#model-free-model)
  * [Cliff World: Q-learning vs SARSA](#cliff-world-q-learning-vs-sarsa)
    * [Exercise - Cliff World](https://colab.research.google.com/github/Cheukting/rl_workshop/blob/master/exercises/rl_workshop_cliff_world.ipynb)
* **Part 3**
  * [Experience Replay](#experience-replay)
  * [Approximate Q-learning and Deep Q-Network](#approximate-q-learning-and-deep-q-network)
    * [Exercise - DQN](https://colab.research.google.com/github/Cheukting/rl_workshop/blob/master/exercises/rl_workshop_dqn.ipynb)
----
## What is Reinforcement Learning
Also classified as machine learning, what makes reinforcement learning stands out is that an example is not necessary for training, so it is not supervised learning. However, different from un-supervised learning like k-mean clustering or anomaly detection, reinforcement learning takes a bottom-up approach rather than top-down approach. By trying out different actions with different policy and record different outcomes (rewards), we train an agent that creates it's own 'training data' from trials and 'learn' from it. Sometime, reinforcement learning is listing alongside supervised learning and unsupervised learning as one of three basic machine learning paradigms[1]

## 101 of Reinforcement Learning

First we will go through the basics of reinforcement learning. Almost all problems we solved using reinforcement learning will involve defining a set of agent states in the environment and a set of actions that can be taken by the agent with what rewards those can lead to. The very basic of how this works is to make use of Markov decision process (MDP).

#### Markov decision process

![Markov decision process](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Markov_Decision_Process.svg/800px-Markov_Decision_Process.svg.png)

To explain, for the agent at each state, it can take an action which will have different probability to move to a different state which will lead to different rewards.[2]

#### Finding the 'winning' policy

The goal of reinforcement learning is to find the policy, a strategies of what series of actions to take, that gain the maximum rewards possible. You may want to try brute force, which is to try all combination of actions to take and pick the best policy. But most of the time it will not work as the number of policies can be large, or even infinite. Practically speaking, we will need other algorithms to pick the best policy (or the best one we came across so far). We will introduce some of the popular ones in this workshop. We will also try to implement them in Python (with Keras and Tensorflow) to solve problems or play games in OpenAI Gym.

## Crossentropy Method

Corssentropy method is considered as Monte Carlo methods as it's mechanism involve trying different actions many times, provided that:

1. the MDP is finite
2. sufficient memory is available
3. problem is episodic
4. after each episode a new one starts fresh

For details and mathematic explanation of crossentropy method can be found on [Wikipedia](https://en.wikipedia.org/wiki/Cross-entropy_method). To summarize, overview of what we gonna do with crossentropy method:

> While it has not converge:
> 1. Sample N policies with the current distribution
> 2. Evaluate the N policies
> 3. Choosing the best m% of the policies
> 4. Update the distribution according to the policies we have chosen

## Exercises

- Crossentropy Method [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cheukting/rl_workshop/blob/master/exercises/rl_workshop_crossentropy_method.ipynb)

- Deep Crossentropy Method [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cheukting/rl_workshop/blob/master/exercises/rl_workshop_deep_crossentropy_method.ipynb)

----
## Model-free Model

So far we have to know exactly what will happened when we takes a certain action at a current state. That is the rewards and the next state for each state-action pair. What if we are not sure (which is most of what happened in real life) and can only have expectation values for the rewards Qπ(s,a) at a certain state-action (from a statistic point of view). Here comes the Model-free Model, the differences are:

Model-based: you know P(s'|s,a)
 - can apply dynamic programming
 - can plan ahead

Model-free: you can sample trajectories
 - can try stuff out
 - insurance not included

 To find the expectation, there are 2 strategies:

1: Monte-Carlo

- Averages Q over sampled paths
- Needs full trajectory to learn
- Less reliant on markov property

2: temporal difference

- Uses recurrent formula for Q
- Learns from partial trajectory
- Works with infinite MDP
- Needs less experience to learn

## Cliff World: Q-learning vs SARSA

Some time it can also be reference as off-policy vs on-policy, the different between Q-learning and SARSA is

on-policy (e.g. SARSA)

Agent can pick actions
Agent always follows his own policy

off-policy (e.g. Q-learning)

Agent can't pick actions
Learning with exploration, playing without exploration
Learning from expert (expert is imperfect)
Learning from sessions (recorded data)

One famous example is the Cliff World:
![Cliff World](http://ai.berkeley.edu/projects/release/reinforcement/v1/001/discountgrid.png)

As you can see, in theory, if the agent always picks the most optimal path (off-policy/Q-learning) it will always pick the lower path. However, during training, the epsilon-greedy “exploration" (With probability ε take random action; otherwise, take optimal action) can make the robot easily fail as one step downwards will push the robot in the gutter (-10 rewards) so the agent will actually never learn the 'optimal path'. In this case, SARSA (on-policy) is more desirable as it gets optimal rewards under current policy, so for the path at the bottom, the exception rewards for each tile is low as it also count the risk of stepping into the gutter (by mistake or exploration).

## Exercise

- Cliff World [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cheukting/rl_workshop/blob/master/exercises/rl_workshop_cliff_world.ipynb)

----
## Experience Replay

In deep learning, the same set of data will be used to train the model in many epoch. However, the 'training data' we have so far are only used one off. Sometime the game takes a long time to play it once and thus, training will be computationally expensive.

To slightly improve this situation, we can store the 'gaming' experience with a buffer. Then we can train on random subsamples of it so we don't need to re-visit same (s,a) many times in playing the game to learn it. Also note that it only works with off-policy algorithms.

## Approximate Q-learning and Deep Q-Network

State space can be large, and sometimes continuous, so kind of like what we did to make Crossentropy Method into Deep Crossentropy Method, we can approximate agent with a function and learn Q value using neural network. This is what we will do in the following exercise.

The famous DQN Paper was published by Google Deep Mind to play Atari Breakout in 2015, the design involve stacking 4 flames together so you can 'see' the action of the ball movement and use a CNN as an agent. We will try implementing it in the last exercise, before that, feel free to checkout the [video](https://www.youtube.com/embed/V1eYniJ0Rnk?enablejsapi=1) of how a fully trained agent play the game.

## Exercise

- DQN [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cheukting/rl_workshop/blob/master/exercises/rl_workshop_dqn.ipynb)

<!--- References --->

[1]: https://en.wikipedia.org/wiki/Reinforcement_learning

[2]: https://commons.wikimedia.org/wiki/File:Markov_Decision_Process.svg
