# Step into the AI Era: Deep Reinforcement Learning Workshop
In this workshop, through exercises, we will learn about (Deep) Reinforcement Learning and how to implement different strategies and train an agent to solve different tasks (or play games) in [OpenAI Gym](https://gym.openai.com/). For the consistency of the environment and make use of a free GPU, we will use [Google Colaboratory](http://colab.research.google.com/) (Google Account needed)

----

## Table of Contents

* **Part 1**
  * [What is Reinforcement Learning](#what-is-reinforcement-learning)
  * [Agent, Environment, Rewards and MDP](#agent-environment-rewards-and-mdp)
  * [Crossentropy Method](#)
  * [Exercise - Crossentropy Method](#)
  * [Exercise - Deep Crossentropy Method](#)
* **Part 2**
  * TBC
* **Part 3**
  * TBC

----
## What is Reinforcement Learning
Also classified as machine learning, what makes reinforcement learning stands out is that an example is not necessary for training, so it is not supervised learning. However, different from un-supervised learning like k-mean clustering or anomaly detection, reinforcement learning takes a bottom-up approach rather than top-down approach. By trying out different actions with different policy and record different outcomes (rewards), we train an agent that creates it's own 'training data' from trials and 'learn' from it. Sometime, reinforcement learning is listing alongside supervised learning and unsupervised learning as one of three basic machine learning paradigms[1]

## Agent, Environment, Rewards and MDP

The 101 of Reinforcement Learning is define a set of agent states in the environment, set of actions that can be taken by the agent and what rewards those can lead to. The very basic of how this works is to make use of Markov decision process (MDP).

#### Markov decision process

![Markov decision process](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Markov_Decision_Process.svg/800px-Markov_Decision_Process.svg.png)

To explain, for the agent at each state, it can take an action which will have different probability to move to a different state which will lead to different rewards.

## Reference
[1]: https://en.wikipedia.org/wiki/Reinforcement_learning
