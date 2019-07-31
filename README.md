# Step into the AI Era: Deep Reinforcement Learning Workshop
In this workshop, through exercises, we will learn about (Deep) Reinforcement Learning and how to implement different strategies and train an agent to solve different tasks (or play games) in [OpenAI Gym](https://gym.openai.com/). For the consistency of the environment and make use of a free GPU, we will use [Google Colaboratory](http://colab.research.google.com/) (Google Account needed)

----

## Table of Contents

* **Part 1**
  * [What is Reinforcement Learning](#what-is-reinforcement-learning)
  * [101 of Reinforcement Learning](#101-of-reinforcement-learning)
  * [Crossentropy Method](#crossentropy-method)
    * [Exercise - Crossentropy Method](#)
    * [Exercise - Deep Crossentropy Method](#)
* **Part 2**
  * TBC
* **Part 3**
  * TBC
* [References](#references)

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


## References
[1]: https://en.wikipedia.org/wiki/Reinforcement_learning
[2]: https://commons.wikimedia.org/wiki/File:Markov_Decision_Process.svg
