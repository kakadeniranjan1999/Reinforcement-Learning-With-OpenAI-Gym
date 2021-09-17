# Reinforcement-Learning-RL-A-Quick-Overview

This branch of the repository introduces you to [OpenAI Gym](https://gym.openai.com/), a toolkit for developing and comparing reinforcement learning algorithms developed by [OpenAI](https://openai.com/).

## Table of Contents
* [Getting Started](#i-getting-started)
  * [System Requirements](#system-requirements)
  * [Installation and Setup](#installation-and-setup)
  * [OpenAI Gym Demo](#openai-gym-demo)
* [Deep Q-Learning](#ii-deep-q-learning)
  * [FrozenLake-v1](https://github.com/kakadeniranjan1999/Reinforcement-Learning-With-OpenAI-Gym/tree/FrozenLakeDQN)
* [Change Logs](#change-logs)


## I. Getting Started

### System Requirements
* Linux or Mac OS
* Python 3.6 or above

### Installation and Setup
* Install dependencies using [requirements.txt](requirements.txt)
```
    pip install -r requirements.txt
```

### OpenAI Gym Demo
```
    python3 IntroToOpenAIGym.py <environment_id>
```
Note: Get ```<environment_id>``` of your choice from [OpenAI Gym Wiki](https://github.com/openai/gym/wiki/Table-of-environments)

## II. Deep Q-Learning
[Deep Q-Learning](https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/) is an algorithm that combines [Q-Learning](https://www.freecodecamp.org/news/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe/) and Deep Neural Networks to train an agent to execute appropriate actions in an environment.

This repository provides pre-trained agents (models) as well as easy-to-use customised code for training and inferring an agent for the following environments from OpenAI Gym:
* [FrozenLake-v1](https://github.com/kakadeniranjan1999/Reinforcement-Learning-With-OpenAI-Gym/tree/FrozenLakeDQN)

## Change Logs
* Implementation of OpenAI Gym introduction code
* Deep Q-Learning implemented for [FrozenLake-v1](https://github.com/kakadeniranjan1999/Reinforcement-Learning-With-OpenAI-Gym/tree/FrozenLakeDQN)