import random
import time
import os
import numpy as np
import gym
from gym.envs.registration import register


class Agent:
    def __init__(self, current_state, action_space, state_size, action_size):
        self.action_size = action_size
        self.state_size = state_size
        self.action_space = action_space
        self.next_state = None
        self.current_state = current_state
        self.gamma = 0.97
        self.lr = 0.01
        self.epsilon = 1.0

        self.q_table = 1e-4 * np.random.random([self.state_size, self.action_size])

    def get_action(self):
        state_q_values = self.q_table[self.current_state]
        q_max_action = np.argmax(state_q_values)
        random_action = random.choice(range(self.action_size))
        return random_action if random.random() < self.epsilon else q_max_action

    def train(self, reward, done, action):
        q_next = self.q_table[self.next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.gamma * np.max(q_next)
        q_delta = q_target - self.q_table[self.current_state, action]
        self.q_table[self.current_state, action] += self.lr * q_delta

        if done:
            self.epsilon = self.epsilon * 0.99


class Environment:
    def __init__(self):
        # self.env = gym.make('FrozenLakeNoSlip-v0')
        self.env = gym.make('FrozenLake-v1')
        self.state = self.env.reset()
        self.action_space = self.env.action_space


# try:
#     register(
#         id='FrozenLakeNoSlip-v0',
#         entry_point='gym.envs.toy_text:FrozenLakeEnv',
#         kwargs={'map_name': '4x4', 'is_slippery': False},
#         max_episode_steps=100,
#         reward_threshold=0.78,  # optimum = .8196
#     )
# except:
#     pass


environment = Environment()
agent = Agent(environment.state, environment.action_space, environment.env.observation_space.n, environment.action_space.n)

total_reward = 0.0
for i in range(5):
    for episode in range(100):
        agent.current_state = environment.env.reset()
        done = False
        while not done:
            action = agent.get_action()
            agent.next_state, reward, done, info = environment.env.step(action)
            agent.train(reward, done, action)
            agent.current_state = agent.next_state
            total_reward += reward

            print('Iteration: {}'.format(i))
            print("State:", agent.current_state, "Action:", action)
            print("Episode: {}, Total reward: {}, Epsilon: {}".format(episode, total_reward, agent.epsilon))
            environment.env.render()
            # time.sleep(0.05)
            if episode == 99 and done:
                pass
            else:
                os.system('clear')
print(agent.q_table)
