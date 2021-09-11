import datetime
import sys
import logging
import gym
import random
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import matplotlib.pylab as pylab
import warnings
import os
from config_reader import read_config

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

start_timestamp = datetime.datetime.now().strftime("%d-%h-%Y_%H-%M-%S")

logging.basicConfig(format='%(asctime)s :: %(levelname)s :: Line --> %(lineno)d :: %(message)s', level=logging.INFO,
                    filename='logs/FrozenLakeDQN/FrozenLakeDQN_' + start_timestamp + '.log', filemode='w')
data_logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: Line --> %(lineno)d :: %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
data_logger.addHandler(handler)


class Agent:
    def __init__(self, current_state, action_space, state_size, action_size, agent_configs, model_configs):
        self.action_size = action_size
        self.state_size = state_size
        self.action_space = action_space
        self.next_state = None
        self.current_state = np.zeros((1, self.state_size))
        self.current_state[0][current_state] = 1

        self.activations = model_configs['activations']
        self.loss = model_configs['loss']
        self.kernel_initializers = model_configs['kernel_initializers']
        self.learning_rate = model_configs['learning_rate']
        self.model_checkpoint = model_configs['model_checkpoint']

        self.total_episodes = agent_configs['total_episodes']
        self.gamma = agent_configs['gamma']
        self.epsilon = agent_configs['epsilon']
        self.min_epsilon = agent_configs['min_epsilon']

        self.batch_size = model_configs['batch_size']
        self.training_data_deque_max_len = agent_configs['training_data_deque_max_len']
        self.training_data = deque(maxlen=self.training_data_deque_max_len)
        self.train_start_threshold = agent_configs['train_start_threshold']

        self.total_reward = 0.0
        self.total_success = 0.0

        self.log_basic_info()

        self.model = self.build_nn_model('SLAVE')
        self.target_model = self.build_nn_model('MASTER')

    def log_basic_info(self):
        data_logger.info('Action Size --> {}'.format(self.action_size))
        data_logger.info('State Size --> {}'.format(self.state_size))
        data_logger.info('Action Space --> {}'.format(self.action_space))
        data_logger.info('Activation Function --> {}'.format(self.activations))
        data_logger.info('Loss Function --> {}'.format(self.loss))
        data_logger.info('Kernel Initializers --> {}'.format(self.kernel_initializers))
        data_logger.info('Discount Factor --> {}'.format(self.gamma))
        data_logger.info('Learning Rate --> {}'.format(self.learning_rate))
        data_logger.info('Epsilon-Greedy Factor --> {}'.format(self.epsilon))
        data_logger.info('Minimum Epsilon-Greedy Factor --> {}'.format(self.min_epsilon))
        data_logger.info('Training Data Array Deque Length --> {}'.format(self.training_data_deque_max_len))
        data_logger.info('Training Batch Size --> {}'.format(self.batch_size))
        data_logger.info('Training Start Threshold --> {}'.format(self.train_start_threshold))
        data_logger.info('Model Checkpoint --> {}'.format(self.model_checkpoint))

    def build_nn_model(self, name):
        model = Sequential(name=name)
        model.add(Dense(32, input_dim=self.state_size, activation=self.activations[0],
                        kernel_initializer=self.kernel_initializers[0]))
        model.add(Dense(64, activation=self.activations[1],
                        kernel_initializer=self.kernel_initializers[1]))
        model.add(Dense(self.action_size, activation=self.activations[2],
                        kernel_initializer=self.kernel_initializers[2]))
        model.summary(print_fn=data_logger.info)
        model.compile(loss=self.loss, optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(self.current_state)
            return np.argmax(q_value[0])

    def train(self):
        if len(self.training_data) < self.train_start_threshold:
            return

        train_batch = random.sample(self.training_data, self.batch_size)

        update_input = np.zeros((self.batch_size, self.state_size))
        update_target = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = train_batch[i][0]
            action.append(train_batch[i][1])
            reward.append(train_batch[i][2])
            update_target[i] = train_batch[i][3]
            done.append(train_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + (self.gamma * np.amax(target_val[i]))

        self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)


class Environment:
    def __init__(self, env_configs):
        self.environment_name = env_configs['env_name']
        self.env = gym.make(env_configs['env_name'])
        self.state = self.env.reset()
        self.action_space = self.env.action_space
        self.render = env_configs['render_flag']

        self.positive_step_reward = env_configs['rewards']['positive_step_reward']
        self.negative_step_reward = env_configs['rewards']['negative_step_reward']
        self.goal_step_reward = env_configs['rewards']['goal_step_reward']

        self.log_basic_info()

    def log_basic_info(self):
        data_logger.info('Environment --> {}'.format(self.environment_name))
        data_logger.info('Positive Step Reward --> {}'.format(self.positive_step_reward))
        data_logger.info('Negative Step Reward --> {}'.format(self.negative_step_reward))
        data_logger.info('Goal Step Reward --> {}'.format(self.goal_step_reward))

    def get_reward(self, reward, done, total_success):
        if not done and reward == 0:
            reward = self.positive_step_reward
        elif done and reward == 0:
            reward = self.negative_step_reward
        elif done and reward == 1:
            reward = self.goal_step_reward
            total_success += 1
        else:
            data_logger.error('Unexpected Condition --> done = False and reward = 1.0')

        return reward, total_success


if __name__ == "__main__":
    config_data = read_config('config/FrozenLakeDQN.yml', data_logger)
    plot_data = {'EPISODES': [], 'TOTAL_REWARD': [], 'TOTAL_SUCCESS': []}

    environment = Environment(env_configs=config_data['ENVIRONMENT'])
    agent = Agent(environment.state, environment.action_space, environment.env.observation_space.n,
                  environment.action_space.n, config_data['AGENT'], config_data['NN_MODEL'])

    for episode in range(agent.total_episodes):
        agent.current_state = np.zeros((1, agent.state_size))
        agent.current_state[0][environment.env.reset()] = 1
        done = False

        while not done:
            action = agent.get_action()
            next_state, reward, done, info = environment.env.step(action)

            agent.next_state = np.zeros((1, agent.state_size))
            agent.next_state[0][next_state] = 1

            reward, agent.total_success = environment.get_reward(reward, done, agent.total_success)

            agent.training_data.append((agent.current_state, action, reward, agent.next_state, done))
            if done:
                agent.epsilon = agent.epsilon * 0.99
            agent.train()
            agent.current_state = agent.next_state
            agent.total_reward += reward

            if environment.render:
                environment.env.render()

            if done:
                plot_data['TOTAL_REWARD'].append(agent.total_reward)
                plot_data['TOTAL_SUCCESS'].append(agent.total_success)
                plot_data['EPISODES'].append(episode)

                pylab.plot(plot_data['EPISODES'], plot_data['TOTAL_REWARD'], 'b')
                pylab.plot(plot_data['EPISODES'], plot_data['TOTAL_SUCCESS'], 'r')
                pylab.savefig("InferenceData/FrozenLakeDQN/InferenceGraph_" + start_timestamp + ".png")

                data_logger.info("episode:" + str(episode) + "  total reward:" + str(agent.total_reward) + "  memory length:" +
                                 str(len(agent.training_data)) + "  total success:" + str(agent.total_success) +
                                 "  current reward:" + str(reward) +
                                 "  epsilon:" + str(agent.epsilon))

                # if np.mean(plot_data['TOTAL_REWARD'][-min(10, len(plot_data['TOTAL_REWARD'])):]) > 490:
                #     sys.exit()

            if episode % agent.model_checkpoint == 0:
                agent.model.save("trained_models/FrozenLakeDQN/FrozenLakeDQN_" + start_timestamp + ".h5")
