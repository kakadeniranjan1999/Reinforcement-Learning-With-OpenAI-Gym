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
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
import warnings
import os
from config_reader import read_config

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

start_timestamp = datetime.datetime.now().strftime("%d-%h-%Y_%H-%M-%S")

logging.basicConfig(format='%(asctime)s :: %(levelname)s :: Line --> %(lineno)d :: %(message)s', level=logging.INFO,
                    filename='logs/FrozenLakeDQN/GeneralLog_' + start_timestamp + '.log', filemode='w')
data_logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: Line --> %(lineno)d :: %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
data_logger.addHandler(handler)


class Agent:
    def __init__(self, current_state, action_space, state_size, action_size, agent_configs, model_configs):
        """
        Deep Q Learning Agent class
        :param current_state: Current state of the environment
        :param action_space: Action space related to the environment
        :param state_size: Size of the environment state
        :param action_size: Size of the action space
        :param agent_configs: User defined agent configurations
        :param model_configs: User defined model configurations
        """
        self.action_size = action_size
        self.state_size = state_size
        self.action_space = action_space
        self.next_state = None
        self.current_state = np.zeros((1, self.state_size))
        self.current_state[0][current_state] = 1

        self.neuron_units = model_configs['neuron_units']
        self.activations = model_configs['activations']
        self.loss = model_configs['loss']
        self.kernel_initializers = model_configs['kernel_initializers']
        self.learning_rate = model_configs['learning_rate']
        self.model_checkpoint = model_configs['model_checkpoint']
        self.history = History()

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

        # Log basic agent related info to the log files
        self.log_basic_info()

        # Build and compile master and slave models
        self.slave_model = self.build_nn_model('SLAVE')
        self.master_model = self.build_nn_model('MASTER')

    def log_basic_info(self):
        """
        Logs basic Agent and Neural Network model related info to the log files
        :return:
        """
        data_logger.info('Action Size --> {}'.format(self.action_size))
        data_logger.info('State Size --> {}'.format(self.state_size))
        data_logger.info('Action Space --> {}'.format(self.action_space))
        data_logger.info('Activation Function --> {}  # [layer1, layer2, output_layer]'.format(self.activations))
        data_logger.info('Neuron Units --> {}  # [hidden layer1, hidden layer2]'.format(self.neuron_units))
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
        """
        Builds and compiles neural network model
        :param name: Name of the model
        :return:
        """
        model = Sequential(name=name)
        model.add(Dense(self.neuron_units[0], input_dim=self.state_size, activation=self.activations[0],
                        kernel_initializer=self.kernel_initializers[0]))
        model.add(Dense(self.neuron_units[1], activation=self.activations[1],
                        kernel_initializer=self.kernel_initializers[1]))
        model.add(Dense(self.action_size, activation=self.activations[2],
                        kernel_initializer=self.kernel_initializers[2]))
        model.summary(print_fn=data_logger.info)
        model.compile(loss=self.loss, optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Updates master NN model with weights of the slave NN model
        :return:
        """
        self.master_model.set_weights(self.slave_model.get_weights())

    def get_action(self):
        """
        Computes executable action for the agent considering epsilon-greedy factor
        :return: executable action
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.slave_model.predict(self.current_state)
            return np.argmax(q_value[0])

    def train(self):
        """
        Trains the model on user defined batch size
        :return:
        """
        # Validate training data array size
        if len(self.training_data) < self.train_start_threshold:
            return

        # Sample random batch sized data entries
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

        # predict q-values for sampled batch sized data
        target = self.slave_model.predict(update_input)
        target_val = self.master_model.predict(update_target)

        # Update target q-values
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + (self.gamma * np.amax(target_val[i]))

        # Fit model
        self.slave_model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0,
                             callbacks=[self.history])


class Environment:
    def __init__(self, env_configs):
        """
        Environment class that contains customized environment entities
        :param env_configs: User defined environment configurations
        """
        self.environment_name = env_configs['env_name']
        self.custom_map_flag = env_configs['custom_map_flag']
        if self.custom_map_flag:
            self.custom_map = env_configs['custom_map']
        else:
            self.custom_map = None
        self.env = gym.make(self.environment_name, desc=self.custom_map)
        self.state = self.env.reset()
        self.action_space = self.env.action_space
        self.render = env_configs['render_flag']

        self.positive_step_reward = env_configs['rewards']['positive_step_reward']
        self.negative_step_reward = env_configs['rewards']['negative_step_reward']
        self.goal_step_reward = env_configs['rewards']['goal_step_reward']

        self.log_basic_info()

    def log_basic_info(self):
        """
        Logs basic Environment related info to the log files
        :return:
        """
        data_logger.info('Environment --> {}'.format(self.environment_name))
        data_logger.info('Custom Environment Map Flag --> {}'.format(self.custom_map_flag))
        data_logger.info('Environment Map --> {}'.format(self.custom_map if self.custom_map_flag else ['SFFF',
                                                                                                       'FHFH',
                                                                                                       'FFFH',
                                                                                                       'HFFG'
                                                                                                       ]))
        data_logger.info('Positive Step Reward --> {}'.format(self.positive_step_reward))
        data_logger.info('Negative Step Reward --> {}'.format(self.negative_step_reward))
        data_logger.info('Goal Step Reward --> {}'.format(self.goal_step_reward))

    def get_reward(self, reward, done, total_success):
        """
        Assigns customized rewards to the executed actions
        :param reward: Reward provided by the environment
        :param done: Boolean flag indicating episode termination status
        :param total_success: Current number of successful goal chases
        :return:
        """
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
    try:
        # Get configuration for environment, agent and neural network model
        config_data = read_config('config/FrozenLakeDQN.yml', data_logger)

        os.rename('logs/FrozenLakeDQN/GeneralLog_' + start_timestamp + '.log', 'logs/FrozenLakeDQN/' +
                  config_data['ENVIRONMENT']['env_name'] + '_' + start_timestamp + '.log')

        # Instantiate environment
        environment = Environment(env_configs=config_data['ENVIRONMENT'])

        # Instantiate agent
        agent = Agent(environment.state, environment.action_space, environment.env.observation_space.n,
                      environment.action_space.n, config_data['AGENT'], config_data['NN_MODEL'])

        # Data loggers for plotting training progress
        plot_data = {'EPISODES': [0.0],
                     'TOTAL_REWARD': [agent.total_reward],
                     'TOTAL_SUCCESS': [agent.total_success],
                     'EPSILON': [agent.epsilon]
                     }

        # Create a subplot with 2 rows and columns each
        figure, axis = plt.subplots(2, 2, figsize=(20, 20))
        axis[0, 0].set_title("Episodes vs Epsilon")
        axis[0, 1].set_title("Episodes vs Total Reward")
        axis[1, 0].set_title("Episodes vs Total Success")
        axis[1, 1].set_title("Episodes vs Avg Loss")

        # Array of average loss after termination of each episode
        avg_loss = np.array([])
        loss_episodes = np.array([])

        for episode in range(agent.total_episodes):
            # Reset environment at the beginning of the episode
            agent.current_state = np.zeros((1, agent.state_size))
            agent.current_state[0][environment.env.reset()] = 1
            done = False

            while not done:
                # get action for the agent
                action = agent.get_action()

                # perform action in the environment
                next_state, reward, done, info = environment.env.step(action)

                agent.next_state = np.zeros((1, agent.state_size))
                agent.next_state[0][next_state] = 1

                # get customized reward for the action
                reward, agent.total_success = environment.get_reward(reward, done, agent.total_success)

                # append data to training array
                agent.training_data.append((agent.current_state, action, reward, agent.next_state, done))

                agent.train()
                agent.current_state = agent.next_state
                agent.total_reward += reward

                if environment.render:
                    environment.env.render()

                if done:
                    # update epsilon-greedy factor
                    agent.epsilon = agent.epsilon * 0.99

                    # append training progress to array
                    plot_data['EPSILON'].append(agent.epsilon)
                    plot_data['TOTAL_REWARD'].append(agent.total_reward)
                    plot_data['TOTAL_SUCCESS'].append(agent.total_success)
                    plot_data['EPISODES'].append(episode)

                    # plot training progress
                    axis[0, 0].plot(plot_data['EPISODES'], plot_data['EPSILON'], 'y')
                    axis[0, 1].plot(plot_data['EPISODES'], plot_data['TOTAL_REWARD'], 'b')
                    axis[1, 0].plot(plot_data['EPISODES'], plot_data['TOTAL_SUCCESS'], 'g')

                    # log data to log files
                    try:
                        avg_loss = np.append(avg_loss, np.average(agent.history.history['loss']))
                        loss_episodes = np.append(loss_episodes, episode)
                        axis[1, 1].plot(loss_episodes, avg_loss, 'r')

                        data_logger.info("episode:" + str(episode) + "  total reward:" + str(agent.total_reward) +
                                         "  train data length:" + str(len(agent.training_data)) +
                                         "  total success:" + str(agent.total_success) + "  current reward:" + str(
                            reward) +
                                         "  epsilon:" + str(agent.epsilon) +
                                         "  Avg Loss:" + str(avg_loss[-1]))
                    except KeyError:
                        data_logger.info("episode:" + str(episode) + "  total reward:" + str(agent.total_reward) +
                                         "  train data length:" + str(len(agent.training_data)) +
                                         "  total success:" + str(agent.total_success) + "  current reward:" + str(
                            reward) +
                                         "  epsilon:" + str(agent.epsilon))

                    # save graph plot
                    plt.tight_layout()
                    plt.savefig("InferenceData/FrozenLakeDQN/" + config_data['ENVIRONMENT'][
                        'env_name'] + '_' + start_timestamp + ".png")

                    # if np.mean(plot_data['TOTAL_REWARD'][-min(10, len(plot_data['TOTAL_REWARD'])):]) > 5000:
                    #     sys.exit()

                # save model after every checkpoint episode provided by user
                if episode % agent.model_checkpoint == 0:
                    agent.slave_model.save(
                        "trained_models/FrozenLakeDQN/" + config_data['ENVIRONMENT']['env_name'] + '_' +
                        start_timestamp + ".h5")
    except KeyboardInterrupt:
        data_logger.warning('KeyBoard Interrupt')
    except Exception as e:
        data_logger.error('Error at LINE --> {} --> {}'.format(sys.exc_info()[2].tb_lineno, e))
