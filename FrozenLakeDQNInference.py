import os
import sys
import time
import gym
import warnings
import datetime

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from get_logger import get_logger

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

start_timestamp = datetime.datetime.now().strftime("%d-%h-%Y_%H-%M-%S")
log_filename = 'Inference/InferenceLogs/FrozenLakeDQN/GeneralLog_' + start_timestamp + '.log'
console_output_format = '%(asctime)s :: %(levelname)s :: Line --> %(lineno)d :: %(message)s'

data_logger = get_logger(console_output_format, log_filename)


if __name__ == "__main__":
    try:
        model_path = sys.argv[1]

        data_logger.info('Inference Game with Model {}'.format(model_path.split('/')[-1]))
        data_logger.info('--------------------------------------------------------------------------------------------------------------------------------')
        os.rename(log_filename, 'Inference/InferenceLogs/FrozenLakeDQN/FrozenLake-v1_' + start_timestamp + '.log')

        env = gym.make('FrozenLake-v1')
        state = env.reset()

        # load pre-trained model
        model = load_model(model_path)

        episodes = 100
        total_reward = 0.0
        plot_data = {'EPISODES': [],
                     'TOTAL_REWARD': []
                     }
        action_desc = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'}

        figure, axis = plt.subplots(2, 1, figsize=(20, 20))
        axis[0].set_title("Episodes vs Total Success")
        axis[1].set_title("Model Accuracy")

        for e in range(episodes):
            done = False
            state = env.reset()

            path_followed = []
            action_executed = []

            while not done:
                env.render()

                path_followed.append(state)

                arr = np.zeros((1, 16))
                arr[0][state] = 1
                action = np.argmax(model.predict(np.array([arr])))

                action_executed.append(action)

                state, reward, done, info = env.step(action)

                total_reward += reward

                # time.sleep(0.05)
                if e == 99 and done:
                    pass
                else:
                    os.system('clear')

                if done:
                    # append inference data to array
                    plot_data['EPISODES'].append(e)
                    plot_data['TOTAL_REWARD'].append(total_reward)

                    # plot inference data
                    axis[0].plot(plot_data['EPISODES'], plot_data['TOTAL_REWARD'], '^-g')

                    # save graph plot
                    plt.savefig('Inference/InferencePlots/FrozenLakeDQN/FrozenLake-v1_' + start_timestamp + '.png')

            # log data to log files
            data_logger.info("episode:{}".format(e + 1) +
                             "  total reward:{}".format(total_reward) +
                             "  Path Followed:{}".format(path_followed) +
                             "  Action Executed:{}".format(action_executed))
        axis[1].pie(x=[total_reward, episodes - total_reward], explode=[0.2, 0.0], autopct='%1.0f%%', pctdistance=0.6,
                    labeldistance=1.2, radius=0.8, shadow=True)
        axis[1].legend(['Successful Goal Chase', 'Failed Goal Chase'], loc=1)

        plt.tight_layout()
        plt.savefig('Inference/InferencePlots/FrozenLakeDQN/FrozenLake-v1_' + start_timestamp + '.png')

        data_logger.info("episode:{}".format(episodes) +
                         "  total reward:{}".format(total_reward) +
                         "  Model Accuracy:{}".format((total_reward / episodes) * 100))
    except KeyboardInterrupt:
        data_logger.warning('Keyboard Interrupt')
    except IndexError:
        data_logger.error('Error at LINE --> {} --> {}'.format(sys.exc_info()[2].tb_lineno,
                                                               'Require Model Path as an Argument!!!'))
    except Exception as e:
        data_logger.error('Error at LINE --> {} --> {}'.format(sys.exc_info()[2].tb_lineno, e))
