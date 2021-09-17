import sys
import gym
import logging


class RawAgent:
    def __init__(self, current_state, action_space):
        self.action_space = action_space
        self.current_state = current_state

    def get_action(self):
        pole_inclination = self.current_state[2]
        if pole_inclination > 0:
            return 1
        else:
            return 0


class Environment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state = self.env.reset()
        self.action_space = self.env.action_space


if __name__ == "__main__":
    try:
        environment = Environment(sys.argv[1])
        agent = RawAgent(environment.state, environment.action_space)
        for _ in range(300):
            environment.env.render()
            action = agent.get_action()
            agent.current_state, reward, done, info = environment.env.step(action)
            if done:
                environment.env.reset()
        environment.env.close()
    except KeyboardInterrupt:
        logging.warning('Keyboard Interrupt')
    except Exception as e:
        logging.error('Error at LINE --> {} --> {}'.format(sys.exc_info()[2].tb_lineno, e))
