import gym


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
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.state = self.env.reset()
        self.action_space = self.env.action_space


environment = Environment()
agent = RawAgent(environment.state, environment.action_space)
for _ in range(300):
    action = agent.get_action()
    agent.current_state, reward, done, info = environment.env.step(action)
    environment.env.render()
environment.env.reset()
environment.env.close()
