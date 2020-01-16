import gym
from gym.spaces import Discrete, Tuple
import numpy as np
import matplotlib.pyplot as plt

PLOT = False
INTERVAL = 100  # How often to plot results and print to terminal


def tuplize(domain):
    """If the input is not a tuple, make it a tuple"""
    if isinstance(domain, Tuple):
        spaces = domain.spaces
        return tuple([space.n for space in spaces])
    elif isinstance(domain, Discrete):
        return tuple((domain.n,))
    else:
        raise Exception


def sample_from(values, tau):
    """Select an action based on 'tau',
    which is a measure of the uncertainty of the value approximation.
    Also return 'zeta', a measure of randomness."""
    n = values.size  # the number of available actions
    a = np.arange(n)  # the vector representation of available actions
    tau = 1 if np.isnan(tau) else tau  # if tau is not a number, set it equal to 1
    i = 1 + tau  # the base of the exponential
    x = np.asarray([i**j for j in values])  # the weights!
    x += np.random.random_sample(n)  # white noise to prevent stagnation
    x = np.clip(x, 1 / (n ** 2), a_max=None)  # raise minimum weight to prevent stagnation  # use max() instead?
    s = np.sum(x)  # sum of the modified weights
    p = x / s  # weights transformed into probabilites
    zeta = 1 - np.amax(p)  # 'randomness'
    """If no option seems better than any other and that's unlikely to change anytime soon,
    just pick the option that seems best. If we have a good option but we're not sure it's the best,
    take a weighted sample! That way, if there are two good options they have equal chance to be selected."""
    if zeta > tau:
        action = np.argmax(values)
    else:
        action = np.random.choice(a, p=p)
    return action, zeta


class Agent:
    def __init__(self, env, alpha=0.1, style='-b'):
        self.name = "modified Q"
        self.style = style  # for plotting
        self.params = "a: {}".format(alpha)
        self.env = env
        self.alpha = alpha
        self.n_actions = tuplize(env.action_space)
        self.n_states = tuplize(env.observation_space)
        self.table = Table(self.n_actions, self.n_states, alpha)
        self.score = Record("score")
        self.zeta = Record("zeta")
        self.tau = Record("tau")
        self.episode_n = 0

    def act(self, state):
        action, zeta, tau = self.table.act(state)
        self.zeta.step.append(zeta)
        self.tau.step.append(tau)
        if isinstance(self.env.action_space, Discrete):
            action = action[0]
        return action

    def update(self, state, action, result, reward, done):
        self.score.step.append(reward)
        self.table.update(state, action, result, reward, done)

    def reset(self):
        self.episode_n += 1
        self.score.clear_step("sum")
        self.zeta.clear_step("avg")
        self.tau.clear_step("avg")
        if not self.episode_n % INTERVAL:
            self.score.clear_episode(self.episode_n)
            self.zeta.clear_episode(self.episode_n)
            self.tau.clear_episode(self.episode_n)
            print("{}\tEpisode: {}\tScore: {:.3}\tEpsilon: {:.3}\tTau: {:.3}"
                  .format(self.name,
                          self.episode_n,
                          self.score.period["mean"][-1],
                          self.zeta.period["mean"][-1],
                          self.tau.period["mean"][-1]))


class Table:
    def __init__(self, n_actions, n_states, alpha):
        self.n_actions = n_actions
        self.n_states = n_states
        self.alpha = alpha
        shape = (n_states + n_actions)
        self.value = np.random.random_sample(shape)
        self.delta = np.random.random_sample(shape)

    def act(self, state):
        state = state if isinstance(state, tuple) else (state,)
        values = self.value[state].ravel()
        deltas = self.delta[state].ravel()
        tau = np.amax(deltas)
        action, zeta = sample_from(values, tau)
        action = np.unravel_index(action, self.n_actions)
        return action, zeta, tau

    def update(self, state, action, result, reward, done):
        state = state if isinstance(state, tuple) else (state,)
        action = action if isinstance(action, tuple) else (action,)
        result = result if isinstance(result, tuple) else (result,)
        values = self.value[result].ravel()
        deltas = self.delta[result].ravel()
        tau = np.amax(deltas)
        _, zeta = sample_from(values, tau)
        potential = 0 if done else zeta * np.mean(self.value[result]) + (1 - zeta) * np.amax(self.value[result])
        target = reward + potential
        index = state + action
        q = self.value[index]
        self.value[index] += self.alpha * (target - q)
        self.delta[index] = np.abs(1 - (np.e**self.value[index]) / (np.e**q))


class Record:
    def __init__(self, name):
        self.name = name
        self.step = []
        self.episode = []
        self.period = {"episode_n": [], "min": [], "mean": [], "max": []}

    def clear_step(self, operation):
        if operation == "avg":
            self.episode.append(np.mean(self.step))
        elif operation == "sum":
            self.episode.append(np.sum(self.step))
        else:
            raise Exception
        self.step.clear()

    def clear_episode(self, episode_n):
        self.period["episode_n"].append(episode_n)
        self.period["min"].append(np.amin(self.episode))
        self.period["mean"].append(np.mean(self.episode))
        self.period["max"].append(np.amax(self.episode))
        self.episode.clear()
        if PLOT:
            self.plot()

    def plot(self):
        x = self.period["episode_n"]
        keys = ["min", "mean", "max"]
        for key in keys:
            plt.plot(x, self.period[key], label=key)
        plt.xscale('symlog', linthreshy=1)
        plt.yscale('symlog', linthreshy=1)
        plt.gca().xaxis.grid(True, which='minor')
        plt.grid(True)
        plt.title(self.name)
        plt.legend()
        plt.savefig("{}.png".format(self.name))
        plt.close()


if __name__ == "__main__":
    import svrl_demo_1
    env = gym.make("svrl-v3")
    agent = Agent(env)
    while True:
        done = 0
        state = env.reset()
        while not done:
            action = agent.act(state)
            result, reward, done, _ = env.step(action)
            agent.update(state, action, result, reward, 0)
            state = result
        agent.reset()



