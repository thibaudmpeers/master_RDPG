import numpy as np


class Gaussian_noise:
    def __init__(self, action_dim, std=0.2):
        self.action_dim = action_dim
        self.std = std

    def sample(self):
        return np.random.randn(self.action_dim) * self.std


class Load_disturbance:
    def __init__(self, nb_steps, max_load):
        self.number_steps = nb_steps
        self.max_load = max_load
        self.start = False
        self.current_step = 0

    def sample(self, start_load=False):
        if start_load:
            self.start = True
        if not self.start:
            return 0
        if self.current_step == self.number_steps:
            self.current_step = 0
            self.start = False
        else:
            self.current_step += 1
        if self.number_steps / 6 < self.current_step < 5 * self.number_steps / 6:
            return self.max_load
        else:
            return self.max_load * np.sin(3 * np.pi * self.current_step / self.number_steps)


class Ornstein_Uhlenbeck:
    def __init__(self, action_dim, mu=0, theta=0.2, sigma=0.10):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx += self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


increase_rate_sigma = 0.01
# x = np.arange(1, 9, 2)
x = np.arange(1, 6, 1)
# x = np.array([5])
mean = 5
start_delay_variation = 500


def get_random_param():
    return x, mean, increase_rate_sigma


class Delay:
    def __init__(self):
        self.count = 0

    def new_delay(self):
        self.count += 1
        if self.count <= start_delay_variation:
            return mean
        else:
            self.count += 1
            sigma = self.count*increase_rate_sigma
            prob = np.exp(-0.5*((x - mean)/sigma)**2)
            return np.random.choice(x, p=prob/prob.sum())


if __name__ == '__main__':
    gauss = True
    noise = Gaussian_noise(1, 0.1) if gauss else Ornstein_Uhlenbeck(1)
    states = []
    for _ in range(1000):
        states.append(noise.sample())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()

    load = Load_disturbance(20, 2.6)
    signal = []
    for i in range(100):
        if i == 36:
            signal.append(load.sample(True))
        else:
            signal.append(load.sample())
    plt.plot(signal)
    plt.show()
