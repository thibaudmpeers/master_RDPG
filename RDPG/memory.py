import random
import torch


class Memory:

    def __init__(self, size, number_steps, observation_dim, action_dim, extended_inputs, device):
        self.max_size = size
        self.number_steps = number_steps
        self.obs_dim = observation_dim
        self.action_dim = action_dim
        self.index = self.max_size
        self.len = 0
        self.batch_size = 16
        self.device = device

        self.actions_hist = torch.zeros((self.max_size, self.number_steps, self.action_dim), dtype=torch.float32).to(self.device)
        self.rewards_hist = torch.zeros((self.max_size, self.number_steps, 1), dtype=torch.float32).to(self.device)
        self.observations_hist = torch.zeros((self.max_size, self.number_steps, self.obs_dim), dtype=torch.float32).to(self.device)
        self.target_values_hist = torch.zeros((self.max_size, self.number_steps, 1), dtype=torch.float32).to(self.device)

        if extended_inputs == 0:
            self.inputs_hist = torch.zeros((self.max_size, self.number_steps, self.obs_dim), dtype=torch.float32).to(self.device)
        elif extended_inputs == 1:
            self.inputs_hist = torch.zeros((self.max_size, self.number_steps, self.obs_dim + self.action_dim),
                                           dtype=torch.float32).to(self.device)
        else:
            self.inputs_hist = torch.zeros((self.max_size, self.number_steps, self.obs_dim + self.action_dim + 1),
                                           dtype=torch.float32).to(self.device)

    def new_epsisode(self):
        if self.len != self.max_size:
            self.len += 1
        if self.index < self.max_size - 1:
            self.index += 1
        else:
            self.index = 0

    def save_trasition(self, observation, action, reward, inputs, step):
        self.observations_hist[self.index, step, :] = observation.detach()
        self.actions_hist[self.index, step, :] = action.detach()
        self.rewards_hist[self.index, step, :] = reward.detach()
        self.inputs_hist[self.index, step, :] = inputs.detach()

    def save_target_value(self, target_value, step):
        self.target_values_hist[self.index, step, :] = target_value.detach()

    def sample(self):
        count = min(self.len, self.batch_size)
        batch = random.sample(range(self.len), count)

        s_arr = self.observations_hist[batch]
        a_arr = self.actions_hist[batch]
        r_arr = self.rewards_hist[batch]
        inputs_arr = self.inputs_hist[batch]
        target_values = self.target_values_hist[batch]

        return s_arr, a_arr, r_arr, inputs_arr, target_values, count

    def len(self):
        return self.index
