# TODO number layers ? + action on recurrent layer ?
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

size_layer = 128


def weight_init(size):
    random_params_layer = np.sqrt(2./(size[0] + size[1]))
    return torch.randn(size)*random_params_layer


def return_nn_arch_params():
    return 3, size_layer


class Critic(nn.Module):

    def __init__(self, observation_dim, action_dim, extended_inputs, device):
        super(Critic, self).__init__()

        self.device = device
        self.size_rnn_layer = size_layer
        size_layer1 = size_layer
        size_layer2 = size_layer

        if extended_inputs == 0:
            self.rnn_layer = nn.GRUCell(observation_dim, self.size_rnn_layer).to(self.device)
        elif extended_inputs == 1:
            self.rnn_layer = nn.GRUCell(observation_dim + action_dim, self.size_rnn_layer).to(self.device)
        else:
            self.rnn_layer = nn.GRUCell(observation_dim + action_dim + 1, self.size_rnn_layer).to(self.device)
            self.rnn_layer.weight_ih.data[:, -action_dim - 1:] = torch.zeros((3 * self.size_rnn_layer, action_dim + 1))

        self.hidden_layer1 = nn.Linear(self.size_rnn_layer + action_dim, size_layer1)
        self.hidden_layer1.weight.data = weight_init(self.hidden_layer1.weight.data.size())

        self.hidden_layer2 = nn.Linear(size_layer1, size_layer2)
        self.hidden_layer2.weight.data = weight_init(self.hidden_layer2.weight.data.size())

        self.output_layer = nn.Linear(size_layer2, 1)
        self.output_layer.weight.data = weight_init(self.output_layer.weight.data.size())

    def forward(self, inputs, action, h_critic):
        new_h_critic = self.rnn_layer(inputs, h_critic)

        x = torch.cat((new_h_critic, action), 1)

        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        return self.output_layer(x), new_h_critic

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.size_rnn_layer).to(self.device)


class Actor(nn.Module):

    def __init__(self, observation_dim, action_dim, extended_inputs, device):
        super(Actor, self).__init__()

        self.device = device
        self.size_rnn_layer = size_layer
        size_layer1 = size_layer
        size_layer2 = size_layer

        if extended_inputs == 0:
            self.rnn_layer = nn.GRUCell(observation_dim, self.size_rnn_layer)
        elif extended_inputs == 1:
            self.rnn_layer = nn.GRUCell(observation_dim + action_dim, self.size_rnn_layer)
        else:
            self.rnn_layer = nn.GRUCell(observation_dim + action_dim + 1, self.size_rnn_layer)
            self.rnn_layer.weight_ih.data[:, -action_dim - 1:] = torch.zeros((3 * self.size_rnn_layer, action_dim + 1))

        self.hidden_layer1 = nn.Linear(self.size_rnn_layer, size_layer1)
        self.hidden_layer1.weight.data = weight_init(self.hidden_layer1.weight.data.size())

        self.hidden_layer2 = nn.Linear(size_layer1, size_layer2)
        self.hidden_layer2.weight.data = weight_init(self.hidden_layer2.weight.data.size())

        self.output_layer = nn.Linear(size_layer2, action_dim)
        self.output_layer.weight.data = weight_init(self.output_layer.weight.data.size())

    def forward(self, inputs, h_actor):
        new_h_actor = self.rnn_layer(inputs, h_actor)

        x = F.relu(self.hidden_layer1(new_h_actor))
        x = F.relu(self.hidden_layer2(x))
        return self.output_layer(x), new_h_actor

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.size_rnn_layer).to(self.device)
