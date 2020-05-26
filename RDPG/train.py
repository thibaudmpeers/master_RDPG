import numpy as np
import torch
import RDPG.utils as utils
import RDPG.model as model
from RDPG.TBTT import TBTT_critic, TBTT_actor

start_input_2_learning = 500


def get_nn_arch_params():
    return model.return_nn_arch_params()


class Trainer:

    def __init__(self, observation_dim, action_dim, env, number_steps, memory, device, l_r_actor=1e-5, l_r_critic=1e-5,
                 tau=1e-5, decay_tau=0, extended_inputs=True, tensorboard=None):

        self.env = env

        self.batch_size = 16
        self.number_steps = number_steps

        self.gamma = 0.99
        self.init_tau = tau
        self.tau = self.init_tau
        self.decay_factor_tau = decay_tau
        self.device = device

        self.learning_rate_actor = l_r_actor
        self.learning_rate_critic = l_r_critic

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.memory = memory
        self.iter = 0
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = model.Actor(self.observation_dim, self.action_dim, extended_inputs, device).to(device)
        self.target_actor = model.Actor(self.observation_dim, self.action_dim, extended_inputs, device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), l_r_actor)

        self.critic = model.Critic(self.observation_dim, self.action_dim, extended_inputs, device).to(device)
        self.target_critic = model.Critic(self.observation_dim, self.action_dim, extended_inputs, device).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), l_r_critic)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

        truncated_param = 10

        self.TBTT_critic = TBTT_critic(self.critic, truncated_param, self.critic_optimizer, number_steps, self.gamma, tensorboard)
        self.TBTT_actor = TBTT_actor(self.actor, truncated_param, self.actor_optimizer, number_steps, self.critic, tensorboard)

        self.extended_inputs = extended_inputs

        self.tensorboard = tensorboard

    def get_exploitation_action(self, inputs, h_actor):
        action, new_h_actor = self.actor.forward(inputs, h_actor)
        return action.detach(), new_h_actor

    def get_exploration_action(self, inputs, h_actor):
        std = 2
        action, new_h_actor = self.actor.forward(inputs, h_actor)
        new_action = action.detach() + (torch.tensor(self.noise.sample()).float().to(self.device) * std)
        return new_action, new_h_actor

    def run_episode(self, save_traj, to_pickle, exploitation, episode_count):

        # if episode_count == start_input_2_learning:
        #     self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate_actor/10)
        #     self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate_critic/10)

        observation, initial_state, new_delay = self.env.reset(get_init_state=True, save_traj=save_traj)
        to_pickle['initial_states'].append(initial_state)
        to_pickle['delay'].append(new_delay)

        sum_rewards = 0

        previous_action = torch.zeros(1, self.action_dim, dtype=torch.float32).to(self.device)
        previous_reward = torch.zeros(1, 1, dtype=torch.float32).to(self.device)

        obs = torch.from_numpy(np.float32(observation).reshape(1, self.observation_dim)).to(self.device)

        if self.extended_inputs == 0:
            inputs = obs
        elif self.extended_inputs == 1:
            inputs = torch.cat((obs, previous_action), dim=1)
        else:
            inputs = torch.cat((obs, previous_action, previous_reward), dim=1)

        h_target_critic = self.target_critic.init_hidden_state(1)
        h_actor = self.actor.init_hidden_state(1)
        h_target_actor = self.target_actor.init_hidden_state(1)

        target_action, h_target_actor = self.target_actor(inputs, h_target_actor)
        _, h_target_critic = self.target_critic(inputs, target_action, h_target_critic)

        for step in range(self.number_steps):

            with torch.no_grad():
                if exploitation:
                    action, h_actor = self.get_exploitation_action(inputs, h_actor)
                else:
                    action, h_actor = self.get_exploration_action(inputs, h_actor)

            action_cpu = action.cpu().numpy()

            new_observation, reward, done, info = self.env.step(action_cpu, step)
            # env.render()
            sum_rewards += reward

            reward = torch.from_numpy(reward.astype(np.float32).reshape(1, 1)).to(self.device)

            self.memory.save_trasition(obs, action, reward, inputs, step)

            obs = torch.from_numpy(np.float32(new_observation).reshape(1, self.observation_dim)).to(self.device)
            if episode_count >= start_input_2_learning:
                previous_action = action
                previous_reward = reward

            if self.extended_inputs == 0:
                inputs = obs
            elif self.extended_inputs == 1:
                inputs = torch.cat((obs, previous_action), dim=1)
            else:
                inputs = torch.cat((obs, previous_action, previous_reward), dim=1)

            with torch.no_grad():
                target_action, h_target_actor = self.target_actor(inputs, h_target_actor)
                target_value, h_target_critic = self.target_critic(inputs, target_action, h_target_critic)
            self.memory.save_target_value(target_value, step)

        return sum_rewards, info, new_delay

    def optimize(self):
        batch_history = self.memory.sample()

        actor_loss = self.TBTT_actor.train(batch_history)
        critic_loss, mean_errors, mean_relative_errors = self.TBTT_critic.train(batch_history)

        utils.soft_update(self.target_actor, self.actor, self.tau)
        utils.soft_update(self.target_critic, self.critic, self.tau)

        return critic_loss, actor_loss, mean_errors, mean_relative_errors

    def save_models(self, episode_count, path):
        torch.save(self.target_actor.state_dict(), path + 'Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), path + 'Models/' + str(episode_count) + '_critic.pt')
        print('Models saved successfully')

    def load_models(self, episode, path_folder):
        self.actor.load_state_dict(torch.load(path_folder + 'Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(path_folder + 'Models/' + str(episode) + '_critic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        print('Models loaded successfully')

    def return_params(self):
        return [self.batch_size, self.gamma, self.init_tau, self.decay_factor_tau]

    def get_models_params(self):
        nb_actor_params = 0
        for parameter in self.actor.parameters():
            nb_params_layer = 1
            for nb_nn in list(parameter.size()):
                nb_params_layer = nb_params_layer * nb_nn
            nb_actor_params += nb_params_layer

        nb_critic_params = 0
        for parameter in self.critic.parameters():
            nb_params_layer = 1
            for nb_nn in list(parameter.size()):
                nb_params_layer = nb_params_layer * nb_nn
            nb_critic_params += nb_params_layer
        return nb_actor_params, nb_critic_params

    def update_tau(self, episode_number):
        self.tau = self.init_tau/(1 + self.decay_factor_tau*episode_number)
