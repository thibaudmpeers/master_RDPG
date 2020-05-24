import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from random import sample
from Environment.LoopFollowingAgents import LoopFollowingAgents
from RDPG.train import Trainer
sys.path.append(os.getcwd())


str_N = '6'
obs = '2'
gam = '0.99'
alp = '-5'
tau = '-5'
decay = '0'
name = 'test1'

model_number = 5000
number_steps = 1000

device = torch.device("cpu")

folder = './Saved_results/number_agents_' + str_N
folder += '/observation_' + obs
folder += '/gamma_' + gam
folder += '/2_layers_size_128'
folder += '/learning_rate_10e' + alp + '_tau_10e' + tau
folder += '_decay_tau_' + decay
folder += '/' + name + '/'

with open(folder + 'infos.pickle', 'rb') as file:
    from_pickle = pickle.load(file)

available_obs = from_pickle['available_observations']
obs_u_i = from_pickle['observation_integral_term']
mode_action = from_pickle['mode_action']
type_action = from_pickle['type_action']
nb_agents = from_pickle['number_agents']
delay = from_pickle['delay']
dt = from_pickle['time_step']
d_ref = from_pickle['distance_ref']
enable_u_i = from_pickle['enable_u_i']
gains = from_pickle['gains_reward']
local_time_step = from_pickle['local_time_step']
random_start = from_pickle['random_start']
stacked_frames = from_pickle['stacked_frames']
obs_front = from_pickle['obs_front']
active_agents_train = from_pickle['active_agents']


# dim_obs, action_space

del from_pickle

reward_analysis = True
perturbations = True

act_agents = [0]
load_agent = 0

env = LoopFollowingAgents(available_obs=available_obs, obs_u_i=obs_u_i, mode_action=mode_action,
                          type_action=type_action, number_agents=nb_agents, random_start=random_start, delay=delay,
                          dt=dt, d_ref=d_ref, local_time_step=local_time_step, active_agents=act_agents,
                          gains_reward=gains, u_i=enable_u_i, stacked_frames=stacked_frames,
                          reward_analysis=reward_analysis, perturbations=perturbations, obs_front=obs_front,
                          load_agent=load_agent, number_steps=number_steps)

S_DIM = env.observation_space
A_DIM = env.action_space
ram = None

trainer = Trainer(S_DIM, A_DIM, ram, device)
trainer.load_models(episode=model_number, path_folder=folder)

number_tests = 6
for nb_active_agents in range(1, nb_agents+1):
    print('number active agents: ', nb_active_agents)
    previous_tests = np.full((number_tests, nb_active_agents), np.nan)
    number_tests = 1 if nb_active_agents == nb_agents else number_tests
    for j in range(0, number_tests):
        new_sample = True
        while new_sample:
            act_agents = sample(range(nb_agents), nb_active_agents)
            act_agents = np.sort(act_agents)
            new_sample = any(np.array_equal(act_agents, previous) for previous in previous_tests)
        previous_tests[j] = act_agents

        env = LoopFollowingAgents(available_obs=available_obs, obs_u_i=obs_u_i, mode_action=mode_action,
                                  type_action=type_action, number_agents=nb_agents, random_start=random_start,
                                  delay=delay, active_agents=act_agents,
                                  gains_reward=gains, u_i=enable_u_i, stacked_frames=stacked_frames,
                                  reward_analysis=reward_analysis, perturbations=perturbations, obs_front=obs_front,
                                  load_agent=load_agent, number_steps=number_steps)

        observations = env.reset(save_traj=True)
        for step in range(number_steps):
            state = np.float32(observations)
            if len(act_agents) != 1:
                actions = np.array([])
                for i in range(len(act_agents)):
                    action = trainer.get_exploitation_action(state[i])
                    action = action.cpu().numpy()
                    actions = np.concatenate([actions, action], axis=None)
            else:
                actions = trainer.get_exploitation_action(state)
                actions = actions.cpu().numpy()

            observations, _, _, infos = env.step(actions, step)

        reward = infos['reward']
        sum_reward_tot = np.sum(reward, axis=2)
        positions = infos['positions']
        speed = infos['speeds']
        error_speeds_learning = infos['error_speed_learning']
        error_learning = infos['error_position_learning']
        action = infos['action']
        control_learning = infos['control_input']
        u_i = infos['int_term']
        time = np.linspace(dt, number_steps * dt, number_steps)
        for i in range(number_steps - 1):
            for k in range(nb_agents):
                if np.abs(positions[i + 1][k] - positions[i][k]) > d_ref * (nb_agents - 1):
                    positions[i + 1][k] = np.nan

        save_path = './Figures/copy_policy/number_agents_' + str_N
        save_path += '/alpha_10e' + alp + '_tau_10e' + tau + '_decay_'
        save_path += '0' if decay == '0' else '10e' + decay
        save_path += '/perturbations_load_' + str(load_agent) if perturbations else '/no_perturbations'
        save_path += '/obs_' + obs
        save_path += '/2_layer_size_128'
        save_path += '/model_number_' + str(model_number)
        save_path += '/number_active_agents_' + str(nb_active_agents)
        save_path += '/active_agents_' + str(act_agents) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        title = 'model number ' + str(model_number)
        title += ', stacked frames' if stacked_frames else 'no stacked frames'
        title += ', type action = ' + str(type_action)
        title += ', active agents ' + str(act_agents)
        f1, ax = plt.subplots(4, 2)
        f1.suptitle(title)
        ax[0][0].plot(time, positions)
        ax[0][0].set_title('Agent Positions')
        ax[0][0].axis([0, number_steps * dt, 0, d_ref * nb_agents])
        ax[0][1].plot(time, speed)
        ax[0][1].set_title('Agent Speeds')
        ax[1][0].plot(time, sum_reward_tot)
        ax[1][0].set_title('Reward')
        ax[1][1].plot(time, error_learning)
        ax[1][1].set_title('Error Position of the Learning Agent')
        ax[2][0].plot(time, error_speeds_learning)
        ax[2][0].set_title('Error Speed of the Learning Agent')
        ax[2][1].plot(time, action)
        ax[2][1].set_title('Action')
        ax[3][0].plot(time, control_learning)
        ax[3][0].set_title('Control Input of the Learning Agent')
        ax[3][1].plot(time, u_i)
        ax[3][1].set_title('Integral Term of the learning agent')
        f1.set_size_inches(20, 14)
        f1.savefig(save_path + 'global_infos')
        f1.clf()
        if len(act_agents) != 1:
            figs = [None] * len(act_agents)
            axs = [None] * len(act_agents)
            for i in range(len(act_agents)):
                title = 'model number ' + str(model_number)
                title += ', stacked frames' if stacked_frames else ', no stacked frames'
                title += ', type action = ' + str(type_action)
                title += ', agents nb ' + str(act_agents[i])
                figs[i], axs[i] = plt.subplots(3, 2)
                figs[i].suptitle(title)
                axs[i][0][0].plot(time, error_learning[:, i])
                axs[i][0][0].set_title('Error Position of the Learning Agent')
                axs[i][0][1].plot(time, error_speeds_learning[:, i])
                axs[i][0][1].set_title('Error Speed of the Learning Agent')
                reward_learning = [sum(el[i]) for el in reward]
                r_i = [sum(el) for el in reward[:, i]]
                axs[i][1][0].plot(time, r_i)
                axs[i][1][0].set_title('Reward')
                axs[i][1][1].plot(time, action[:, i * A_DIM:(i + 1) * A_DIM])
                axs[i][1][1].set_title('Action')
                u = [el[i] for el in control_learning]
                axs[i][2][0].plot(time, control_learning[:, i])
                axs[i][2][0].set_title('Control input')
                axs[i][2][1].plot(time, u_i[:, i])
                axs[i][2][1].set_title('Integral Term')
                figs[i].set_size_inches(20, 14)
                figs[i].savefig(save_path + 'agents_' + str(act_agents[i]))
                figs[i].clf()
