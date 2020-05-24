import numpy as np
import pickle
import torch
import sys
import os
from Environment.LoopFollowingAgents import LoopFollowingAgents
import RDPG.train as train
import matplotlib.pyplot as plt
import argparse
sys.path.append(os.getcwd())


parser = argparse.ArgumentParser()

parser.add_argument('-e', '--number_episodes', type=int, required=True)
parser.add_argument('-model', '--model_number', type=int, required=True)
parser.add_argument('-gpu', '--enable_gpu', action='store_true')
parser.add_argument('-s', '--number_steps', type=int, required=True)
parser.add_argument('-b', '--batch', type=int)

load_agent = 0

args = parser.parse_args()

model_number = args.model_number
batch_size = args.number_episodes

if args.enable_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
number_steps = args.number_steps
if args.batch:
    batch = args.batch
else:
    batch = 1

str_N = '6'
obs = '0+front_agent'
gam = '0.99'
alp = '-5'
t = '-5'
k = '0'
name = 'test1'

copy = True

folder = './Saved_results/number_agents_' + str_N
folder += '/observation_' + obs
folder += '/gamma_' + gam
folder += '/learning_rate_10e' + alp + '_tau_10e' + t
folder += '_decay_tau_' + k
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
u_i = from_pickle['enable_u_i']
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
plot_episode = True

act_agents = [0, 3]

env = LoopFollowingAgents(available_obs=available_obs, obs_u_i=obs_u_i, mode_action=mode_action,
                          type_action=type_action, number_agents=nb_agents, random_start=random_start, delay=delay,
                          dt=dt, d_ref=d_ref, local_time_step=local_time_step, active_agents=act_agents,
                          gains_reward=gains, u_i=u_i, stacked_frames=stacked_frames, reward_analysis=reward_analysis,
                          perturbations=perturbations, obs_front=obs_front, load_agent=load_agent,
                          number_steps=number_steps)

S_DIM = env.observation_space
A_DIM = env.action_space
A_MAX = torch.tensor(env.action_space_max).float().to(device)
ram = None

if copy:
    A_DIM = int(A_DIM / len(act_agents))
    print(A_DIM)
    A_MAX = torch.tensor(env.action_space_max[:A_DIM]).float().to(device)

trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram, device)
trainer.load_models(episode=model_number, path_folder=folder)

sum_rewards_pos = np.zeros((batch, batch_size))
sum_rewards_spe = np.zeros((batch, batch_size))
sum_rewards_u = np.zeros((batch, batch_size))
sum_rewards_4 = np.zeros((batch, batch_size))
sum_rewards_tot = np.zeros((batch, batch_size))

for nb_batch in range(batch):
    for epi in range(batch_size):
        print('batch number ', nb_batch, ' : episode number ', epi)

        sum_reward_pos = 0
        sum_reward_spe = 0
        sum_reward_u = 0
        sum_reward_4 = 0

        observations = env.reset(save_traj=True)

        for step in range(number_steps):

            state = np.float32(observations)
            if available_obs != 1 and len(act_agents) != 1:
                actions = np.array([])
                for i in range(len(act_agents)):
                    action = trainer.get_exploitation_action(state[i])
                    action = action.cpu().numpy()
                    actions = np.concatenate([actions, action], axis=None)
            else:
                actions = trainer.get_exploitation_action(state)
                actions = actions.cpu().numpy()

            observations, reward, done, infos = env.step(actions, step)

            if len(act_agents) == 1:
                sum_reward_pos += reward[0]
                sum_reward_spe += reward[1]
                sum_reward_u += reward[2]
                sum_reward_4 += reward[3]
            else:
                agent_reward_analysis = 0
                sum_reward_pos += reward[agent_reward_analysis][0]
                sum_reward_spe += reward[agent_reward_analysis][1]
                sum_reward_u += reward[agent_reward_analysis][2]
                sum_reward_4 += reward[agent_reward_analysis][3]

        sum_reward_tot = sum_reward_pos + sum_reward_spe + sum_reward_u + sum_reward_4

        sum_rewards_tot[nb_batch][epi] = sum_reward_tot
        sum_rewards_pos[nb_batch][epi] = sum_reward_pos
        sum_rewards_spe[nb_batch][epi] = sum_reward_spe
        sum_rewards_u[nb_batch][epi] = sum_reward_u
        sum_rewards_4[nb_batch][epi] = sum_reward_4

mean_rewards_tot = [0] * batch
var_rewards_tot = [0] * batch
mean_rewards_pos = [0] * batch
var_rewards_pos = [0] * batch
mean_rewards_spe = [0] * batch
var_rewards_spe = [0] * batch
mean_rewards_u = [0] * batch
var_rewards_u = [0] * batch
mean_rewards_4 = [0] * batch
var_rewards_4 = [0] * batch


def mean(array):
    return sum(array)/len(array)


def variance(array):
    m = mean(array)
    return mean([(x - m)**2 for x in array])


for i in range(batch):
    mean_rewards_tot[i] = mean(sum_rewards_tot[i])
    var_rewards_tot[i] = variance(sum_rewards_tot[i])
    mean_rewards_pos[i] = mean(sum_rewards_pos[i])
    var_rewards_pos[i] = variance(sum_rewards_pos[i])
    mean_rewards_spe[i] = mean(sum_rewards_spe[i])
    var_rewards_spe[i] = variance(sum_rewards_spe[i])
    mean_rewards_u[i] = mean(sum_rewards_u[i])
    var_rewards_u[i] = variance(sum_rewards_u[i])
    mean_rewards_4[i] = mean(sum_rewards_4[i])
    var_rewards_4[i] = variance(sum_rewards_4[i])

print('\n')
print('mean of mean pos = ', mean(mean_rewards_pos))
print('Var of mean pos = ', variance(mean_rewards_pos))
print('mean of var pos = ', mean(var_rewards_pos))
print('Var of var pos = ', variance(var_rewards_pos))
print('\n')
print('mean of mean spe = ', mean(mean_rewards_spe))
print('Var of mean spe = ', variance(mean_rewards_spe))
print('mean of var spe = ', mean(var_rewards_spe))
print('Var of var spe = ', variance(var_rewards_spe))
print('\n')
print('mean of mean u = ', mean(mean_rewards_u))
print('Var of mean u = ', variance(mean_rewards_u))
print('mean of var u = ', mean(var_rewards_u))
print('Var of var u = ', variance(var_rewards_u))
print('\n')
print('mean of mean 4 = ', mean(mean_rewards_4))
print('Var of mean 4 = ', variance(mean_rewards_4))
print('mean of var 4 = ', mean(var_rewards_4))
print('Var of var 4 = ', variance(var_rewards_4))
print('\n')
print('mean of mean tot = ', mean(mean_rewards_tot))
print('Var of mean tot = ', variance(mean_rewards_tot))
print('mean of var tot = ', mean(var_rewards_tot))
print('Var of var tot = ', variance(var_rewards_tot))

if plot_episode:
    reward = infos['reward']
    sum_reward_tot = np.sum(reward, axis=2)
    # print(sum_reward_tot)
    positions = infos['positions']
    speed = infos['speeds']
    error_speeds_learning = infos['error_speed_learning']
    error_learning = infos['error_position_learning']
    action = infos['action']
    control_learning = infos['control_input']
    u_i = infos['int_term']

    print('sum reward = ', np.sum(reward))

    time = np.linspace(dt, number_steps * dt, number_steps)
    for i in range(number_steps - 1):
        for j in range(nb_agents):
            if np.abs(positions[i + 1][j] - positions[i][j]) > d_ref * (nb_agents - 1):
                positions[i + 1][j] = np.nan

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
    if len(act_agents) != 1:
        figs = [None]*len(act_agents)
        axs = [None]*len(act_agents)
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
            axs[i][1][1].plot(time, action[:, i*A_DIM:(i+1)*A_DIM])
            axs[i][1][1].set_title('Action')
            u = [el[i] for el in control_learning]
            axs[i][2][0].plot(time, control_learning[:, i])
            axs[i][2][0].set_title('Control input')
            axs[i][2][1].plot(time, u_i[:, i])
            axs[i][2][1].set_title('Integral Term')

    # f2.show()
    # save_path = folderh + '/episode_' + str(episode_number)
    # f1.set_size_inches(20, 14)
    # f1.savefig(save_path)
    plt.show()
