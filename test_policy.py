import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
# import os
# import sys
from random import sample
from Environment.LoopFollowingAgents import LoopFollowingAgents
from RDPG.train import Trainer
# sys.path.append(os.getcwd())


str_N = '6'
obs = '1'
learning_rate_actor = '-5'
learning_rate_critic = '-5'
adaptive_learning_rate_actor = '-5'
adaptive_learning_rate_critic = '-5'
tau = '-2'
decay = '0'
input_dim = 2
name = 'test1'

model_number = 5000
number_steps = 1000

device = torch.device("cpu")

folder = './Saved_results/number_agents_' + str_N
folder += '/test_parameters/delay_variation'
folder += '/observation_' + obs
folder += '/3_layers_size_128'
folder += '/learning_rate_actor_10e' + learning_rate_actor + '_critic_10e' + learning_rate_critic
folder += '/tau_10e' + tau + '_decay_tau_' + decay
folder += '/adaptive_learning_rate_actor_10e' + adaptive_learning_rate_actor + '_critic_10e' + \
          adaptive_learning_rate_critic
folder += '/input_' + str(input_dim)
folder += '/' + name + '/'

with open(folder + 'infos.pickle', 'rb') as file:
    from_pickle = pickle.load(file)

available_obs = from_pickle['available_observations']
nb_agents = from_pickle['number_agents']
dt = from_pickle['time_step']
d_ref = from_pickle['distance_ref']
enable_u_i = from_pickle['enable_u_i']
gains = from_pickle['gains_reward']
local_time_step = from_pickle['local_time_step']
random_start = from_pickle['random_start']
active_agents_train = from_pickle['active_agents']
range_delay = from_pickle['range_delay']

# dim_obs, action_space

del from_pickle

reward_analysis = True
perturbations = True

active_agents = [0]
load_agent = 0

delay = None

env = LoopFollowingAgents(available_obs=available_obs, number_agents=nb_agents, random_start=random_start, dt=dt,
                          d_ref=d_ref, local_time_step=local_time_step, active_agents=active_agents,
                          gains_reward=gains, number_steps=number_steps, delay=delay)

observation_dim = env.observation_space
action_dim = env.action_space
memory = None

trainer = Trainer(observation_dim=observation_dim, action_dim=action_dim, env=env, number_steps=number_steps, memory=memory,
                  device=device, extended_inputs=input_dim)
trainer.load_models(episode=model_number, path_folder=folder)

for new_delay in range_delay:

    observation, init_state, _ = env.reset(get_init_state=True, save_traj=True, delay=new_delay)

    obs = torch.from_numpy(np.float32(observation).reshape(1, observation_dim)).to(device)

    previous_action = torch.zeros(1, action_dim, dtype=torch.float32).to(device)
    previous_reward = torch.zeros(1, 1, dtype=torch.float32).to(device)

    h_actor = trainer.actor.init_hidden_state(1)

    inputs = torch.cat((obs, previous_action, previous_reward), dim=1)

    with torch.no_grad():
        for step in range(number_steps):
            action_input_2, h_actor = trainer.get_exploitation_action(inputs, h_actor)

            action_cpu = action_input_2.cpu().numpy()

            new_observation, reward_input_2, done, info_input_2 = env.step(action_cpu, step)

            reward_input_2 = torch.from_numpy(reward_input_2.astype(np.float32).reshape(1, 1)).to(device)

            obs = torch.from_numpy(np.float32(new_observation).reshape(1, observation_dim)).to(device)

            previous_action = action_input_2
            previous_reward = reward_input_2

            inputs = torch.cat((obs, previous_action, previous_reward), dim=1)

    observation = env.reset(initial_state=init_state, save_traj=True, delay=new_delay)

    obs = torch.from_numpy(np.float32(observation).reshape(1, observation_dim)).to(device)

    previous_action = torch.zeros(1, action_dim, dtype=torch.float32).to(device)
    previous_reward = torch.zeros(1, 1, dtype=torch.float32).to(device)

    h_actor = trainer.actor.init_hidden_state(1)

    inputs = torch.cat((obs, previous_action, previous_reward), dim=1)

    with torch.no_grad():
        for step in range(number_steps):
            action_input_2, h_actor = trainer.get_exploitation_action(inputs, h_actor)

            action_cpu = action_input_2.cpu().numpy()

            new_observation, reward_input_2, done, info_no_input_2 = env.step(action_cpu, step)

            reward_input_2 = torch.from_numpy(reward_input_2.astype(np.float32).reshape(1, 1)).to(device)

            obs = torch.from_numpy(np.float32(new_observation).reshape(1, observation_dim)).to(device)

            inputs = torch.cat((obs, previous_action, previous_reward), dim=1)

    reward_input_2 = info_input_2['reward']
    positions_input_2 = info_input_2['positions']
    speed_input_2 = info_input_2['speeds']
    error_speeds_learning_input_2 = info_input_2['error_speed_learning']
    error_learning_input_2 = info_input_2['error_position_learning']
    action_input_2 = info_input_2['action']
    control_learning_input_2 = info_input_2['control_input']
    u_i_input_2 = info_input_2['int_term']

    reward_no_input_2 = info_no_input_2['reward']
    positions_no_input_2 = info_no_input_2['positions']
    speed_no_input_2 = info_no_input_2['speeds']
    error_speeds_learning_no_input_2 = info_no_input_2['error_speed_learning']
    error_learning_no_input_2 = info_no_input_2['error_position_learning']
    action_no_input_2 = info_no_input_2['action']
    control_learning_no_input_2 = info_no_input_2['control_input']
    u_i_no_input_2 = info_no_input_2['int_term']
    # print('HEY')
    # print(len(reward))
    # print(len(reward[0]))
    # print('sum reward = ', np.sum(reward))

    time = np.linspace(dt, number_steps * dt, number_steps)
    for epi in range(number_steps - 1):
        for j in range(nb_agents):
            if np.abs(positions_input_2[epi + 1][j] - positions_input_2[epi][j]) > d_ref * (nb_agents - 1):
                positions_input_2[epi + 1][j] = np.nan
            if np.abs(positions_no_input_2[epi + 1][j] - positions_no_input_2[epi][j]) > d_ref * (nb_agents - 1):
                positions_no_input_2[epi + 1][j] = np.nan

    fig, ax = plt.subplots(4, 2)
    title = 'model number ' + str(model_number) + ', delay ' + str(new_delay)
    fig.suptitle(title)
    ax[0][0].plot(time, positions_input_2)
    ax[0][0].set_title('Agent Positions input 2')
    ax[0][1].plot(time, positions_no_input_2)
    ax[0][1].set_title('Agent Positions no input 2')
    ax[1][0].plot(time, reward_input_2, time, reward_no_input_2)
    ax[1][0].set_title('Reward')
    ax[1][1].plot(time, error_learning_input_2, time, error_learning_no_input_2)
    ax[1][1].set_title('Error Position of the Learning Agent')
    ax[2][0].plot(time, error_speeds_learning_input_2, time, error_speeds_learning_no_input_2)
    ax[2][0].set_title('Error Speed of the Learning Agent')
    ax[2][1].plot(time, action_input_2, time, action_no_input_2)
    ax[2][1].set_title('action')
    ax[3][0].plot(time, control_learning_input_2, time, control_learning_no_input_2)
    ax[3][0].set_title('Control Input of the Learning Agent')
    ax[3][1].plot(time, action_input_2 - action_no_input_2)
    # ax[3][1].plot(time, error_speeds_learning_input_2 - error_speeds_learning_no_input_2)
    ax[3][1].set_title('diff between action input 2 and no input 2')

    # print(error_speeds_learning_input_2[0] - error_speeds_learning_no_input_2[0])

    ax[0][0].axis([0, number_steps * dt, 0, d_ref * nb_agents])
    ax[0][1].axis([0, number_steps * dt, 0, d_ref * nb_agents])
    # fig.show()
    save_path = './test_input_2/model_' + str(model_number) + '_delay_' + str(new_delay)
    fig.set_size_inches(20, 14)
    plt.show()
    # fig.savefig(save_path)
    # plt.close(fig)

