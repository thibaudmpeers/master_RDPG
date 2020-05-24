import numpy as np
import torch
import os
import sys
import gc
import pickle
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from RDPG.memory import Memory
import RDPG.train as train
from RDPG.train import get_nn_arch_params
from Environment.LoopFollowingAgents import LoopFollowingAgents
from Environment.utils import get_random_param
sys.path.append(os.getcwd())

t_init = time.time()

E_pickle = 50

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--nb_steps', type=int, required=True)
parser.add_argument('-e', '--nb_episodes', type=int, required=True)
parser.add_argument('-a_o', '--available_obs', type=int)
parser.add_argument('-o_ui', '--obs_int_term', action='store_true')
parser.add_argument('-a_ref', '--action_on_reference', action='store_true')
parser.add_argument('-a_u', '--action_on_terms', action='store_true')
parser.add_argument('-a_k', '--action_on_gains', action="store_true")
parser.add_argument('-e_p', '--enable_action_P', action='store_true')
parser.add_argument('-e_d', '--enable_action_D', action='store_true')
parser.add_argument('-e_i', '--enable_action_I', action='store_true')
parser.add_argument('-n', '--nb_agents', type=int)
parser.add_argument('-rs', '--random_start', type=int)
parser.add_argument("-gr", '--gains_reward', type=float, nargs=4)
parser.add_argument('-ui', '--enable_integral_term', action='store_true')
parser.add_argument('-gpu', '--enable_gpu', action='store_true')
parser.add_argument('-p', '--plot_sum_reward', action='store_true')
parser.add_argument('-ex', '--only_exploitation', action='store_true')

parser.add_argument('-act_ag', '--active_agents', type=int, nargs='*')
parser.add_argument('-l_actor', '--learning_rate_actor', type=float)
parser.add_argument('-l_critic', '--learning_rate_critic', type=float)
parser.add_argument('-tau', '--tau_ddpg', type=float)
parser.add_argument('-k_tau', '--decay_tau', type=float)
parser.add_argument('-inputs', '--extended_inputs', type=int)

args = parser.parse_args()

number_steps = args.nb_steps if args.nb_steps else 1000

number_episodes = args.nb_episodes

available_obs = args.available_obs

obs_u_i = args.obs_int_term

action_P = args.enable_action_P
action_D = args.enable_action_D
action_I = args.enable_action_I

choice = False
if args.action_on_reference:
    mode_action = 0
    choice = True
    type_action = None
    str_action = '/action_on_reference'

if args.action_on_terms:
    if choice:
        print('Multiple action modes have been chosen')
        sys.exit()
    mode_action = 1
    choice = True
    type_action = [True if (action_P or action_D) else False, action_I]
    str_action = '/action_on_terms_' + str(type_action).replace(' ', '')

if args.action_on_gains:
    if choice:
        print('Multiple action modes have been chosen')
        sys.exit()
    mode_action = 2
    choice = True
    type_action = [action_P, action_I, action_D]
    str_action = 'action_on_gains_' + str(type_action).replace(' ', '')

if not choice:
    print('No action mode selected')
    sys.exit()

print('mode action = ', mode_action)
print('type action = ', type_action)

nb_agents = args.nb_agents if args.nb_agents else 6

random_start = args.random_start if args.random_start is not None else 1

gains = args.gains_reward if args.gains_reward else [1, 1, 0.25, 10]

u_i = args.enable_integral_term

only_exploitation = args.only_exploitation

device = torch.device("cuda") if args.enable_gpu else torch.device("cpu")

active_agents = args.active_agents if args.active_agents is not None else [0]

learning_rate_actor = args.learning_rate_actor if args.learning_rate_actor else 1e-5

learning_rate_critic = args.learning_rate_critic if args.learning_rate_critic else 1e-5

tau = args.tau_ddpg if args.tau_ddpg else 1e-5

decay_tau = args.decay_tau if args.decay_tau else 0

extended_inputs = args.extended_inputs if args.extended_inputs is not None else 0

print(active_agents)

time_step = 0.025
local_time_step = 0.0125
dist_ref = 2

env = LoopFollowingAgents(available_obs=available_obs, number_agents=nb_agents, random_start=random_start, dt=time_step,
                          d_ref=dist_ref, local_time_step=local_time_step, active_agents=active_agents,
                          gains_reward=gains, number_steps=number_steps)

size_replay_memory = 100

observation_dim = env.observation_space
action_dim = env.action_space

memory = Memory(size=size_replay_memory, number_steps=number_steps, observation_dim=observation_dim,
                action_dim=action_dim, extended_inputs=extended_inputs, device=device)

comment = '_observation_' + str(available_obs)
comment += '_learning_rate_actor_10e' + str(int(np.log10(learning_rate_actor))) + '_critic_10e' + str(int(np.log10(learning_rate_critic)))
tensorboard = SummaryWriter(comment=comment)

manager_nn = train.Trainer(observation_dim=observation_dim, action_dim=action_dim, number_steps=number_steps,
                           memory=memory, device=device, l_r_actor=learning_rate_actor, l_r_critic=learning_rate_critic,
                           tau=tau, decay_tau=decay_tau, extended_inputs=extended_inputs, env=env, tensorboard=tensorboard)

learning_params = manager_nn.return_params()
random_params = manager_nn.noise.return_params()
number_layers, size_layers = get_nn_arch_params()

gamma = learning_params[1]
init_tau = learning_params[2]
decay_factor_tau = learning_params[3]

nb_actor_params, nb_critic_params = manager_nn.get_models_params()
print('actor parameters = ', nb_actor_params, '; critic_parameters = ', nb_critic_params)

path = './Saved_results/number_agents_' + str(nb_agents)
path += '/delay_variation_freeze_inputs_2_learning_decrease_learning_rate_10'
path += '/observation_' + str(available_obs)
path += '/' + str(number_layers) + '_layers_size_' + str(size_layers)
path += '/learning_rate_actor_10e' + str(int(np.log10(learning_rate_actor))) + '_critic_10e' + str(int(np.log10(learning_rate_critic)))
path += '/tau_10e' + str(int(np.log10(init_tau)))
path += '_decay_tau_0' if decay_factor_tau == 0 else '_decay_tau_10e' + str(int(np.log10(decay_factor_tau)))
path += '/input_' + str(extended_inputs)
path += '/test'

nb_tests = 1
while os.path.exists(path + str(nb_tests)):
    nb_tests += 1

path += str(nb_tests) + '/'
os.makedirs(path + 'Models/')

range_delay, mean_delay, increase_rate_delay = get_random_param()

to_pickle = {'number_agents': nb_agents, 'delay': [], 'time_step': time_step, 'local_time_step': local_time_step,
             'distance_ref': dist_ref, 'active_agents': active_agents, 'number_layers': number_layers,
             'available_observations': available_obs, 'observation_integral_term': obs_u_i, 'mode_action': mode_action,
             'type_action': type_action, 'random_start': random_start, 'gains_reward': gains, 'enable_u_i': u_i,
             'number_episodes': number_episodes, 'number_steps': number_steps,
             'size_replay_memory': size_replay_memory, 'observation_space': observation_dim, 'action_space': action_dim,
             'learning_parameters': learning_params, 'random_parameters': random_params, 'size_layers': size_layers,
             'nb_actor_parameters': nb_actor_params, 'nb_critic_parameters': nb_critic_params,
             'feedback_parameters': env.feedback_params(), 'infos': [], 'saved_episode_number': [],
             'sum_reward_history': [], 'initial_states': [], 'critic_loss': [], 'actor_loss': [], 'saved_episodes': [],
             'range_delay': range_delay, 'mean_delay': mean_delay, 'sigma_varation_delay': increase_rate_delay,
             'mean_errors': [], 'mean_relative_errors': []}

print(' State Dimensions :- ', observation_dim)
print(' Action Dimensions :- ', action_dim)

execution_times = np.zeros(number_episodes)

for episode in range(number_episodes):
    t_start = time.time()

    save_traj = True if episode % E_pickle == 0 else False

    memory.new_epsisode()

    manager_nn.update_tau(episode)

    t1 = time.time()

    exploitation = True if only_exploitation or episode % 20 == 0 else False

    sum_rewards, info, new_delay = manager_nn.run_episode(save_traj=save_traj, to_pickle=to_pickle,
                                                          exploitation=exploitation, episode_count=episode)
    tensorboard.add_scalar('delay', new_delay, episode)

    t2 = time.time()

    critic_loss, actor_loss, mean_errors, mean_relative_errors = manager_nn.optimize()

    t3 = time.time()

    print('episode number ', episode)
    print('delay = ', new_delay)
    print('env time = ', t2 - t1)
    print('optim time = ', t3 - t2)
    print('sum rewards = ', sum_rewards)
    print('critic_loss = ', critic_loss)
    print('actor loss = ', actor_loss)

    gc.collect()

    execution_times[episode] = t3 - t2

    to_pickle['sum_reward_history'].append(sum_rewards)
    to_pickle['critic_loss'].append(critic_loss)
    to_pickle['actor_loss'].append(actor_loss)
    to_pickle['mean_errors'].append(mean_errors)
    to_pickle['mean_relative_errors'].append(mean_relative_errors)
    if save_traj:
        to_pickle['infos'].append(info)
        to_pickle['saved_episode_number'].append(episode)
        to_pickle['saved_episodes'].append(episode)
    if episode % 1000 == 0:
        with open(path + 'infos.pickle', 'wb') as file:
            pickle.dump(to_pickle, file)
    if episode % 500 == 0:
        manager_nn.save_models(episode, path)

to_pickle['execution_times'] = execution_times
with open(path + 'infos.pickle', 'wb') as file:
    pickle.dump(to_pickle, file)

total_time = time.time() - t_init

mean = sum(execution_times) / len(execution_times)
sd = np.sqrt(sum((execution_times - mean) ** 2) / len(execution_times))
print('mean execution time = ', mean)
print('standard deviation execution time = ', sd)
print('total time = ', total_time)
print('Completed episodes')
