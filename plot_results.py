import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from Environment.LoopFollowingAgents import LoopFollowingAgents

save = True

n_ex = 20

learning_rate_actor = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
learning_rate_critic = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
# learning_rate_actor = [1e-3]
# learning_rate_critic = [1e-5]
tau = [1e-1, 1e-2]
decay_tau = [0]

extended_inputs = [0, 1, 2]
# test_names = ['test1', 'test2', 'test3']
test_names = ['test2']

# available_obs = [0, 2]
available_obs = [0, 1, 2]
number_agents = [6]


def modify_pickle(folder_path, from_pickle, params):

    # obs_dim = from_pickle['observation_space']
    # # stacked = from_pickle['stacked_frames']
    # # print(stacked)
    # print(obs_dim)
    from_pickle['range_delay'] = np.array([5])
    from_pickle['mean_delay'] = 5
    from_pickle['sigma_varation_delay'] = 0.01
    # infos = from_pickle.pop('infos')
    # saved_batch_number = from_pickle.pop('saved_batch_number')
    #
    # N_saved = 2/params
    #
    # new_infos = []
    # new_saved_batch_number = []
    # for i in range(len(infos)):
    #     if i % N_saved == 0 or i == len(infos) - 1:
    #         new_infos.append(infos[i])
    #         new_saved_batch_number.append(saved_batch_number[i])
    #
    # from_pickle['infos'] = new_infos
    # from_pickle['saved_batch_number'] = new_saved_batch_number
    #
    # # from_pickle['infos'] = infos
    with open(folder_path + 'infos.pickle', 'wb') as file:
        pickle.dump(from_pickle, file)
        print('pickle saved')


def save_reward(folder_path, from_pickle):
    number_episodes = from_pickle['number_episodes']
    # number_episodes = 100
    number_steps = from_pickle['number_steps']
    # print(number_steps)
    n = from_pickle['number_agents']
    random_start = from_pickle['random_start']
    # delay = from_pickle['delay']
    dt = from_pickle['time_step']
    d_ref = from_pickle['distance_ref']
    local_dt = from_pickle['local_time_step']
    gains_reward = from_pickle['gains_reward']
    u_i = from_pickle['enable_u_i']
    delay = from_pickle['delay']
    range_delay = from_pickle['range_delay']

    reward_no_delay_split = [[]for _ in range_delay]
    reward_no_action_split = [[] for _ in range_delay]
    reward_no_control_split = [[] for _ in range_delay]
    reward_rl_split = [[] for _ in range_delay]
    episode_split = [[] for _ in range_delay]
    all_reward_rl = from_pickle['sum_reward_history']
    reward_rl = []
    nor_reward_rl = []
    reward_no_delay = []
    nor_reward_no_delay = []
    reward_no_action = []
    reward_no_control = []

    initial_states = from_pickle['initial_states']

    env_no_control = LoopFollowingAgents(available_obs=0, number_agents=n, random_start=random_start, dt=dt, d_ref=d_ref,
                                         local_time_step=local_dt, gains_reward=gains_reward, number_steps=number_steps)
    env_no_action = LoopFollowingAgents(available_obs=0, number_agents=n, random_start=random_start, dt=dt, d_ref=d_ref,
                                        local_time_step=local_dt, gains_reward=gains_reward, number_steps=number_steps)
    env_no_delay = LoopFollowingAgents(available_obs=0, number_agents=n, random_start=random_start, dt=dt, d_ref=d_ref,
                                       local_time_step=local_dt, gains_reward=gains_reward, number_steps=number_steps)

    [kp, kd] = env_no_control.feedback_params()
    #
    # action = [-kp, -ki, -kd]
    # print('action = ', action)

    for episode in range(0, number_episodes, n_ex):
        print(episode)
        initial_state = initial_states[episode]
        new_delay = delay[episode]
        obs_no_control = env_no_control.reset(initial_state=initial_state, delay=new_delay, save_traj=True)
        _ = env_no_action.reset(initial_state=initial_state, delay=new_delay, save_traj=True)
        _ = env_no_delay.reset(initial_state=initial_state, delay=1, save_traj=True)

        sum_reward1 = 0
        sum_reward2 = 0
        sum_reward3 = 0

        for step in range(number_steps):
            action = -kp * obs_no_control[0] - kd * obs_no_control[1]
            action = np.array([action])
            obs_no_control, reward1, _, infos = env_no_control.step(action, step)
            _, reward2, _, infos = env_no_action.step([0], step)
            _, reward3, _, _ = env_no_delay.step([0], step)
            sum_reward1 += reward1
            sum_reward2 += reward2
            sum_reward3 += reward3
            # print('1 = ',sum_reward1)
            # print('2 = ',sum_reward2)
            # print('3 = ',sum_reward3)

        reward_no_control.append(sum_reward1)
        reward_no_action.append(sum_reward2)
        reward_no_delay.append(sum_reward3)
        reward_rl.append(all_reward_rl[episode])

        index_delay = np.where(range_delay == new_delay)[0][0]
        reward_no_control_split[index_delay].append(sum_reward1)
        reward_no_action_split[index_delay].append(sum_reward2)
        reward_no_delay_split[index_delay].append(sum_reward3)
        reward_rl_split[index_delay].append(all_reward_rl[episode])
        episode_split[index_delay].append(episode)

        # print(sum_reward1)
        # print(sum_reward2)
        # print(sum_reward3)

        nor_reward_no_delay.append((sum_reward3 - sum_reward2) / (sum_reward1 - sum_reward2))
        nor_reward_rl.append((all_reward_rl[episode] - sum_reward2) / (sum_reward1 - sum_reward2))

    episodes = range(0, number_episodes, n_ex)
    # print('mean no control = ', mean(reward_no_control))
    # print('mean no action = ', mean(reward_no_action))
    # print('mean no delay = ', mean(reward_no_delay))
    # print('mean RL = ', mean(reward_rl))

    f1 = plt.figure(1)
    # plt.plot(episodes, reward_rl)
    plt.plot(episodes, reward_rl, episodes, reward_no_delay, episodes, reward_no_control, episodes,
             reward_no_action)
    plt.legend(('RL', 'no delay', 'no control', 'no action'))
    # f1.show()
    # f2 = plt.figure(2)
    # plt.plot(episodes, nor_reward_rl, episodes, nor_reward_no_delay)
    # plt.legend(('RL', 'no delay'))
    # f2.show()
    # plt.show()
    save_path = folder_path + 'reward_history_all_RL.png'
    f1.set_size_inches(20, 14)
    f1.savefig(save_path)
    # plt.show()
    f1.clf()
    # f2.clf()

    for delay, reward_no_control, reward_no_action, reward_no_delay, reward_rl, episodes \
            in zip(range_delay, reward_no_control_split, reward_no_action_split, reward_no_delay_split, reward_rl_split,
                   episode_split):
        fig = plt.figure()
        plt.plot(episodes, reward_rl, episodes, reward_no_delay, episodes, reward_no_control, episodes,
                 reward_no_action)
        plt.legend(('RL', 'no delay', 'no control', 'no action'))
        save_path = folder_path + 'reward_history_delay_' + str(delay) + '.png'
        fig.set_size_inches(20, 14)
        fig.savefig(save_path)
        # plt.show()
        fig.clf()


def save_episode_trajector(folder_path, from_pickle, index_episode):
    infos = from_pickle['infos']
    # print('length infos = ', len(infos))
    infos = infos[index_episode]
    index_episode = from_pickle['saved_episodes'][index_episode]
    episode_steps = from_pickle['number_steps']
    dt = from_pickle['time_step']
    d_ref = from_pickle['distance_ref']
    n = from_pickle['number_agents']
    active_agents = from_pickle['active_agents']
    delay_episode = from_pickle['delay'][index_episode]
    # sum_reward_history = from_pickle['sum_reward_history']
    # type_action = from_pickle['type_action']
    # print('mode_actor_critic = ', from_pickle['mode_actor_critic'])
    # execution_times = from_pickle['execution_times']
    # print(execution_times)
    del from_pickle

    # f1 = plt.figure(1)
    # plt.plot(execution_times)
    # f1.show()
    save_traj = True
    if save_traj:

        reward = infos['reward']
        positions = infos['positions']
        speed = infos['speeds']
        error_speeds_learning = infos['error_speed_learning']
        error_learning = infos['error_position_learning']
        action = infos['action']
        control_learning = infos['control_input']
        u_i = infos['int_term']
        # print('HEY')
        # print(len(reward))
        # print(len(reward[0]))
        # print('sum reward = ', np.sum(reward))

        time = np.linspace(dt, episode_steps * dt, episode_steps)
        for epi in range(episode_steps - 1):
            for j in range(n):
                if np.abs(positions[epi + 1][j] - positions[epi][j]) > d_ref * (n - 1):
                    positions[epi + 1][j] = np.nan

        f2, ax = plt.subplots(4, 2)
        title = 'episode number ' + str(index_episode) + ', active agents ' + str(active_agents)
        f2.suptitle(title)
        ax[0][0].plot(time, positions)
        ax[0][0].set_title('Agent Positions')
        ax[0][1].plot(time, speed)
        ax[0][1].set_title('Agent Speeds')
        ax[1][0].plot(time, reward)
        ax[1][0].set_title('Reward')
        ax[1][1].plot(time, error_learning)
        ax[1][1].set_title('Error Position of the Learning Agent')
        ax[2][0].plot(time, error_speeds_learning)
        ax[2][0].set_title('Error Speed of the Learning Agent')
        ax[2][1].plot(time, action)
        ax[2][1].set_title('action')
        ax[3][0].plot(time, control_learning)
        ax[3][0].set_title('Control Input of the Learning Agent')
        ax[3][1].plot(time, u_i)
        ax[3][1].set_title('integral term of the learning agent')

        ax[0][0].axis([0, episode_steps * dt, 0, d_ref * n])
        # f2.show()
        save_path = folder_path + '/episode_' + str(index_episode) + '_delay_' + str(delay_episode)
        f2.set_size_inches(20, 14)
        f2.savefig(save_path)
        f2.clf()


def save_trajectory(folder_path, from_pickle):

    episode_number = from_pickle['saved_episodes']
    delay_history = from_pickle['delay']
    range_delay = from_pickle['range_delay']

    for research_delay in range_delay:
        for count, episode in reversed(list(enumerate(episode_number))):
            if delay_history[episode] == research_delay:
                save_episode_trajector(folder_path, from_pickle, index_episode=count)
                # print(count, episode)
                break


def plot_learning_graphs(folder_path, from_pickle):
    reward_history = from_pickle['sum_reward_history']
    value_loss = from_pickle['critic_loss']
    policy_loss = from_pickle['actor_loss']
    delay = from_pickle['delay']
    range_delay = from_pickle['range_delay']
    number_delays = len(range_delay)
    number_episodes = from_pickle['number_episodes']
    mean_error = from_pickle['mean_errors']
    mean_relative_error = from_pickle['mean_relative_errors']

    nb_episode_plot = 10

    analyzed_episodes = range(0, number_episodes, nb_episode_plot)

    rewards_delays = [[] for _ in range(number_delays)]
    episodes_delays = [[] for _ in range(number_delays)]

    for episode in analyzed_episodes:
        index_delay = np.where(range_delay == delay[episode])
        rewards_delays[index_delay[0][0]].append(reward_history[episode])
        episodes_delays[index_delay[0][0]].append(episode)

    # mean_ratio = information['mean_ratio']
    # std_ratio = information['std_ratio']

    value_loss = np.log10(value_loss)

    f1 = plt.figure(1)
    plt.plot(reward_history)

    save_path = folder_path + '/reward_history.png'
    f1.set_size_inches(20, 14)
    f1.savefig(save_path)
    f1.clf()

    f2 = plt.figure(3)
    plt.plot(value_loss)

    save_path = folder_path + '/critic_loss_history.png'
    f2.set_size_inches(20, 14)
    f2.savefig(save_path)
    f2.clf()

    f3 = plt.figure(4)
    plt.plot(policy_loss)

    save_path = folder_path + '/actor_loss_history.png'
    f3.set_size_inches(20, 14)
    f3.savefig(save_path)
    f3.clf()

    f4 = plt.figure(5)
    plt.plot(delay)

    save_path = folder_path + '/delai_history.png'
    f4.set_size_inches(20, 14)
    f4.savefig(save_path)
    f4.clf()

    f5 = plt.figure(6)
    plt.plot(mean_error)
    save_path = folder_path + '/mean_error_history.png'
    f5.set_size_inches(20, 14)
    f5.savefig(save_path)
    f5.clf()

    f6 = plt.figure(7)
    plt.plot(mean_relative_error)
    save_path = folder_path + '/mean_relative_error_history.png'
    f6.set_size_inches(20, 14)
    f6.savefig(save_path)
    f6.clf()

    f5 = plt.figure(5)
    for plot_delay in range(number_delays):
        plt.plot(episodes_delays[plot_delay], rewards_delays[plot_delay], label=str(range_delay[plot_delay]))
    plt.legend()
    save_path = folder_path + 'reward_delay.png'
    f5.set_size_inches(20, 14)
    f5.savefig(save_path)
    # plt.show()
    f5.clf()

    # if save:
    #     save_path = folder_path + '/reward_history.png'
    #     f1.set_size_inches(20, 14)
    #     f1.savefig(save_path)
    #     f1.clf()
    #     save_path = folder_path + '/critic_loss_history.png'
    #     f2.set_size_inches(20, 14)
    #     f2.savefig(save_path)
    #     f2.clf()
    #     save_path = folder_path + '/actor_loss_history.png'
    #     f3.set_size_inches(20, 14)
    #     f3.savefig(save_path)
    #     f3.clf()
    #     save_path = folder_path + '/delai_history.png'
    #     f4.set_size_inches(20, 14)
    #     f4.savefig(save_path)
    #     f4.clf()


def action_on_pickle(folder_path, count, params):
    if os.path.exists(folder_path):
        print(folder_path)
        count[0] += 1

        with open(folder_path + 'infos.pickle', 'rb') as file:
            from_pickle = pickle.load(file)

        save_reward(folder_path, from_pickle)
        save_trajectory(folder_path, from_pickle)
        plot_learning_graphs(folder_path, from_pickle)
        # modify_pickle(folder_path, from_pickle, params)


i = [0]
for l_r_a in learning_rate_actor:
    for l_r_c in learning_rate_critic:
        for t in tau:
            for decay_t in decay_tau:
                for obs in available_obs:
                    for n in number_agents:
                        for name in test_names:
                            for extended in extended_inputs:
                                folder_path = './Saved_results/'
                                folder_path += 'number_agents_' + str(n)
                                folder_path += '/delay_variation_freeze_inputs_2_learning_decrease_learning_rate_10'
                                folder_path += '/observation_' + str(obs)
                                folder_path += '/3_layers_size_128'
                                folder_path += '/learning_rate_actor_10e' + str(int(np.log10(l_r_a))) + \
                                               '_critic_10e' + str(int(np.log10(l_r_c)))
                                folder_path += '/tau_10e' + str(int(np.log10(t)))
                                folder_path += '_decay_tau_' + str(decay_t)
                                folder_path += '/input_' + str(extended) + '/'
                                folder_path += name + '/'
                                # print(folder_path)
                                action_on_pickle(folder_path, i, n)

print(i)
