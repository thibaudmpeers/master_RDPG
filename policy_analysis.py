import torch
import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
import RDPG.train as train
from Environment.LoopFollowingAgents import LoopFollowingAgents
sys.path.append(os.getcwd())



str_N = '6'
obs = '2'
gam = '0.99'
alp = '-5'
t = '-5'
k = '0'
name = 'test1'

plot_graphs = True
frequency_analysis = True
temporal_analysis = True
start_step_analysis = 750


model_number = 5000
number_steps = 1000
n = number_steps - start_step_analysis
number_episodes = 1
number_batch = 1

reward_analysis = False
perturbations = False
load_agent = None
device = torch.device("cpu")

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
act_agents = from_pickle['active_agents']

del from_pickle

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

models = train.Trainer(S_DIM, A_DIM, A_MAX, ram, device)
models.load_models(episode=model_number, path_folder=folder)

time = np.linspace(dt, number_steps * dt, number_steps)

shift_inf = -12
shift_sup = 3
shifts = range(shift_inf, shift_sup + 1, 1)
nb_shift = len(shifts)

max_shift = np.maximum(np.abs(shift_inf), shift_sup)


scores_action = np.zeros((number_batch, number_episodes, nb_shift, 2))
feedback_gains_action = np.zeros((number_batch, number_episodes, nb_shift, 2, 2))
intercepts_action = np.zeros((number_batch, number_episodes, nb_shift))
best_shift_action = np.zeros((number_batch, number_episodes, 2))

scores_control = np.zeros((number_batch, number_episodes, nb_shift, 2))
feedback_gains_control = np.zeros((number_batch, number_episodes, nb_shift, 2, 2))
intercepts_control = np.zeros((number_batch, number_episodes, nb_shift))
best_shift_control = np.zeros((number_batch, number_episodes, 2))

for batch in range(number_batch):
    for epi in range(number_episodes):
        print('batch number ', batch, ', episode number ', epi)
        observation = env.reset(save_traj=True)
        for step in range(number_steps):
            state = np.float32(observation)
            analysis_action = models.get_exploitation_action(state)
            analysis_action = analysis_action.cpu().numpy()
            observation, reward, done, infos = env.step(analysis_action, step)

        action = infos['action']
        control_input = infos['control_input']
        err_position = infos['error_position_learning']
        err_speed = infos['error_speed_learning']

        if frequency_analysis:

            analysis_action = action[start_step_analysis:].squeeze()
            analysis_control = control_input[start_step_analysis:].squeeze()
            analysis_err_position = err_position[start_step_analysis:].squeeze()
            analysis_err_speed = err_speed[start_step_analysis:].squeeze()

            analysis_PID = analysis_control - analysis_action

            freq = np.fft.rfftfreq(n)/dt

            ft_err_position = np.fft.rfft(analysis_err_position) / n
            ft_err_position_norm = np.absolute(ft_err_position)
            angle_err_postion = np.angle(ft_err_position)

            ft_action = np.fft.rfft(analysis_action)/n
            ft_action_norm = np.absolute(ft_action)
            angle_action = np.angle(ft_action)

            ft_PID = np.fft.rfft(analysis_PID)/n
            ft_PID_norm = np.absolute(ft_PID)
            angle_PID = np.angle(ft_PID)

            ft_control = np.fft.rfft(analysis_control)/n
            ft_control_norm = np.absolute(ft_control)
            angle_control = np.angle(ft_control)

            ind = np.array(argrelextrema(ft_err_position_norm, np.greater))
            ind = np.insert(ind, 0, 0)
            mean_ft_err = ft_err_position_norm.mean()
            ind = ind[ft_err_position_norm[ind] > mean_ft_err]

            ratio_action_PID = ft_action/ft_PID
            ratio_control_PID = ft_control/ft_PID

            angle_diff_PID_err = np.angle(ft_PID/ft_err_position)
            angle_diff_action_err = np.angle(ft_action/ft_err_position)
            angle_diff_control_err = np.angle(ft_control/ft_err_position)

            angle_diff_action_PID = np.angle(ratio_action_PID) % (2*np.pi)
            angle_diff_control_PID = np.angle(ratio_control_PID) % (2*np.pi)

            print('diff angle action-control at frequency ', freq[ind], ' = ', angle_diff_action_PID[ind])
            print('diff angle total-control at frequency ', freq[ind], ' = ', angle_diff_control_PID[ind])
            print('ratio action-control at frequency ', freq[ind], ' = ', np.absolute(ratio_action_PID[ind]))
            print('ratio angle total-control at frequency ', freq[ind], ' = ', np.absolute(ratio_control_PID[ind]))
            advanced_action = angle_diff_action_PID[ind] / (freq[ind]*2*np.pi)
            advanced_control = angle_diff_control_PID[ind] / (freq[ind]*2*np.pi)
            print('advance action-control at frequency ', freq[ind], ' = ', advanced_action)
            print('advance total-control at frequency ', freq[ind], ' = ', advanced_control)

            x = np.concatenate((analysis_err_position.reshape(-1, 1), analysis_err_speed.reshape(-1, 1)), axis=1)
            y_PID = analysis_PID

            model_PID = LinearRegression(fit_intercept=False)
            model_PID.fit(x, y_PID)
            print('\n')
            print('error linear regression of the PID control = ', model_PID.score(x, y_PID))
            print('feedback gains = ', model_PID.coef_)

        if temporal_analysis:

            if plot_graphs:
                Y_action = np.zeros((n, nb_shift))
                legend_action = []
                Y_control = np.zeros((n, nb_shift))
                legend_control = []

            shift_err_position = err_position[start_step_analysis - max_shift:-max_shift].reshape(-1, 1)
            shift_err_speed = err_speed[start_step_analysis - max_shift:-max_shift].reshape(-1, 1)
            shift_x = np.concatenate((shift_err_position, shift_err_speed), axis=1)
            y_action = action[start_step_analysis-max_shift:-max_shift].squeeze()
            y_control = control_input[start_step_analysis-max_shift:-max_shift].squeeze()
            for i in range(nb_shift):
                shift = shifts[i]
                shift_y_action = action[start_step_analysis-shift-max_shift:-shift-max_shift].squeeze() \
                    if shift+max_shift != 0 else action[start_step_analysis:].squeeze()
                model_bias = LinearRegression(fit_intercept=True)
                model_bias.fit(shift_x, shift_y_action)
                model_no_bias = LinearRegression(fit_intercept=False)
                model_no_bias.fit(shift_x, shift_y_action)

                scores_action[batch, epi, i, 0] = model_bias.score(shift_x, shift_y_action)
                scores_action[batch, epi, i, 1] = model_no_bias.score(shift_x, shift_y_action)
                feedback_gains_action[batch, epi, i, 0] = model_bias.coef_
                feedback_gains_action[batch, epi, i, 1] = model_no_bias.coef_
                intercepts_action[batch, epi, i] = model_bias.intercept_
                # print('\n')
                # print('linear regression error with a shift of ', shift, ' = ', model.score(shift_x, shift_y_action))
                # print('action gains = ', model.coef_, '; intercept = ', model.intercept_)
                if plot_graphs:
                    shift_err_position = err_position[start_step_analysis + shift - max_shift:shift - max_shift].reshape(-1, 1)\
                        if shift != max_shift else err_position[start_step_analysis:].reshape(-1, 1)
                    shift_err_speed = err_speed[start_step_analysis + shift - max_shift:shift - max_shift].reshape(-1, 1) \
                        if shift != max_shift else err_speed[start_step_analysis:].reshape(-1, 1)
                    shift_x_action = np.concatenate((shift_err_position, shift_err_speed), axis=1)
                    y_predicted = model_bias.predict(shift_x_action)
                    Y_action[:, i] = y_predicted.squeeze()
                    legend_action.append(str(shift))

                shift_y_control = control_input[start_step_analysis - shift - max_shift:-shift - max_shift].squeeze() \
                    if shift + max_shift != 0 else control_input[start_step_analysis:].squeeze()
                model_bias = LinearRegression(fit_intercept=True)
                model_bias.fit(shift_x, shift_y_control)
                model_no_bias = LinearRegression(fit_intercept=False)
                model_no_bias.fit(shift_x, shift_y_control)

                scores_control[batch, epi, i, 0] = model_bias.score(shift_x, shift_y_control)
                scores_control[batch, epi, i, 1] = model_no_bias.score(shift_x, shift_y_control)
                # print('\n')
                # print('linear regression error with a shift of ', shift, ' = ', model_no_bias.score(shift_x, shift_y_control))
                # print('action gains = ', model_no_bias.coef_, '; intercept = ', model_no_bias.intercept_)
                feedback_gains_control[batch, epi, i, 0] = model_bias.coef_
                feedback_gains_control[batch, epi, i, 1] = model_no_bias.coef_
                intercepts_control[batch, epi, i] = model_bias.intercept_

                if plot_graphs:
                    shift_err_position = err_position[start_step_analysis + shift - max_shift:shift - max_shift].reshape(-1, 1)\
                        if shift != max_shift else err_position[start_step_analysis:].reshape(-1, 1)
                    shift_err_speed = err_speed[start_step_analysis + shift - max_shift:shift - max_shift].reshape(-1, 1) \
                        if shift != max_shift else err_speed[start_step_analysis:].reshape(-1, 1)
                    shift_x_action = np.concatenate((shift_err_position, shift_err_speed), axis=1)
                    y_predicted = model_bias.predict(shift_x_action)
                    Y_control[:, i] = y_predicted.squeeze()
                    legend_control.append(str(shift))

            best_shift_action[batch, epi, 0] = np.argmax(scores_action[batch, epi, 0])
            best_shift_action[batch, epi, 1] = np.argmax(scores_action[batch, epi, 1])
            if plot_graphs:
                fig_action = plt.figure()
                legend_action.append('real action')
                plt.plot(range(n), Y_action, range(n), y_action, 'k')
                plt.legend(legend_action)
                fig_action.suptitle('action model')

                fig_control = plt.figure()
                legend_control.append('real control')
                plt.plot(range(n), Y_control, range(n), y_control, 'k')
                plt.legend(legend_control)
                fig_control.suptitle('control model')

        # plt.legend(('error positions', 'error speeds', 'action'))
        # kp_ft = ft_PID[0]/ft_control_norm[0]
        # kd_ft = np.sqrt((ft_PID_norm[ind[1]]/ft_err_position_norm[ind[1]])**2 - kp_ft**2)
        # print(kp_ft, kd_ft)
        if plot_graphs:
            title = 'analysis of error positions, action, feedback, and control signals from step ' + \
                    str(start_step_analysis) + ' to ' + str(number_steps)
            fig, ax1 = plt.subplots(2, 2)
            fig.suptitle(title)
            ax1[0][0].plot(time, action, time, control_input, time, control_input - action, time, err_position)
            ax1[0][0].legend(('action', 'control input', 'PID control', 'error position'))
            ax1[0][0].axhline(xmin=time[0], xmax=time[-1], color='k', linestyle='--', linewidth=1)
            ax1[0][1].plot(freq, ft_action_norm, freq, ft_PID_norm, freq, ft_control_norm, freq, ft_err_position_norm)
            ax1[0][1].legend(('action', 'PID', 'control', 'error position'))
            ax1[0][1].set_title('Norm of the DFTs')
            ax1[1][0].plot(freq, angle_diff_action_err, freq, angle_diff_PID_err, freq, angle_diff_control_err)
            ax1[1][0].legend(('action', 'PID', 'control'))
            ax1[1][0].set_title('Phase between the DFTs and the error position')
            ax1[1][1].plot(freq, angle_diff_action_PID, freq, angle_diff_control_PID)
            ax1[1][1].legend(('action-PID', 'control-PID'))
            ax1[1][1].set_title('Phase of the RL and the control signals with the PID')

            # f1 = plt.figure(3, 2)
            # title = 'DFT of action signal from step ' + str(start_step_analysis) + ' to step ' + str(number_steps)
            # f1.suptitle(title)
            # plt.plot(freq, ft_action_norm, freq, ft_action_real, freq, ft_action_ima)
            # plt.legend(('norm', 'real', 'imaginary'))
            # f2 = plt.figure()
            # title = 'DFT of action signal from step ' + str(start_step_analysis) + ' to step ' + str(number_steps)
            # f2.suptitle(title)
            # plt.plot(freq, ft_control_norm, freq, ft_control_real, freq, ft_control_ima)
            # plt.legend(('norm', 'real', 'imaginary'))
            # f3 = plt.figure()
            # title = 'DFT of action signal from step ' + str(start_step_analysis) + ' to step ' + str(number_steps)
            # f3.suptitle(title)
            # plt.plot(freq, ft_total_norm, freq, ft_total_real, freq, ft_total_ima)
            # plt.legend(('norm', 'real', 'imaginary'))
            # f4 = plt.figure()
            # plt.plot(freq, angle_action - angle_control, freq, angle_total - angle_control)
            # plt.legend(('angle action-control', 'angle total-control'))
            # f5 = plt.figure()
            # plt.plot(time, action, time, control_learning, time, control_learning-action)
            # plt.legend(('action', 'control input', 'control without RL'))
            # plt.axhline(color='k')

        if plot_graphs and A_DIM == 1:
            reward = infos['reward']
            positions = infos['positions']
            speed = infos['speeds']
            error_speeds_learning = infos['error_speed_learning']
            error_learning = infos['error_position_learning']
            analysis_action = infos['action']
            analysis_control = infos['control_input']
            u_i = infos['int_term']
            time = np.linspace(dt, number_steps * dt, number_steps)
            for i in range(number_steps - 1):
                for j in range(nb_agents):
                    if np.abs(positions[i + 1][j] - positions[i][j]) > d_ref * (nb_agents - 1):
                        positions[i + 1][j] = np.nan

            title = 'model number ' + str(model_number)
            title += ', stacked frames' if stacked_frames else 'no stacked frames'
            title += ', type action = ' + str(type_action)
            title += ', active agents ' + str(act_agents)
            f1, ax1 = plt.subplots(4, 2)
            f1.suptitle(title)
            ax1[0][0].plot(time, positions)
            ax1[0][0].set_title('Agent Positions')
            ax1[0][0].axis([0, number_steps * dt, 0, d_ref * nb_agents])
            ax1[0][1].plot(time, speed)
            ax1[0][1].set_title('Agent Speeds')
            ax1[1][0].plot(time, reward)
            ax1[1][0].set_title('Reward')
            ax1[1][1].plot(time, error_learning)
            ax1[1][1].set_title('Error Position of the Learning Agent')
            ax1[2][0].plot(time, error_speeds_learning)
            ax1[2][0].set_title('Error Speed of the Learning Agent')
            ax1[2][1].plot(time, analysis_action)
            ax1[2][1].set_title('Action')
            ax1[3][0].plot(time, analysis_control)
            ax1[3][0].set_title('Control Input of the Learning Agent')
            ax1[3][1].plot(time, u_i)
            ax1[3][1].set_title('Integral Term of the learning agent')

if temporal_analysis:

    mean_scores_action = scores_action.mean(axis=1)
    std_scores_action = scores_action.std(axis=1)

    mean_mean_scores_action = mean_scores_action.mean(axis=0)
    std_mean_scores_action = mean_scores_action.std(axis=0)
    mean_std_scores_action = std_scores_action.mean(axis=0)
    std_std_scores_action = std_scores_action.std(axis=0)

    mean_feedback_gains_action = feedback_gains_action.mean(axis=1)
    std_feedback_gains_action = feedback_gains_action.std(axis=1)

    mean_mean_feedback_gains_action = mean_feedback_gains_action.mean(axis=0)
    std_mean_feedback_gains_action = mean_feedback_gains_action.std(axis=0)
    mean_std_feedback_gains_action = std_feedback_gains_action.mean(axis=0)
    std_std_feedback_gains_action = std_feedback_gains_action.std(axis=0)

    mean_intercepts_action = intercepts_action.mean(axis=1)
    std_intercepts_action = intercepts_action.std(axis=1)

    mean_mean_intercepts_action = mean_intercepts_action.mean(axis=0)
    std_mean_intercepts_action = mean_intercepts_action.std(axis=0)
    mean_std_intercepts_action = std_intercepts_action.mean(axis=0)
    std_std_intercepts_action = std_intercepts_action.std(axis=0)

    mean_scores_control = scores_control.mean(axis=1)
    std_scores_control = scores_control.std(axis=1)

    mean_mean_scores_control = mean_scores_control.mean(axis=0)
    std_mean_scores_control = mean_scores_control.std(axis=0)
    mean_std_scores_control = std_scores_control.mean(axis=0)
    std_std_scores_control = std_scores_control.std(axis=0)

    mean_feedback_gains_control = feedback_gains_control.mean(axis=1)
    std_feedback_gains_control = feedback_gains_control.std(axis=1)

    mean_mean_feedback_gains_control = mean_feedback_gains_control.mean(axis=0)
    std_mean_feedback_gains_control = mean_feedback_gains_control.std(axis=0)
    mean_std_feedback_gains_control = std_feedback_gains_control.mean(axis=0)
    std_std_feedback_gains_control = std_feedback_gains_control.std(axis=0)

    mean_intercepts_control = intercepts_control.mean(axis=1)
    std_intercepts_control = intercepts_control.std(axis=1)

    mean_mean_intercepts_control = mean_intercepts_control.mean(axis=0)
    std_mean_intercepts_control = mean_intercepts_control.std(axis=0)
    mean_std_intercepts_control = std_intercepts_control.mean(axis=0)
    std_std_intercepts_control = std_intercepts_control.std(axis=0)

    f1, ax1 = plt.subplots(2, 2)
    ax1[0][0].plot(shifts, mean_mean_scores_action[:, 0], shifts, mean_mean_scores_action[:, 1])
    ax1[0][0].legend(('intercept', 'no intercept'))
    ax1[0][0].set_title('Mean of mean score action')

    ax1[0][1].plot(shifts, std_mean_scores_action[:, 0], shifts, std_mean_scores_action[:, 1])
    ax1[0][1].legend(('intercept', 'no intercept'))
    ax1[0][1].set_title('std of mean score action')

    ax1[1][0].plot(shifts, mean_std_scores_action[:, 0], shifts, mean_std_scores_action[:, 1])
    ax1[1][0].legend(('intercept', 'no intercept'))
    ax1[1][0].set_title('Mean of std score action')

    ax1[1][1].plot(shifts, std_std_scores_action[:, 0], shifts, std_std_scores_action[:, 1])
    ax1[1][1].legend(('intercept', 'no intercept'))
    ax1[1][1].set_title('std of std score action')

    f2, ax2 = plt.subplots(2, 2)
    ax2[0][0].plot(shifts, mean_mean_feedback_gains_action[:, 0], shifts, mean_mean_feedback_gains_action[:, 1],
                   shifts, mean_mean_intercepts_action)
    ax2[0][0].legend(('kp intercept', 'kp no intercept', 'kd intercept', 'kd no intercept', 'intercept'))
    ax2[0][0].set_title('Mean of mean feedback gains action')

    ax2[0][1].plot(shifts, std_mean_feedback_gains_action[:, 0], shifts, std_mean_feedback_gains_action[:, 1],
                   shifts, std_mean_intercepts_action)
    ax2[0][1].legend(('kp intercept', 'kp no intercept', 'kd intercept', 'kd no intercept', 'intercept'))
    ax2[0][1].set_title('std of mean feedback gains action')

    ax2[1][0].plot(shifts, mean_std_feedback_gains_action[:, 0], shifts, mean_std_feedback_gains_action[:, 1],
                   shifts, mean_std_intercepts_action)
    ax2[1][0].legend(('kp intercept', 'kp no intercept', 'kd intercept', 'kd no intercept', 'intercept'))
    ax2[1][0].set_title('Mean of std feedback gains action')

    ax2[1][1].plot(shifts, std_std_feedback_gains_action[:, 0], shifts, std_std_feedback_gains_action[:, 1],
                   shifts, std_std_intercepts_action)
    ax2[1][1].legend(('kp intercept', 'kp no intercept', 'kd intercept', 'kd no intercept', 'intercept'))
    ax2[1][1].set_title('std of std feedback gains action')

    print(mean_mean_feedback_gains_action[np.where(np.isclose(shifts, -2)), 1])

    f3, ax3 = plt.subplots(2, 2)
    ax3[0][0].plot(shifts, mean_mean_scores_control[:, 0], shifts, mean_mean_scores_control[:, 1])
    ax3[0][0].legend(('intercept', 'no intercept'))
    ax3[0][0].set_title('Mean of mean score control')

    ax3[0][1].plot(shifts, std_mean_scores_control[:, 0], shifts, std_mean_scores_control[:, 1])
    ax3[0][1].legend(('intercept', 'no intercept'))
    ax3[0][1].set_title('std of mean score control')

    ax3[1][0].plot(shifts, mean_std_scores_control[:, 0], shifts, mean_std_scores_control[:, 1])
    ax3[1][0].legend(('intercept', 'no intercept'))
    ax3[1][0].set_title('Mean of std score control')

    ax3[1][1].plot(shifts, std_std_scores_control[:, 0], shifts, std_std_scores_control[:, 1])
    ax3[1][1].legend(('intercept', 'no intercept'))
    ax3[1][1].set_title('std of std score control')

    f4, ax4 = plt.subplots(2, 2)
    ax4[0][0].plot(shifts, mean_mean_feedback_gains_control[:, 0], shifts, mean_mean_feedback_gains_control[:, 1],
                   shifts, mean_mean_intercepts_control)
    ax4[0][0].legend(('kp intercept', 'kp no intercept', 'kd intercept', 'kd no intercept', 'intercept'))
    ax4[0][0].set_title('Mean of mean feedback gains control')

    ax4[0][1].plot(shifts, std_mean_feedback_gains_control[:, 0], shifts, std_mean_feedback_gains_control[:, 1],
                   shifts, std_mean_intercepts_control)
    ax4[0][1].legend(('kp intercept', 'kp no intercept', 'kd intercept', 'kd no intercept', 'intercept'))
    ax4[0][1].set_title('std of mean feedback gains control')

    ax4[1][0].plot(shifts, mean_std_feedback_gains_control[:, 0], shifts, mean_std_feedback_gains_control[:, 1],
                   shifts, mean_std_intercepts_control)
    ax4[1][0].legend(('kp intercept', 'kp no intercept', 'kd intercept', 'kd no intercept', 'intercept'))
    ax4[1][0].set_title('Mean of std feedback gains control')

    ax4[1][1].plot(shifts, std_std_feedback_gains_control[:, 0], shifts, std_std_feedback_gains_control[:, 1],
                   shifts, std_std_intercepts_control)
    ax4[1][1].legend(('kp intercept', 'kp no intercept', 'kd intercept', 'kd no intercept', 'intercept'))
    ax4[1][1].set_title('std of std feedback gains control')

    print('gains at shift -7 = ', mean_mean_feedback_gains_control[np.where(np.isclose(shifts, -7)), 1])
    print('gains at shift -6 = ', mean_mean_feedback_gains_control[np.where(np.isclose(shifts, -6)), 1])
    print('gains at shift -5 = ', mean_mean_feedback_gains_control[np.where(np.isclose(shifts, -5)), 1])


    plt.show()

if plot_graphs:
    plt.show()
