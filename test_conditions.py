import matplotlib.pyplot as plt
import numpy as np
from Environment.LoopFollowingAgents import LoopFollowingAgents

number_episodes = 10
number_steps = 1000
N = 6
random_start = 1
delay = 0.1
time_step = 0.025
d_ref = 2
local_dt = 0.0125
gains_reward = [1, 1, 0.25, 10]
u_i = False


# number_episodes = 1
# number_steps = 10

def mean(array):
    return sum(array) / len(array)


def variance(array):
    m = mean(array)
    return np.sqrt(mean([(x - m) ** 2 for x in array]))


reward_no_delay = []
nor_reward_no_delay = []
reward_no_action = []
reward_no_control = []
a_a = [0]

env_no_control = LoopFollowingAgents(available_obs=0, mode_action=2, type_action=[True, True, True], number_agents=N,
                                     random_start=random_start, delay=delay, dt=time_step, d_ref=d_ref,
                                     active_agents=a_a, local_time_step=local_dt, gains_reward=gains_reward, u_i=u_i,
                                     number_steps=number_steps)
env_no_action = LoopFollowingAgents(available_obs=0, mode_action=2, type_action=[True, True, True], number_agents=N,
                                    random_start=random_start, delay=delay, dt=time_step, d_ref=d_ref,
                                    active_agents=a_a, local_time_step=local_dt, gains_reward=gains_reward, u_i=u_i,
                                    number_steps=number_steps)
env_no_delay = LoopFollowingAgents(available_obs=0, mode_action=2, type_action=[True, True, True], number_agents=N,
                                   random_start=random_start, delay=0, dt=time_step, d_ref=d_ref, active_agents=a_a,
                                   local_time_step=local_dt, gains_reward=gains_reward, u_i=u_i,
                                   number_steps=number_steps)

[kp, ki, kd] = env_no_control.feedback_params()

action = [-kp, -ki, -kd]*len(a_a)
print('action = ', action)

for episode in range(0, number_episodes):
    print(episode)
    [obs1, initial_state] = env_no_control.reset(get_init_state=True, save_traj=True)
    obs2 = env_no_action.reset(initial_state, save_traj=True)
    obs3 = env_no_delay.reset(initial_state, save_traj=True)
    # initial_state = None
    # [obs1, initial_state] = env_no_control.reset(get_init_state = True)
    # obs2 = env_no_action.reset(initial_state)
    # obs3 = env_no_delay.reset(initial_state)

    sum_reward1 = 0
    sum_reward2 = 0
    sum_reward3 = 0

    for step in range(number_steps):
        _, reward1, _, infos1 = env_no_control.step(action, step)
        _, reward2, _, infos2 = env_no_action.step([0, 0, 0] * len(a_a), step)
        _, reward3, _, infos3 = env_no_delay.step([0, 0, 0] * len(a_a), step)
        sum_reward1 += reward1
        sum_reward2 += reward2
        sum_reward3 += reward3
        # print('1 = ',sum_reward1)
        # print('2 = ',sum_reward2)
        # print('3 = ',sum_reward3)

    reward_no_control.append(sum_reward1)
    reward_no_action.append(sum_reward2)
    reward_no_delay.append(sum_reward3)
    # print(sum_reward1)
    # print(sum_reward2)
    # print(sum_reward3)

    nor_reward_no_delay.append((sum_reward3 - sum_reward2) / (sum_reward1 - sum_reward2))
    if reward_no_control[episode] > 20000:
        break

episodes = range(0, number_episodes)
print('mean no control = ', mean(reward_no_control))
print('mean no action = ', mean(reward_no_action))
print('mean no delay = ', mean(reward_no_delay))

f = plt.figure(1)
plt.plot(episodes, reward_no_delay, episodes, reward_no_control, episodes, reward_no_action)
plt.legend(('no delay', 'no control', 'no action'))
f.show()

reward1 = infos1['reward']
positions1 = infos1['positions']
speed1 = infos1['speeds']
error_speeds_learning1 = infos1['error_speed_learning']
error_learning1 = infos1['error_position_learning']
action1 = infos1['action']
control_learning1 = infos1['control_input']
u_i1 = infos1['int_term']

# print('sum reward = ', np.sum(reward))

t = np.linspace(time_step, number_steps * time_step, number_steps)
# t = np.linspace(0,number_steps-1,number_steps)
for i in range(number_steps - 1):
    for j in range(N):
        if np.abs(positions1[i + 1][j] - positions1[i][j]) > d_ref * (N - 1):
            positions1[i + 1][j] = np.nan

f1, ax1 = plt.subplots(4, 2)
ax1[0][0].plot(t, positions1)
ax1[0][0].set_title('Agent Positions')
ax1[0][1].plot(t, speed1)
ax1[0][1].set_title('Agent Speeds')
ax1[1][0].plot(t, reward1)
ax1[1][0].set_title('Reward')
ax1[1][1].plot(t, error_learning1)
ax1[1][1].set_title('Error Position of the Learning Agent')
ax1[2][0].plot(t, error_speeds_learning1)
ax1[2][0].set_title('Error Speed of the Learning Agent')
ax1[2][1].plot(t, action1)
ax1[2][1].set_title('action')
ax1[3][0].plot(t, control_learning1)
ax1[3][0].set_title('Control Input of the Learning Agent')
ax1[3][1].plot(t, u_i1)
ax1[3][1].set_title('integral term of the learning agent')

ax1[0][0].axis([0, number_steps * time_step, 0, d_ref * N])
f1.show()

reward2 = infos2['reward']
positions2 = infos2['positions']
speed2 = infos2['speeds']
error_speeds_learning2 = infos2['error_speed_learning']
error_learning2 = infos2['error_position_learning']
action2 = infos2['action']
control_learning2 = infos2['control_input']
u_i2 = infos2['int_term']

# print('sum reward = ', np.sum(reward))

t = np.linspace(time_step, number_steps * time_step, number_steps)
# t = np.linspace(0,number_steps-1,number_steps)
for i in range(number_steps - 1):
    for j in range(N):
        if np.abs(positions2[i + 1][j] - positions2[i][j]) > d_ref * (N - 1):
            positions2[i + 1][j] = np.nan

f2, ax2 = plt.subplots(4, 2)
ax2[0][0].plot(t, positions2)
ax2[0][0].set_title('Agent Positions')
ax2[0][1].plot(t, speed2)
ax2[0][1].set_title('Agent Speeds')
ax2[1][0].plot(t, reward2)
ax2[1][0].set_title('Reward')
ax2[1][1].plot(t, error_learning2)
ax2[1][1].set_title('Error Position of the Learning Agent')
ax2[2][0].plot(t, error_speeds_learning2)
ax2[2][0].set_title('Error Speed of the Learning Agent')
ax2[2][1].plot(t, action2)
ax2[2][1].set_title('action')
ax2[3][0].plot(t, control_learning2)
ax2[3][0].set_title('Control Input of the Learning Agent')
ax2[3][1].plot(t, u_i2)
ax2[3][1].set_title('integral term of the learning agent')

ax2[0][0].axis([0, number_steps * time_step, 0, d_ref * N])

f2.show()

reward3 = infos3['reward']
positions3 = infos3['positions']
speed3 = infos3['speeds']
error_speeds_learning3 = infos3['error_speed_learning']
error_learning3 = infos3['error_position_learning']
action3 = infos3['action']
control_learning3 = infos3['control_input']
u_i3 = infos3['int_term']

# print('sum reward = ', np.sum(reward))

t = np.linspace(time_step, number_steps * time_step, number_steps)
# t = np.linspace(0,number_steps-1,number_steps)
for i in range(number_steps - 1):
    for j in range(N):
        if np.abs(positions3[i + 1][j] - positions3[i][j]) > d_ref * (N - 1):
            positions3[i + 1][j] = np.nan

f3, ax3 = plt.subplots(4, 2)
ax3[0][0].plot(t, positions3)
ax3[0][0].set_title('Agent Positions')
ax3[0][1].plot(t, speed3)
ax3[0][1].set_title('Agent Speeds')
ax3[1][0].plot(t, reward3)
ax3[1][0].set_title('Reward')
ax3[1][1].plot(t, error_learning3)
ax3[1][1].set_title('Error Position of the Learning Agent')
ax3[2][0].plot(t, error_speeds_learning3)
ax3[2][0].set_title('Error Speed of the Learning Agent')
ax3[2][1].plot(t, action3)
ax3[2][1].set_title('action')
ax3[3][0].plot(t, control_learning3)
ax3[3][0].set_title('Control Input of the Learning Agent')
ax3[3][1].plot(t, u_i3)
ax3[3][1].set_title('integral term of the learning agent')

ax3[0][0].axis([0, number_steps * time_step, 0, d_ref * N])

f3.show()

plt.show()
