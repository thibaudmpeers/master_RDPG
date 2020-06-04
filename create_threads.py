import os
from threading import Thread
from itertools import product


class Task(Thread):
    def __init__(self, command_line):
        Thread.__init__(self)
        self.command_line = command_line

    def run(self):
        os.system(self.command_line)
        # print(self.command_line)


number_steps = 600
number_episode = 5001
mode_action = 1
type_action = [True, False]

# available_obs = [0, 1, 2]
available_obs = [1]
# available_obs = [1, 0]
number_agents = [6]
observation_front = [True]
# observation_front = [True]
# alpha = [1e-7]
# alpha = [1e-4, 1e-5, 1e-3, 1e-2]
# alpha_actor = [1e-4, 1e-5, 1e-6]
# alpha_critic = [1e-4, 1e-5, 1e-6]
# alpha_actor = [1e-4, 1e-5, 1e-6]
alpha_actor = [1e-5]
alpha_critic = [1e-5]

# adap_alpha_actor = [1e-5, 1e-6, 1e-7]
# adap_alpha_critic = [1e-5, 1e-6, 1e-7]
adap_alpha_actor = [1e-7]
adap_alpha_critic = [1e-6]
# tau = [1e-1, 1e-2]
tau = [1e-2]
decay_tau = [0]
# extended_inputs = [0, 1, 2]
extended_inputs = [2]

threads = []

for n, alp_actor, alp_critic, t, k, obs, input_dim, adap_alp_actor, adap_alp_critic \
        in product(number_agents, alpha_actor, alpha_critic, tau, decay_tau, available_obs, extended_inputs,
                   adap_alpha_actor, adap_alpha_critic):
    command = 'python3 main.py -s ' + str(number_steps)
    command += ' -e ' + str(number_episode)
    command += ' -a_o ' + str(obs)
    command += ' -inputs ' + str(input_dim)
    if mode_action == 0:
        command += ' -a_ref'
    elif mode_action == 1:
        command += ' -a_u'
        command += ' -e_p' if type_action[0] else ''
        command += ' -e_i' if type_action[1] else ''
    else:
        command += ' -a_k'
        command += ' -e_p' if type_action[0] else ''
        command += ' -e_i' if type_action[1] else ''
        command += ' -e_d' if type_action[2] else ''
    command += ' -n ' + str(n)
    command += ' -l_actor ' + str(alp_actor)
    command += ' -l_critic ' + str(alp_critic)
    command += ' -l_adap_critic ' + str(adap_alp_critic)
    command += ' -l_adap_actor ' + str(adap_alp_actor)
    command += ' -tau ' + str(t)
    command += ' -k_tau ' + str(k)
    # command += ' -gpu'
    threads.append(Task(command))

# for n in number_agents:
#     for alp_actor in alpha_actor:
#         for alp_critic in alpha_critic:
#             for t in tau:
#                 for k in decay_tau:
#                     for obs in available_obs:
#                         for input_dim in extended_inputs:
#                             command = 'python3 main.py -s ' + str(number_steps)
#                             command += ' -e ' + str(number_episode)
#                             command += ' -a_o ' + str(obs)
#                             command += ' -inputs ' + str(input_dim)
#                             if mode_action == 0:
#                                 command += ' -a_ref'
#                             elif mode_action == 1:
#                                 command += ' -a_u'
#                                 command += ' -e_p' if type_action[0] else ''
#                                 command += ' -e_i' if type_action[1] else ''
#                             else:
#                                 command += ' -a_k'
#                                 command += ' -e_p' if type_action[0] else ''
#                                 command += ' -e_i' if type_action[1] else ''
#                                 command += ' -e_d' if type_action[2] else ''
#                             command += ' -n ' + str(n)
#                             command += ' -l_actor ' + str(alp_actor)
#                             command += ' -l_critic ' + str(alp_critic)
#                             command += ' -tau ' + str(t)
#                             command += ' -k_tau ' + str(k)
#                             command += ' -gpu'
#                             threads.append(Task(command))

max_threads = 3

number_experiments = len(threads)

print('number of experiments: ', number_experiments)


for j in range(number_experiments):
    threads[j].start()
    if j % max_threads == max_threads - 1:
        for i in range(max_threads):
            threads[j-i].join()
    elif j == number_experiments - 1:
        for i in range(number_experiments % max_threads):
            threads[j - i].join()

plot_thread = Task('python3 plot_results.py')
plot_thread.start()
plot_thread.join()
