import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
# from tensorboardX import SummaryWriter

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import time
from datetime import datetime
import matplotlib.pyplot as plt
from random import randint
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

from restaurant import *
from agent_global import *
from DQN import *


def concatenate_state(all_tables, all_agents, all_groups_of_people):
	state = []
	for elem, each_table in (all_tables.items()):
		state.extend(each_table)
	for elem, each_agent in (all_agents.items()):
		state.extend(each_agent)
	for elem, each_group in (all_groups_of_people.items()):
		state.extend(each_group)
	state_array = np.array(state)
	return state_array

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

if __name__ == "__main__":
	torch.set_default_tensor_type('torch.DoubleTensor')
	number_of_tables = 2
	number_of_agents = 2
	grid_dim_x = 5
	grid_dim_y = 5
	max_number_of_groups = 3
	number_of_training_loops = 2
	number_of_episodes = 4
	number_of_steps = 25
	number_of_stat_runs = 10

	batch_size = 100
	gamma = 0.9
	eps_start = 0.9
	eps_end = 0.05
	eps_decay = 150000
	capacity = 100000
	learning_rate = 1e-3
	weight_decay = 1e-4

	### DQN
	Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
	steps_done = 0

	restaurant = Restaurant(number_of_tables, number_of_agents, max_number_of_groups, grid_dim_x, grid_dim_y, number_of_steps)

	# uncomment this to start visualization
	# restaurant.initialise_visualization()

	all_the_agents = dict()

	# plot_rewards = []    
	# plot_successes = []
	# plot_N = 0
	
	stat_n_iter = list()
	all_stat_system_rewards = list()
	all_stat_system_successes = list()
	all_stat_agent_rewards = list()
	all_stat_agent_successes = list()

	for stat_run in range(number_of_stat_runs):

		stat_system_rewards = list()
		stat_system_successes = list()
		stat_agent_rewards = [list() for _ in range(number_of_agents)]
		stat_agent_successes = [list() for _ in range(number_of_agents)]

		steps_done = 0

		for elem in range(number_of_agents):
			all_the_agents[elem] = Agent(number_of_tables, number_of_agents, max_number_of_groups, grid_dim_x, grid_dim_y, batch_size, gamma, eps_start, eps_end, eps_decay, learning_rate, weight_decay, capacity)
			# all_the_agents[elem].give_memory(ReplayMemory(capacity, Transition))

		global_agent_vdn = global_agent()
		overall_memory = ReplayMemory(capacity,Transition)
		global_agent_vdn.give_memory(overall_memory)
		global_agent_vdn.all_the_agents = all_the_agents
		global_agent_vdn.initialize_hyperparam()

		for training_loop in range(number_of_training_loops):

			for episode in range(number_of_episodes):

				# reset the environment
				restaurant.reset()

				# Get initial observation from the enivorment
				all_table, all_agents, all_people = restaurant.get_observation()
				state = concatenate_state(all_table, all_agents, all_people)
				
				episode_reward = 0
				episode_successes = 0

				agent_wise_rewards = [0] * number_of_agents
				agent_wise_successes = [0] * number_of_agents
				for each_step in range(number_of_steps):

					actions_of_all_agents = [0] * number_of_agents
					actions_of_all_agents = np.asarray(actions_of_all_agents)
					for elem, each_agent in all_the_agents.items():
						policy_net_output = each_agent.policy_net_agent(torch.from_numpy(state))
						action_temp = each_agent.get_action(policy_net_output, steps_done)
						actions_of_all_agents[elem] = action_temp.numpy()
					
					restaurant.step(actions_of_all_agents)
					steps_done += 1
					
					# Get updated observation from the environment
					next_all_table, next_all_agents, next_all_people = restaurant.get_observation()
					next_state = concatenate_state(next_all_table, next_all_agents, next_all_people)
					
					# Get reward for the action taken
					system_reward = restaurant.get_system_reward()
					episode_reward += system_reward
					
					system_reward = np.asarray(system_reward,dtype=float)
					episode_successes = restaurant.get_sucessess()

					for elem, each_agent in all_the_agents.items():
						agent_reward = restaurant.get_local_reward(elem)
						agent_wise_rewards[elem] += agent_reward
						agent_success = restaurant.get_agent_sucessess(elem)
						agent_wise_successes[elem] = agent_success

					overall_memory.push(torch.from_numpy(state).unsqueeze(0),torch.from_numpy(np.asarray([actions_of_all_agents], dtype=int)).unsqueeze(0),torch.from_numpy(next_state).unsqueeze(0),torch.from_numpy(system_reward).unsqueeze(0))

					# modify state for the next step
					state = next_state

				# if episode % 10 == 0: 
				# 	print("Training Loop: {} ; Episode: {} ; Reward : {} ; Successes: {}".format(training_loop, episode, episode_reward, episode_successes))  

				n_iter = number_of_episodes * training_loop + episode

				stat_system_rewards.append(episode_reward)
				stat_system_successes.append(episode_successes)

				if not stat_run:
					stat_n_iter.append(n_iter)

				for elem in range(number_of_agents):
					agent_reward = agent_wise_rewards[elem]
					agent_success = agent_wise_successes[elem]
					stat_agent_rewards[elem].append(agent_reward)
					stat_agent_successes[elem].append(agent_success)

			# optimize_model()
			global_agent_vdn.optimize_model_global(batch_size,gamma)
			
			# for elem,each_agent in all_the_agents.items():
				# each_agent.target_net_agent.load_state_dict(each_agent.policy_net_agent.state_dict())

		all_stat_system_rewards.append(stat_system_rewards)
		all_stat_system_successes.append(stat_system_successes)
		all_stat_agent_rewards.append(stat_agent_rewards)
		all_stat_agent_successes.append(stat_agent_successes)

	################################################################
	##### Numpy arrays - SAVE THESE to the appropriate folders #####
	################################################################

	all_stat_system_rewards_mean = np.mean(np.array(all_stat_system_rewards), axis=0)
	all_stat_system_rewards_std_dev = np.std(np.array(all_stat_system_rewards), axis=0)

	all_stat_system_successes_mean = np.mean(np.array(all_stat_system_successes), axis=0)
	all_stat_system_successes_std_dev = np.std(np.array(all_stat_system_successes), axis=0)

	all_stat_agent_rewards_mean = np.mean(np.array(all_stat_agent_rewards), axis=0)
	all_stat_agent_rewards_std_dev = np.std(np.array(all_stat_agent_rewards), axis=0)

	all_stat_agent_successes_mean = np.mean(np.array(all_stat_agent_successes), axis=0)
	all_stat_agent_successes_std_dev = np.std(np.array(all_stat_agent_successes), axis=0)

	####################################
	########## SAVE THE MODEL ##########
	####################################

	# for elem, each_agent in all_the_agents.items():
	# 	model_path = '/home/abhijeet/Pytorch_models/dqn_' + str(elem)
	# 	# model_path = '/home/risheek/Pytorch_workspace/dqn_'+ str(elem)
	# 	torch.save(each_agent.policy_net_agent.state_dict(), model_path)


	#####################################
	########## IGNORE ALL THIS ##########
	#####################################

	# print("TOTAL: stat_system_rewards: {} ; stat_system_successes: {} ; stat_agent_rewards: {} ; stat_agent_successes: {} ; stat_n_iter: {}".format(len(all_stat_system_rewards), len(all_stat_system_successes), len(all_stat_agent_rewards), len(all_stat_agent_successes), len(stat_n_iter)))

	# average_of_system_rewards = [0 for _ in range(len(stat_n_iter))]
	# average_of_system_successes = [0 for _ in range(len(stat_n_iter))]
	# average_of_agent_rewards = [[0 for _ in range(len(stat_n_iter))] for _ in range(number_of_agents)]
	# average_of_agent_successes = [[0 for _ in range(len(stat_n_iter))] for _ in range(number_of_agents)]

	# for stat_run in range(number_of_stat_runs):
	# 	for n_iter in stat_n_iter:
	# 		average_of_system_rewards[n_iter] += all_stat_system_rewards[stat_run][n_iter]
	# 		average_of_system_successes[n_iter] += all_stat_system_successes[stat_run][n_iter]
	# 		for agent in range(number_of_agents):
	# 			average_of_agent_rewards[agent][n_iter] += all_stat_agent_rewards[stat_run][agent][n_iter]
	# 			average_of_agent_successes[agent][n_iter] += all_stat_agent_successes[stat_run][agent][n_iter]

	# new_average_of_system_rewards = [ i / number_of_stat_runs for i in average_of_system_rewards]
	# new_average_of_system_successes = [ i / number_of_stat_runs for i in average_of_system_successes]
	# new_average_of_agent_rewards = [[0 for _ in range(len(stat_n_iter))] for _ in range(number_of_agents)]
	# new_average_of_agent_successes = [[0 for _ in range(len(stat_n_iter))] for _ in range(number_of_agents)]

	# for agent_id, rewards in enumerate(average_of_agent_rewards):
	# 	for elem, value in enumerate(rewards):
	# 		new_average_of_agent_rewards[agent_id][elem] = average_of_agent_rewards[agent_id][elem] / number_of_stat_runs

	# for agent_id, successes in enumerate(average_of_agent_successes):
	# 	for elem, value in enumerate(successes):
	# 		new_average_of_agent_successes[agent_id][elem] = average_of_agent_successes[agent_id][elem] / number_of_stat_runs

	# # Tensorboard
	# dt = datetime.now()
	# dt_round_microsec = round(dt.microsecond/1000)
	# # dt_round_microsec = dt
	# # filename = 'runs/ChairBot_DQN_' + str(number_of_agents) + '_agents_and_' + str(max_number_of_groups) + '_groups_in_' + str(grid_dim_x) + '_X_' + str(grid_dim_y) + '_in_' + str(number_of_training_loops) + '_training_loops' + str(dt_round_microsec)
	# filename = './runs/ChairBot_VDN_' + str(dt_round_microsec)

	# tensorboard_writer = SummaryWriter(filename)
	
	# tensorboard_writer.add_text('env-params/number_of_tables', str(number_of_tables))
	# tensorboard_writer.add_text('env-params/number_of_agents', str(number_of_agents))
	# tensorboard_writer.add_text('env-params/grid_dim_x', str(grid_dim_x))
	# tensorboard_writer.add_text('env-params/grid_dim_y', str(grid_dim_y))
	# tensorboard_writer.add_text('env-params/max_number_of_groups', str(max_number_of_groups))
	# tensorboard_writer.add_text('env-params/number_of_training_loops', str(number_of_training_loops))
	# tensorboard_writer.add_text('env-params/number_of_episodes', str(number_of_episodes))
	# tensorboard_writer.add_text('env-params/number_of_steps', str(number_of_steps))
	# tensorboard_writer.add_text('hyper-params/batch_size', str(batch_size))
	# tensorboard_writer.add_text('hyper-params/gamma', str(gamma))
	# tensorboard_writer.add_text('hyper-params/eps_start', str(eps_start))
	# tensorboard_writer.add_text('hyper-params/eps_end', str(eps_end))
	# tensorboard_writer.add_text('hyper-params/eps_decay', str(eps_decay))
	# tensorboard_writer.add_text('hyper-params/learning_rate', str(learning_rate))
	# tensorboard_writer.add_text('hyper-params/weight_decay', str(weight_decay))

	# for n_iter in stat_n_iter:
	# 	tensorboard_writer.add_scalar('system/rewards', new_average_of_system_rewards[n_iter], n_iter)
	# 	tensorboard_writer.add_scalar('system/successes', new_average_of_system_successes[n_iter], n_iter)
	# 	for elem in range(number_of_agents):
	# 		tensorboard_writer.add_scalar('agent' +str(elem) +'/rewards', new_average_of_agent_rewards[elem][n_iter], n_iter)
	# 		tensorboard_writer.add_scalar('agent' +str(elem) +'/successes', new_average_of_agent_successes[elem][n_iter], n_iter)

	# for elem, each_agent in all_the_agents.items():
	# 	model_path = '/home/abhijeet/Pytorch_models/dqn_' + str(elem)
	# 	# model_path = '/home/risheek/Pytorch_workspace/dqn_'+ str(elem)
	# 	torch.save(each_agent.policy_net_agent.state_dict(), model_path)

	# tensorboard_writer.export_scalars_to_json("./all_scalars.json")
	# tensorboard_writer.close()

	# print("all_stat_system_rewards: {}".format(all_stat_system_rewards))
	# a =  np.array(all_stat_system_rewards)
	# mean_a = np.mean(a, axis=0)
	# var_a = np.var(a, axis=0)
	# std_a = np.std(a, axis=0)
	# # print("all_stat_system_rewards Mean: {}".format(mean_a))
	# # print("all_stat_system_rewards Var: {}".format(var_a))
	# # print("all_stat_system_rewards Std: {}".format(std_a))

	# plot_reward_text = "Plotting rewards over all the episodes | For " + str(number_of_agents) + " chairbot in 5X5 env"
	# plot_successes_text = "Plotting successes over all the episodes | For " + str(number_of_agents) + " chairbot in 5X5 env"

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.set_title(plot_reward_text)
	# ax.set_xlabel('Number of episodes')
	# ax.set_ylabel('System reward in each episode')

	# x = np.linspace(0, 30, 30)
	# y = np.sin(x/6*np.pi)
	# error = np.random.normal(0.1, 0.02, size=y.shape) +.1
	# y += np.random.normal(0, 0.1, size=y.shape)

	# # plt.plot(x, y, 'k', color='#CC4F1B')
	# smoothening_factor = 1

	# plt.plot(stat_n_iter, smooth(mean_a,smoothening_factor), 'm-', label='Rewards', color='#1B2ACC')
	# plt.fill_between(stat_n_iter, smooth(mean_a-std_a,smoothening_factor), smooth(mean_a+std_a,smoothening_factor),
	#     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
	# plt.legend()
	# plt.show()