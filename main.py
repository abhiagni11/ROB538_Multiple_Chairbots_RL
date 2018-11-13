import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

# torch.set_default_tensor_type('torch.FloatTensor')
torch.set_default_tensor_type('torch.DoubleTensor')

import time
import matplotlib.pyplot as plt
from random import randint
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

from restaurant import *
from agent import *
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


if __name__ == "__main__":

	number_of_tables = 3
	number_of_agents = 3
	grid_dim_x = 5
	grid_dim_y = 5
	max_number_of_groups = 1
	number_of_training_loops = 20
	number_of_episodes = 40
	number_of_steps = 500

	batch_size = 100
	gamma = 0.9
	eps_start = 0.9
	eps_end = 0.05
	eps_decay = 20000
	capacity = 100000

	### DQN
	Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
	steps_done = 0

	restaurant = Restaurant(number_of_tables, number_of_agents, max_number_of_groups, grid_dim_x, grid_dim_y, number_of_steps)
	all_the_agents = dict()


	plot_rewards = []    
	plot_successes = []
	plot_N = 0

	for elem in range(number_of_agents):
		all_the_agents[elem] = Agent(number_of_tables, number_of_agents, max_number_of_groups, grid_dim_x, grid_dim_y, batch_size, gamma, eps_start, eps_end, eps_decay, capacity)
		all_the_agents[elem].give_memory(ReplayMemory(capacity, Transition))

	for training_loop in range(number_of_training_loops):
		# print('training_loop',training_loop)
		
		for episode in range(number_of_episodes):

			# print("episode",episode)
			# reset the environment
			restaurant.reset()

			# Get initial observation from the enivorment
			all_table, all_agents, all_people = restaurant.get_observation()
			state = concatenate_state(all_table, all_agents, all_people)
			
			episode_reward = 0
			episode_successes = 0
			for each_step in range(number_of_steps):

				actions_of_all_agents = [0] * number_of_agents
				actions_of_all_agents = np.asarray(actions_of_all_agents)
				for elem, each_agent in all_the_agents.items():
					# actions_of_all_agents[elem] = randint(0,4)
					policy_net_output = each_agent.policy_net_agent(torch.from_numpy(state))
					# print("elem is {} and policy_net_agent is {}".format(elem, policy_net_output))
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
				
				# print('state',state)
				# print('actions_of_all_agents]0',np.asarray([actions_of_all_agents[0]],dtype=float))
				for elem, each_agent in all_the_agents.items():
					all_the_agents[elem].memory.push(torch.from_numpy(state).unsqueeze(0),torch.from_numpy(np.asarray([actions_of_all_agents[elem]],dtype=int)).unsqueeze(0),torch.from_numpy(next_state).unsqueeze(0),torch.from_numpy(system_reward).unsqueeze(0))

				# modify state for the next step
				state = next_state

				# restaurant.visualize_restaurant()
				# time.sleep(0.001)

			if episode % 1 == 0: 
				# print("Training Loop: {} ; Episode: {} ; Reward : {} ; Successes: {}".format(training_loop, episode, episode_reward, episode_successes))
				plot_rewards.append(episode_reward)        
				plot_successes.append(episode_successes)
				plot_N += 1     

		# optimize_model()
		for elem, each_agent in all_the_agents.items():
			all_the_agents[elem].optimize_model()
	
	for elem, each_agent in all_the_agents.items():
		model_path = '/home/abhijeet/Pytorch_models/dqn_' + str(elem)
		torch.save(each_agent.policy_net_agent.state_dict(), model_path)


	plot_N = list(range(plot_N))

	plot_reward_text = "Plotting rewards over all the episodes | For " + str(number_of_agents) + " chairbot in 5X5 env"
	plot_successes_text = "Plotting successes over all the episodes | For " + str(number_of_agents) + " chairbot in 5X5 env"

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(plot_reward_text)
	ax.set_xlabel('Number of episodes')
	ax.set_ylabel('System reward in each episode')
	plt.plot(plot_N, plot_rewards, 'm-', label='Rewards')
	plt.legend()
	# plt.tight_layout()
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(plot_successes_text)
	ax.set_xlabel('Number of episodes')
	ax.set_ylabel('Successes in each episode')
	plt.plot(plot_N, plot_successes, 'b-', label='Successes')
	plt.legend()
	# plt.tight_layout()
	plt.show()
