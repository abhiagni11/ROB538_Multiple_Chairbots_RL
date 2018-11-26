import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
# from tensorboardX import SummaryWriter

import math
import os
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


folder_save_config_in = 'Models/'
configuration_name = 'n_agents_' + str(number_of_agents) + '__grid_' + str(grid_dim_x) + 'x' + str(grid_dim_y) + '__groups_' + str(max_number_of_groups) + '/'
model_type = 'VDN/'
net_folder_dir = folder_save_config_in + configuration_name + model_type

if __name__ == "__main__":
	torch.set_default_tensor_type('torch.DoubleTensor')
	number_of_tables = 6
	number_of_agents = 6
	grid_dim_x = 8
	grid_dim_y = 8
	max_number_of_groups = 9
	number_of_training_loops = 120
	number_of_episodes = 40
	number_of_steps = 300
	number_of_stat_runs = 1

	batch_size = 100
	gamma = 0.9
	eps_start = 0
	eps_end = 0
	eps_decay = 140
	capacity = 2500
	learning_rate = 1e-3
	weight_decay = 1e-4

	### DQN
	Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
	steps_done = 0

	restaurant = Restaurant(number_of_tables, number_of_agents, max_number_of_groups, grid_dim_x, grid_dim_y, number_of_steps)

	# uncomment this to start visualization
	# restaurant.initialise_visualization()

	all_the_agents = dict()

	
	stat_n_iter = list()
	all_stat_system_rewards = list()
	all_stat_system_successes = list()
	all_stat_agent_rewards = list()
	all_stat_agent_successes = list()

	
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

	for elem, each_agent in all_the_agents.items():
		each_agent.load_state_dict(torch.load(net_folder_dir+'model'+str(elem)))

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

		# for elem, each_agent in all_the_agents.items():
		# 	agent_reward = restaurant.get_local_reward(elem)
		# 	agent_wise_rewards[elem] += agent_reward
		# 	agent_success = restaurant.get_agent_sucessess(elem)
		# 	agent_wise_successes[elem] = agent_success

