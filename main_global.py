import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from tensorboardX import SummaryWriter

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


if __name__ == "__main__":

	torch.set_default_tensor_type('torch.DoubleTensor')
	number_of_tables = 2
	number_of_agents = 2
	grid_dim_x = 5
	grid_dim_y = 5
	max_number_of_groups = 2
	number_of_training_loops = 10
	number_of_episodes = 40
	number_of_steps = 300

	### eps_decay defines the rate of decay in exploration
	## eps-decacy, gamma.;;	
	
	batch_size = 100
	gamma = 0.9
	eps_start = 0.9
	eps_end = 0.05
	eps_decay = 200000
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


	plot_rewards = []    
	plot_successes = []
	plot_N = 0
	
	# dt = datetime.now()
	# dt_round_microsec = round(dt.microsecond/1000)
	# filename = 'runs/ChairBot_DQN_' + str(number_of_agents) + '_agents_and_' + str(max_number_of_groups) + '_groups_in_' + str(grid_dim_x) + '_X_' + str(grid_dim_y) + '_in_' + str(number_of_training_loops) + '_training_loops' + str(dt_round_microsec)
	# tensorboard_writer = SummaryWriter(filename)
	
	for elem in range(number_of_agents):
		all_the_agents[elem] = Agent(number_of_tables, number_of_agents, max_number_of_groups, grid_dim_x, grid_dim_y, batch_size, gamma, eps_start, eps_end, eps_decay, learning_rate, weight_decay, capacity)
		# all_the_agents[elem].give_memory(ReplayMemory(capacity, Transition))

	print('all_the_agents')
	print(all_the_agents)



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
			for each_step in range(number_of_steps):

				actions_of_all_agents = [0] * number_of_agents
				actions_of_all_agents = np.asarray(actions_of_all_agents)
				for elem, each_agent in all_the_agents.items():
					policy_net_output = each_agent.policy_net_agent(torch.from_numpy(state))
					action_temp = each_agent.get_action(policy_net_output, steps_done)
					# print('action_temp \n',action_temp)
					actions_of_all_agents[elem] = action_temp.numpy()
				
				print('actions_of_all_agents')
				print(actions_of_all_agents)
				
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
				# 	agent_reward = np.asarray(agent_reward, dtype=float)
					# all_the_agents[elem].memory.push(torch.from_numpy(state).unsqueeze(0),torch.from_numpy(np.asarray([actions_of_all_agents[elem]], dtype=int)).unsqueeze(0),torch.from_numpy(next_state).unsqueeze(0),torch.from_numpy(agent_reward).unsqueeze(0))

				overall_memory.push(torch.from_numpy(state).unsqueeze(0),torch.from_numpy(np.asarray([actions_of_all_agents], dtype=int)).unsqueeze(0),torch.from_numpy(next_state).unsqueeze(0),torch.from_numpy(system_reward).unsqueeze(0))

				# modify state for the next step
				state = next_state

				# restaurant.visualize_restaurant()
				# time.sleep(0.001)

			if episode % 10 == 0: 
				print("Training Loop: {} ; Episode: {} ; Reward : {} ; Successes: {}".format(training_loop, episode, episode_reward, episode_successes))
	
			# plot_rewards.append(episode_reward)        
			# plot_successes.append(episode_successes)
			# plot_N += 1    

			# n_iter = number_of_episodes * training_loop + episode
			# tensorboard_writer.add_scalar('data/rewards', episode_reward, n_iter)
			# tensorboard_writer.add_scalar('data/successes', episode_successes, n_iter)

		# optimize_model()
		# global_agent_vdn.optimize_model_global(batch_size,gamma)
		
		# for elem,each_agent in all_the_agents.items():
		# 	each_agent.target_net_agent.load_state_dict(each_agent.policy_net_agent.state_dict())


	# tensorboard_writer.export_scalars_to_json("./all_scalars.json")
	# tensorboard_writer.close()

	for elem, each_agent in all_the_agents.items():
		# model_path = '/home/abhijeet/Pytorch_models/dqn_' + str(elem)
		model_path = '/home/risheek/Pytorch_workspace/dqn_'+ str(elem)
		torch.save(each_agent.policy_net_agent.state_dict(), model_path)


	# plot_N = list(range(plot_N))

	# plot_reward_text = "Plotting rewards over all the episodes | For " + str(number_of_agents) + " chairbot in 5X5 env"
	# plot_successes_text = "Plotting successes over all the episodes | For " + str(number_of_agents) + " chairbot in 5X5 env"

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.set_title(plot_reward_text)
	# ax.set_xlabel('Number of episodes')
	# ax.set_ylabel('System reward in each episode')
	# plt.plot(plot_N, plot_rewards, 'm-', label='Rewards')
	# plt.legend()
	# plt.show()

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.set_title(plot_successes_text)
	# ax.set_xlabel('Number of episodes')
	# ax.set_ylabel('Successes in each episode')
	# plt.plot(plot_N, plot_successes, 'b-', label='Successes')
	# plt.legend()
	# plt.show()

