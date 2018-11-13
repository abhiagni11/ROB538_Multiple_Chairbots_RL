
# coding: utf-8

# In[1]:


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

class Restaurant:

	def __init__(self, number_of_tables, number_of_agents, max_number_of_groups, grid_dim_x, grid_dim_y, number_of_steps):
		self._number_of_tables = number_of_tables
		self._number_of_agents = number_of_agents
		self._Ny = grid_dim_y  # y grid size
		self._Nx = grid_dim_x  # x grid size
		self._number_of_steps = number_of_steps
		self._step = 0
		self._all_tables = dict() # (x, y, max_capacity)
		self._all_agents = dict() # (previousX, previousY, X, Y, group_of_people)
		self._groups_of_people = dict() # (X, Y, number_of_people)
		self._number_of_groups_present = 0
		self._max_number_of_groups = max_number_of_groups
		self._agent_rewards = [0] * self._number_of_agents
		self._system_reward = 0
		self._action_dim = (5,)  # up, right, down, left
		self._action_dict = {"none": 0, "up": 1, "right": 2, "down": 3, "left": 4}
		self._action_coords = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
		self.fig, self.ax = plt.subplots()
		self._grid_viz = [[0 for x in range(self._Nx + 2)] for x in range(self._Ny + 2)]
		self._successes = 0
		# self.initialise_visualization()

	def initialise_tables(self):
		table_locations = [(1.,1.)]
		# K = 5
		# kx = (K + 1)
		# remainder = 1
		# if self._number_of_tables % K:
		# 	remainder += 1 
		# ky = ((int(self._number_of_tables / K)) + remainder)
		# for y in np.arange(0 + ky, self._Ny - ky, ky):
		# 	for x in np.arange(0 + kx, self._Nx - kx, kx):
		# 		table_locations.append((x, y))

		for elem in range(self._number_of_tables):
			self._all_tables[elem] = (table_locations[elem][0], table_locations[elem][1], randint(2,2))

	def initialise_agents(self):
		for elem in range(self._number_of_agents):
			self._all_agents[elem] = (0, 0, self._all_tables[elem][0], self._all_tables[elem][1], -1)
			# self._all_agents[elem] = (0, 0, randint(0, self._Nx), randint(0, self._Ny), -1)
			# self._all_agents[elem] = (0, 0, float(int(self._Nx/2)), 0, -1)
			self._agent_rewards[elem] = 0

	def initialise_groups(self):
		for elem in range(self._max_number_of_groups):
			self._groups_of_people[elem] = (-1, -1, randint(2,2))

	def initialise_visualization(self):
		plt.tight_layout()
		plt.imshow(self._grid_viz, cmap='Greys')
		plt.ion()
		plt.show()

	def reset(self):
		self._step = 0
		self._successes = 0
		self._all_tables.clear()
		self._all_agents.clear()
		self._groups_of_people.clear()
		self._number_of_groups_present = 0
		self.initialise_tables()
		self.initialise_agents()
		self.initialise_groups()

	def step(self, all_agent_actions):
		self._agent_rewards = [0] * self._number_of_agents

		for elem, each_agent in (self._all_agents.items()):
			# Get agent's next state
			allowed_actions = self.check_allowed_action((each_agent[2], each_agent[3]))
			given_action = all_agent_actions[elem]
			if given_action in allowed_actions:
				agent_state_next = (each_agent[2] + self._action_coords[given_action][0], each_agent[3] + self._action_coords[given_action][1])
			else:
				agent_state_next = (each_agent[2], each_agent[3])
			# Update the state and reward of each agent
			self.update_state_and_reward(elem, each_agent, agent_state_next)

		# Introduce people in the system
		self.add_new_group_of_people()

		# Calculate the system reward
		self.calculate_system_reward()

		self._step += 1

	def check_allowed_action(self, state):
		# Generate list of actions allowed depending on agent grid location
		actions_allowed = [self._action_dict["none"]]
		y, x = state[0], state[1]
		if (y > 0):  # no passing top-boundary
			actions_allowed.append(self._action_dict["up"])
		if (x < self._Nx - 1):  # no passing right-boundary
			actions_allowed.append(self._action_dict["right"])
		if (y < self._Ny - 1):  # no passing bottom-boundary
			actions_allowed.append(self._action_dict["down"])
		if (x > 0):  # no passing left-boundary
			actions_allowed.append(self._action_dict["left"])
		return actions_allowed

	def update_state_and_reward(self, elem, each_agent, agent_state_next):
		old_group_number = each_agent[-1]
		new_group_number = old_group_number
		# check if agent has any group with it
		if each_agent[-1] is not -1:
			group_number = each_agent[-1]
			# it had a group, so update that group's location according to the agent's updated state and check if the group reached any table
			reached_table_number = self.update_group_location(group_number, agent_state_next)
			# Agent's reward is -1 for carrying the group with it
			self._agent_rewards[elem] -= 1
			if reached_table_number is not -1:
				# print("reached_table: {}".format(reached_table_number))
				self._agent_rewards[elem] += 40
				self._all_agents[elem] = list(self._all_agents[elem])
				self._all_agents[elem][-1] = -1
				new_group_number = -1
				self._successes += 1
		else:
			# check if it just stumbled into a group or not
			group_number = self.check_agent_found_group(agent_state_next)
			new_group_number = group_number
		if old_group_number is -1 and group_number is not -1:
			# Agent just found a group, so it'll get +40 reward
			self._agent_rewards[elem] += 40
			# print("Found the group")
		# Update the agent's reward in the dictionary
		self._all_agents[elem] = (each_agent[2], each_agent[3], agent_state_next[0], agent_state_next[1], new_group_number)
		

	def update_group_location(self, group_number, location):
		# Update the particular group's location
		# print("group: {}".format(self._groups_of_people[group_number][0]))
		self._groups_of_people[group_number] = list(self._groups_of_people[group_number])
		self._groups_of_people[group_number][0] = location[0]
		self._groups_of_people[group_number][1] = location[1]
		# print("group: {}".format(self._groups_of_people[group_number]))
		return self.check_group_reached_which_table(group_number)

	def check_agent_found_group(self, state):
		# iterate through all the groups' locations and find a match with the agent's updated state
		# returns the group id if agent finds a group, otherwise return -1
		group = -1
		for elem, each_group in self._groups_of_people.items():
			if state[0] == (each_group[0]) and state[1] == each_group[1]:
				group = elem
				break
		return group

	def check_group_reached_which_table(self, group_number):
		reached_table_number = -1
		group = self._groups_of_people[group_number]
		for elem, each_table in self._all_tables.items():
			if group[0] == each_table[0] and group[1] == each_table[1]:
				reached_table_number = elem
				(self._groups_of_people[group_number][0], self._groups_of_people[group_number][1]) = (0, 0)
				break
		return reached_table_number

	def add_new_group_of_people(self):
		iter_step = self._number_of_steps/100
		while self._number_of_groups_present < self._max_number_of_groups:
			if ((self._step < (iter_step * 10)) and (self._step % iter_step == 0)):
				self._groups_of_people[self._number_of_groups_present] = (self._Nx/2, 0, randint(2,2))
				self._number_of_groups_present += 1
				# print("Total groups: {}".format(self._number_of_groups_present))
		# if (self._step == 0):
		# 	self._groups_of_people[self._number_of_groups_present] = (0, 0, randint(2,2))
		# 	self._number_of_groups_present += 1

	def calculate_system_reward(self):
		self._system_reward = 0
		for reward in self._agent_rewards:
			self._system_reward += reward

	def get_system_reward(self):
		return self._system_reward

	def get_sucessess(self):
		return self._successes

	def get_observation(self):
		return self._all_tables, self._all_agents, self._groups_of_people

	def get_local_reward(self, agent):
		None

	def get_local_observation(self, agent):
		None

	def visualize_restaurant(self):
		"""Visualize the grid with the agents and target.
		""" 
		plt.cla()
		# self.ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
		# self.ax.set_xticks(np.arange(0, self._Nx+1, 5));
		# self.ax.set_yticks(np.arange(0, self._Ny+1, 5));
		self.ax.set_xlim([0, self._Nx+2])
		self.ax.set_ylim([0, self._Ny+2])

		# Plot tables
		for elem, each_table in (self._all_tables.items()):
			table_state = (each_table[0], each_table[1])
			self.ax.plot(table_state[0]+1, table_state[1]+1, "cs", markersize=10)
			table_text = 'T' + str(elem)
			self.ax.text(table_state[0] -.0+1, table_state[1] -.0+1, table_text)

		# Plot agents
		for elem, each_agent in (self._all_agents.items()):
			agent_state = (each_agent[2], each_agent[3])
			chairbot = plt.Circle((agent_state[0]+1, agent_state[1]+1), .1, color='y')
			self.ax.add_artist(chairbot)
			chairbot_text = 'C' + str(elem)
			self.ax.text(agent_state[0] -.0+1, agent_state[1] -.0+1, chairbot_text)
		
		# Plot entrance
		circle_entrance = plt.Circle((int(self._Nx/2)+1, 1), .2, color='g')
		self.ax.add_artist(circle_entrance)
		self.ax.text(int(self._Nx/2) - 2, -2, 'ENTRANCE')

		# Plot reward
		reward_text = 'R-> ' + str(self._system_reward) + "ep" + str(self._step)
		self.ax.text(-10, self._Ny, reward_text)

		self.ax.plot()
		plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
		# plt.axis('off')
		plt.draw()
		plt.pause(.01)

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

class Agent():

	def __init__(self, number_of_tables, number_of_agents, grid_dim_x, grid_dim_y):
		self._number_of_tables = number_of_tables
		self._number_of_agents = number_of_agents
		self._Ny = grid_dim_y  # y grid size
		self._Nx = grid_dim_x  # x grid size
		self._action = 0  # x grid size

### DQN
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
class ReplayMemory:

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):

#     def __init__(self):
	def __init__(self,n_feature = 11,n_hidden = 25,n_hidden_1 = 30,n_hidden_2 = 30,n_output = 5):
		super(DQN, self).__init__()
	 
		self._n_feature = n_feature
		self.n_hidden = n_hidden
		self.n_hidden_1 = n_hidden_1
		self.n_hidden_2 = n_hidden_2
		self.n_output = n_output
		
		
		self.hidden=torch.nn.Linear(self._n_feature, self.n_hidden)   # hidden layer
		init.xavier_normal(self.hidden.weight, gain=np.sqrt(2))
		self.dropout= torch.nn.Dropout(p=0.3)
	
		# self.hidden1= torch.nn.Linear(self.n_hidden, self.n_hidden_1)   # hidden layer
		# init.xavier_normal(self.hidden1.weight, gain=np.sqrt(2))
		# self.dropout1=torch.nn.Dropout(p=0.3)      
	
		# self.hidden2= torch.nn.Linear(self.n_hidden_1, self.n_hidden_2)   # hidden layer
		# init.xavier_normal(self.hidden2.weight, gain=np.sqrt(2))
		# self.dropout2=torch.nn.Dropout(p=0.3)      
		
		# self.predict=torch.nn.Linear(self.n_hidden_2, self.n_output)   # output layer. if hidden 2 is used
		self.predict=torch.nn.Linear(self.n_hidden, self.n_output)   # output layer. if hidden 2 is used
	
		# self.predict=torch.nn.Linear(self._n_hidden_1, self.n_output)   # output layer. if hidden1 is last hidden layer
		init.xavier_normal(self.predict.weight, gain=np.sqrt(2))
		self.sigmoid=torch.nn.Sigmoid()


	def forward(self, x):
		x=F.relu(self.hidden(x)) 
		x=self.dropout(x)
		# x=F.relu(self.hidden1(x)) 
		# x=self.dropout1(x)        
		# x=F.relu(self.hidden2(x)) 
		# x=self.dropout2(x)        
		x=self.predict(x)
		return x

def get_action(policy_net_output,done):
	sample = random.random()
	eps_threshold = eps_end + (eps_start - eps_end) * (math.exp(- 1 * steps_done / eps_decay))
	if sample < eps_threshold:
		# print('action_exploitation')
		with torch.no_grad():
			# return self._policy_net(self._state).max(1)[1].view(1, 1)
			if len(policy_net_output.shape) <= 1:
				_,action = torch.max(policy_net_output,0)
				return action
			else:
				return policy_net_output.max(1)[1].view(1,1)
	else:
		# print('action_exploration')
		# return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)
		return torch.tensor([[random.randrange(5)]], dtype=torch.double)

def policy_network_eval(policy_net,state):
	return policy_net(state)

def optimize_model(policy_net,target_net,memory,optimizer,gamma):
	if len(memory) < batch_size:
		return

	for optim_episode in range(100):
		transitions = memory.sample(batch_size)
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
											  batch.next_state)), dtype=torch.uint8)
		
		non_final_next_states = torch.cat([s for s in batch.next_state
													if s is not None])
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)
		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken
		
		# print("state_batch",type(state_batch),)
		
		state_action_values = policy_net(state_batch).gather(1,action_batch)
		# state_action_values = state_action_values_temp.gather(1,action_batch)
		# .gather(1, action_batch)

		# Compute V(s_{t+1}) for all next states.
		# next_state_values = torch.zeros(BATCH_SIZE, device=device)
		next_state_values = torch.zeros(batch_size) 
		next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
		# next_state_values.double()
		
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * gamma) #+ reward_batch.double()
		
		# Compute Huber loss
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		### Optimize the model
		optimizer.zero_grad()
		loss.backward()
		for param in policy_net.parameters():
			param.grad.data.clamp_(-1, 1)
		optimizer.step()

batch_size = 100
gamma = 0.9
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
capacity = 100000

policy_net_agent_1 = DQN()
policy_net_agent_1.double()

target_net_agent_1 = DQN()
target_net_agent_1.double()

target_net_agent_1.load_state_dict(policy_net_agent_1.state_dict())

optimizer_agent_1 = optim.RMSprop(policy_net_agent_1.parameters(),lr=1e-4)
memory= ReplayMemory(capacity)

steps_done = 0

if __name__ == "__main__":

	number_of_tables = 1
	number_of_agents = 1
	grid_dim_x = 5
	grid_dim_y = 5
	max_number_of_groups = 1
	number_of_training_loops = 10
	number_of_episodes = 20
	number_of_steps = 500

	restaurant = Restaurant(number_of_tables, number_of_agents, max_number_of_groups, grid_dim_x, grid_dim_y, number_of_steps)
	all_the_agents = dict()

	for elem in range(number_of_agents):
		agent = Agent(number_of_tables, number_of_agents, grid_dim_x, grid_dim_y)
		all_the_agents[elem] = agent._action

	for training_loop in range(number_of_training_loops):
		# print('training_loop',training_loop)
		
		for episode in range(number_of_episodes):

			# print("episode",episode)
			# reset the environment
			restaurant.reset()

			# Get initial observation from the enivorment
			all_table, all_agents, all_people = restaurant.get_observation()
			state = concatenate_state(all_table, all_agents, all_people)
#             state = state.astype(float)
			episode_reward = 0
			episode_successes = 0
			for each_step in range(number_of_steps):
#                 print('each_step',each_step)
				# get actions from all the agents
				actions_of_all_agents = [0] * number_of_agents
				actions_of_all_agents = np.asarray(actions_of_all_agents)
				for elem, each_agent in all_the_agents.items():
					# actions_of_all_agents[elem] = each_agent.get_action(state)
#                   actions_of_all_agents[elem] = randint(0,4)
					policy_net_output = policy_net_agent_1(torch.from_numpy(state))
					print("policy_net_output : {}".format(policy_net_output))
#                     print('policy_net_output',policy_net_output)#,policy_net_output.max)
					action_temp = get_action(policy_net_output,steps_done)
#                     print('action',action_temp)#,action_temp.numpy())
					actions_of_all_agents[0] = action_temp.numpy()
				# Step: get next state and reward for the action taken
#                 print('position_agent',state[(3*number_of_tables+2):(3*number_of_tables+4)])
			
				restaurant.step(actions_of_all_agents)
				steps_done += 1
				
				# Get updated observation from the environment
				next_all_table, next_all_agents, next_all_people = restaurant.get_observation()
				next_state = concatenate_state(next_all_table, next_all_agents, next_all_people)
				
#                 print('position_agent',next_state[(3*number_of_tables+2):(3*number_of_tables+4)])
				
				# Get reward for the action taken
				system_reward = restaurant.get_system_reward()
				episode_reward += system_reward
#                 print('system_reward',system_reward)
				
				system_reward = np.asarray(system_reward,dtype=float)
				episode_successes = restaurant.get_sucessess()
#                 print('1',torch.from_numpy(state).unsqueeze(0))
#                 print('2',torch.from_numpy(actions_of_all_agents).unsqueeze(0))
#                 print('3',torch.from_numpy(next_state).unsqueeze(0))
#                 print('4',torch.from_numpy(system_reward).unsqueeze(0))
				
				# Store the transition in memory
#                 memory.push(state, actions_of_all_agents[0], next_state, system_reward)
#                 print(state,system_reward,next_state,actions_of_all_agents)
				memory.push(torch.from_numpy(state).unsqueeze(0),torch.from_numpy(actions_of_all_agents).unsqueeze(0),torch.from_numpy(next_state).unsqueeze(0),torch.from_numpy(system_reward).unsqueeze(0))

				# modify state for the next step
				state = next_state
				# restaurant.visualize_restaurant()
				# time.sleep(0.001)   
			if episode % 10 == 0: 
				print("Training Loop: {} ; Episode: {} ; Reward : {} ; Successes: {}".format(training_loop, episode, episode_reward, episode_successes))        
		# Perform one step of the optimization (on the target network)
		None
		# optimize_model()
		optimize_model(policy_net_agent_1,target_net_agent_1,memory,optimizer_agent_1,gamma)

model_path = '/home/abhijeet/Pytorch_models/dqn_1'
torch.save(policy_net_agent_1.state_dict(), model_path)

