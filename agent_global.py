import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

from DQN import *


class Agent():

	def __init__(self, number_of_tables, number_of_agents, max_number_of_groups, grid_dim_x, grid_dim_y, batch_size, gamma, eps_start, eps_end, eps_decay, lr, weight_decay, capacity):
		self._number_of_tables = number_of_tables
		self._number_of_agents = number_of_agents
		self._max_number_of_groups = max_number_of_groups
		self._Ny = grid_dim_y
		self._Nx = grid_dim_x
		self._batch_size = batch_size
		self._gamma = gamma
		self._eps_start = eps_start
		self._eps_end = eps_end
		self._eps_decay = eps_decay
		self._capacity = capacity
		self._lr = lr
		self._weight_decay = weight_decay
		self._state_dim = 3 * self._number_of_tables + 5 * self._number_of_agents + 3 * self._max_number_of_groups
		self.initialize_policies()

	def initialize_policies(self):
		self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
		self.policy_net_agent = DQN(n_feature = self._state_dim)
		self.policy_net_agent.double()
		self.target_net_agent = DQN(n_feature = self._state_dim)
		self.target_net_agent.double()
		self.target_net_agent.load_state_dict(self.policy_net_agent.state_dict())
		# self.optimizer_agent = optim.RMSprop(self.policy_net_agent.parameters(), lr = self._lr, weight_decay = self._weight_decay)

	# def give_memory(self, memory):
	# 	self.memory = memory

	def get_action(self, policy_net_output, steps_done):
		sample = random.random()
		eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * (math.exp(- 1 * steps_done / self._eps_decay))
		if sample > eps_threshold:
			with torch.no_grad():

				if len(policy_net_output.shape) <= 1:
					_,action = torch.max(policy_net_output,0)
					# print('net_action--1',action)
					return action
				else:
					# print('net_action--2',policy_net_output.max(1)[1].view(1,1))
					return policy_net_output.max(1)[1].view(1,1)
		else:
			# print('random_action',torch.tensor([[random.randrange(5)]], dtype=torch.double))
			return torch.tensor([[random.randrange(5)]], dtype=torch.double)


class global_agent():

	def initialize_hyperparam(self):
		self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))		
		
		lr_1 = 1e-3
		weight_decay_1 = 1e-4

		network_parameters = list()
		params = list()

		for elem,each_agent in self.all_the_agents.items():
			network_parameters.append(each_agent.policy_net_agent.parameters())
			# for x in ....parameters:
			#   network_parameters.append
					
		for elem in network_parameters:
			params += list(elem)  
		
		self.optimizer_agent_global = optim.RMSprop(params, lr = lr_1, weight_decay = weight_decay_1)	

	def give_memory(self,overall_memory):
		self.memory = overall_memory

	def optimize_model_global(self,batch_size,gamma):
		self._batch_size = batch_size
		self._gamma = gamma
		if len(self.memory) < self._batch_size:
			return

		for optim_episode in range(100):
			transitions = self.memory.sample(self._batch_size)
			batch = self.Transition(*zip(*transitions))

			# Compute a mask of non-final states and concatenate the batch elements
			non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
												  batch.next_state)), dtype=torch.uint8)
			
			non_final_next_states = torch.cat([s for s in batch.next_state
														if s is not None])
			state_batch = torch.cat(batch.state)
			action_batch = torch.cat(batch.action)
			reward_batch = torch.cat(batch.reward)
		
			state_action_values = torch.zeros(self._batch_size)
			next_state_values = torch.zeros(self._batch_size) 
			expected_state_action_values = torch.zeros(self._batch_size) 

			for elem, each_agent in self.all_the_agents.items():
		
				state_action_values = state_action_values + each_agent.policy_net_agent(state_batch).gather(1, action_batch[:,:,elem]).squeeze(1)
				next_state_values[non_final_mask] = next_state_values[non_final_mask] + each_agent.target_net_agent(non_final_next_states).max(1)[0].detach()
				
				
			# Compute the expected Q values
			expected_state_action_values = (next_state_values * self._gamma) + reward_batch.double()
			
			# Compute Huber loss
			# loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
			loss = F.mse_loss(state_action_values, expected_state_action_values)
				
			
			network_parameters = list()
			params = list()

			for elem,each_agent in self.all_the_agents.items():
				network_parameters.append(each_agent.policy_net_agent.parameters())
			for elem in network_parameters:
				params += list(elem)  
			
			## optimize_model
			self.optimizer_agent_global.zero_grad()
			loss.backward()
			# for param in params:
			# 	param.grad.data.clamp_(-1, 1)
			self.optimizer_agent_global.step()

