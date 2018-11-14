import time
import matplotlib.pyplot as plt
from random import randint
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count


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
		self._action_dim = (5,)  # none, up, right, down, left
		self._action_dict = {"none": 0, "up": 1, "right": 2, "down": 3, "left": 4}
		self._action_coords = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
		self.fig, self.ax = plt.subplots()
		self._grid_viz = [[0 for x in range(self._Nx + 2)] for x in range(self._Ny + 2)]
		self._successes = 0

	def initialise_tables(self):
		if self._number_of_tables == 1:
			table_locations = [(1.,1.)]
		elif self._number_of_tables == 2:
			table_locations = [(1.,1.),(2.,2.)]
		elif self._number_of_tables == 3:
			table_locations = [(1.,1.),(2.,2.),(3.,3.)]
		else:
			table_locations = []
			K = 5
			kx = (K + 1)
			remainder = 1
			if self._number_of_tables % K:
				remainder += 1 
			ky = ((int(self._number_of_tables / K)) + remainder)
			for y in np.arange(0 + ky, self._Ny - ky, ky):
				for x in np.arange(0 + kx, self._Nx - kx, kx):
					table_locations.append((float(int(x)), (float(int(y)))))

		for elem in range(self._number_of_tables):
			self._all_tables[elem] = (table_locations[elem][0], table_locations[elem][1], randint(2,2))

	def initialise_agents(self):
		for elem in range(self._number_of_agents):
			self._all_agents[elem] = (0, 0, self._all_tables[elem][0], self._all_tables[elem][1], -1)
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