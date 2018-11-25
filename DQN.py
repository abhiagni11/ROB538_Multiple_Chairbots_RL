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


class ReplayMemory:

	def __init__(self, capacity, Transition):
		self.capacity = capacity
		self.Transition = Transition
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = self.Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class DQN(nn.Module):

	def __init__(self, n_feature = 19, n_hidden = 25, n_hidden_1 = 30, n_hidden_2 = 30, n_output = 5):
		super(DQN, self).__init__()
	 
		self._n_feature = n_feature
		self.n_hidden = n_hidden
		self.n_hidden_1 = n_hidden_1
		self.n_hidden_2 = n_hidden_2
		self.n_output = n_output
		
		self.hidden=torch.nn.Linear(self._n_feature, self.n_hidden)
		init.xavier_normal(self.hidden.weight, gain=np.sqrt(2))
		self.dropout= torch.nn.Dropout(p=0.3)

		self.predict=torch.nn.Linear(self.n_hidden, self.n_output)
	
		init.xavier_normal(self.predict.weight, gain=np.sqrt(2))
		self.sigmoid=torch.nn.Sigmoid()

	def forward(self, x):
		x=F.relu(self.hidden(x)) 
		x=self.dropout(x)      
		x=self.predict(x)
		return x