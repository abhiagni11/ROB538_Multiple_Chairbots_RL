
import time
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count



def smooth(y, box_pts):
	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth

if __name__ == "__main__":

	### Load all the numpy arrays

	### For IQL - diff
	IQL_diff_stat_system_rewards_mean = np.mean(np.array(all_stat_system_rewards), axis=0)
	IQL_diff_system_rewards_std_dev = np.std(np.array(all_stat_system_rewards), axis=0)

	IQL_diff_system_successes_mean = np.mean(np.array(all_stat_system_successes), axis=0)
	IQL_diff_system_successes_std_dev = np.std(np.array(all_stat_system_successes), axis=0)

	stat_n_iter = load(stat_n_iter)

	### For IQL - global
	IQL_global_stat_system_rewards_mean = np.mean(np.array(all_stat_system_rewards), axis=0)
	IQL_global_system_rewards_std_dev = np.std(np.array(all_stat_system_rewards), axis=0)

	IQL_global_system_successes_mean = np.mean(np.array(all_stat_system_successes), axis=0)
	IQL_global_system_successes_std_dev = np.std(np.array(all_stat_system_successes), axis=0)

	### For VDN
	VDN_stat_system_rewards_mean = np.mean(np.array(all_stat_system_rewards), axis=0)
	VDN_system_rewards_std_dev = np.std(np.array(all_stat_system_rewards), axis=0)

	VDN_system_successes_mean = np.mean(np.array(all_stat_system_successes), axis=0)
	VDN_system_successes_std_dev = np.std(np.array(all_stat_system_successes), axis=0)

	##################################################################################

	plot_reward_text = "Plotting rewards over all the episodes | For " + '_n_' + " chairbot in 5X5 env"

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(plot_reward_text)
	ax.set_xlabel('Number of episodes')
	ax.set_ylabel('System reward')

	smoothening_factor = 1

	# Plot IQL for difference rewards
	plt.plot(stat_n_iter, smooth(IQL_diff_stat_system_rewards_mean,smoothening_factor), 'k', label='IQL_D', color='#CC4F1B')
	plt.fill_between(stat_n_iter, smooth(IQL_diff_stat_system_rewards_mean-IQL_diff_system_rewards_std_dev,smoothening_factor), smooth(IQL_diff_stat_system_rewards_mean+IQL_diff_system_rewards_std_dev,smoothening_factor),
		alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

	# Plot IQL for global rewards
	plt.plot(stat_n_iter, smooth(IQL_global_stat_system_rewards_mean,smoothening_factor), 'k', label='IQL_G', color='#1B2ACC')
	plt.fill_between(stat_n_iter, smooth(IQL_global_stat_system_rewards_mean-IQL_global_system_rewards_std_dev,smoothening_factor), smooth(IQL_global_stat_system_rewards_mean+IQL_global_system_rewards_std_dev,smoothening_factor),
		alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

	# Plot VDN
	plt.plot(stat_n_iter, smooth(VDN_stat_system_rewards_mean,smoothening_factor), 'k', label='VDN', color='#3F7F4C')
	plt.fill_between(stat_n_iter, smooth(VDN_stat_system_rewards_mean-VDN_system_rewards_std_dev,smoothening_factor), smooth(VDN_stat_system_rewards_mean+VDN_system_rewards_std_dev,smoothening_factor),
		alpha=0.5, edgecolor='#3F7F4C', facecolor='#089FFF')

	plt.legend()
	plt.show()

	##################################################################################
	
	plot_successes_text = "Plotting successes over all the episodes | For " + '_n_' + " chairbot in 5X5 env"

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(plot_successes_text)
	ax.set_xlabel('Number of episodes')
	ax.set_ylabel('System successes')

	smoothening_factor = 1

	# Plot IQL for difference rewards
	plt.plot(stat_n_iter, smooth(IQL_diff_system_successes_mean,smoothening_factor), 'm-', label='IQL_D', color='#CC4F1B')
	plt.fill_between(stat_n_iter, smooth(IQL_diff_system_successes_mean-IQL_diff_system_successes_std_dev,smoothening_factor), smooth(IQL_diff_system_successes_mean+IQL_diff_system_successes_std_dev,smoothening_factor),
		alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

	# Plot IQL for global rewards
	plt.plot(stat_n_iter, smooth(IQL_global_system_successes_mean,smoothening_factor), 'm-', label='IQL_G', color='#1B2ACC')
	plt.fill_between(stat_n_iter, smooth(IQL_global_system_successes_mean-IQL_global_system_successes_std_dev,smoothening_factor), smooth(IQL_global_system_successes_mean+IQL_global_system_successes_std_dev,smoothening_factor),
		alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

	# Plot VDN
	plt.plot(stat_n_iter, smooth(VDN_system_successes_mean,smoothening_factor), 'm-', label='VDN', color='#3F7F4C')
	plt.fill_between(stat_n_iter, smooth(VDN_system_successes_mean-VDN_system_successes_std_dev,smoothening_factor), smooth(VDN_system_successes_mean+VDN_system_successes_std_dev,smoothening_factor),
		alpha=0.5, edgecolor='#3F7F4C', facecolor='#089FFF')

	plt.legend()
	plt.show()