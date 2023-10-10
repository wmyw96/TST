import numpy as np


class TimeSeriesSampler:
	def __init__(self, window, func, noise='uniform'):
		self.func = func
		self.window = window
		if noise == 'uniform':
			self.noise = np.random.uniform
		if noise == 'gaussian':
			self.noise = np.random.normal
		if noise == 'laplace':
			self.noise = np.random.laplace
		

	def sample(self, seq_len):
		u = self.noise(-1, 1, seq_len)
		x = np.zeros((seq_len,))
		x[:self.window] = u[:self.window]
		gt = np.zeros((seq_len,))
		for i in range(self.window, seq_len):
			x[i] = self.func(x[i-self.window:i], u[i])
			gt[i] = self.func(x[i-self.window:i], 0)
		return x, gt


def sample_func1(x, u):
	#return 0.5 * x[-1] + 0.5 * (u > 0)
	#return 1/np.sqrt(5) * (np.sin(x[0]) - np.tanh(x[1]) + np.sin(np.pi * x[2]) - np.sin(x[3]) + np.tanh(x[4])) + u

	#return np.sqrt(1+0.95*x[0]*x[0])*u
	#return 0.2*np.log(0.5+np.abs(x[0])) + u
	return 0.6*x[0] - 0.4*x[1]+0.2*x[2]-0.1*x[3]+0.05*x[4]+u




