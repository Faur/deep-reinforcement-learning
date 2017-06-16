from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import toimage, fromimage


class Annealer():
	"""Simple class that helps with annealing"""
	def __init__(self, initial_value, end_value, period):
		self.initial_value = initial_value
		self.epsilon = initial_value
		self.end_value = end_value
		self.period = float(period)

	def linear(self, step):
		epsilon = self.initial_value - self.initial_value*step/self.period
		if epsilon < self.end_value:
			return self.end_value
		return epsilon


class Experience_buffer():
	""" Consists of a list of 'experience', each of which is a dict.
		The keys should be the same for experience, but are not set in advance
	"""
	def __init__(self, buffer_capacity=int(1e6)):
		self.buffer = []
		self.buffer_capacity = buffer_capacity
	
	def __str__(self):
		""" Simply print the content of the buffer"""
		tostr = ''
		for item in self.buffer:
			tostr += item.__str__()
			tostr += '\n'
		return tostr
	
	def buffer_size(self):
		""" Return the current number of experiences in the buffer."""
		return len(self.buffer)
	
	def add(self, experience):
		""" Add ONE experience to the experience buffer
			Every entry in an expericen shold be i dimensional at most!
		"""
		self.buffer.append(experience)
		while len(self.buffer) > self.buffer_capacity:
			self.buffer.pop(0)
	
	def clear(self):
		self.buffer = []

	def sample(self, size):
		""" Draw 'size' random samples from the experience buffer, 
			with replacement"""
		batch = {'obs':[], 'action':[], 'reward':[],
					  'next_obs':[], 'done':[]}
		for _ in range(size):
			i = np.random.randint(low=0, high=self.buffer_size())
			example = self.buffer[i]
			for item in example:
				batch[item].append(example[item])

		for item in batch:
			# Make the data into an array
			batch[item] = np.vstack(batch[item]) 
			# Remove superfluous dimensions
			batch[item] = np.squeeze(batch[item])
		return batch
				
	def test(self):
		""" Simple test, validating that the replay buffer works
		"""
		print('\n##############################################################################')
		print('TEST: Experience_buffer\n')
		limit = 3
		print('Test that buffer is limited to ' + str(limit))
		buffer = Experience_buffer(limit)
		for i in range(5):
			obs = i
			experience = {'obs':obs, 'action':i, 'reward':i, 'next_obs':obs, 'done':i}
			print(i, buffer.buffer_size())
			buffer.add(experience)
		
		print('\nContent of buffer')
		print(buffer)

		print('Test 0d obs')
		batch = buffer.sample(5)
		print(type(batch['obs']))
		print('Should not have superflouous dimensions')
		print(batch['obs'].shape)
		buffer.clear()
		print('\nTest buffer.clear: len = ' + str(buffer.buffer_size()))

		print('\nTest 1d obs')
		for i in range(5):
			obs = np.array([i,i])
			experience = {'obs':[obs], 'action':i, 'reward':i, 'next_obs':[obs], 'done':i}
			buffer.add(experience)

		batch = buffer.sample(5)
		print(type(batch['obs'])) 
		print(batch['obs'].shape)
		buffer.clear()

		print('\nTest 2d obs')
		for i in range(5):
			obs = np.array([[i,i], [i,i]])
			experience = {'obs':[obs], 'action':i, 'reward':i, 'next_obs':[obs], 'done':i}
			buffer.add(experience)

		batch = buffer.sample(5)
		print(type(batch['obs']))
		print(batch['obs'].shape)
		print()


class ObsBuffer():
	""" Buffer for when multiple information from multiple timeframes are
		used as a single input to the network.

		Arguments:
		* obs_shape: The shape of the individual observations, NOT the 
					 input shape to the final network!

		Methods:
		* add(obs):	Adds obs to the buffer, and removes the oldest
		* get(): 	Returns buffer as a depth stacked numpy array
		* reset(): 	Fills the buffer with zeros
	"""
	def __init__(self, obs_shape, buffer_size):
		self.obs_shape = obs_shape
		self.buffer_size = buffer_size

		# Initialize an empty buffer
		self.buffer = []
		self.reset()
	
	def add(self, obs):
		self.buffer.append(obs)
		# Remove excess obs
		while len(self.buffer) > self.buffer_size:
			self.buffer.pop(0)

	def get(self):
		# TODO: Generalize: This only handles images ATM
		return np.dstack(self.buffer)
	
	def reset(self):
		self.buffer = [np.zeros(self.obs_shape) \
			for i in range(self.buffer_size)]

	def test(self):
		print('\n##############################################################################')
		print('TEST: ObsBuffer\n')

		obs = np.ones([10, 10, 1])
		obsBuf = ObsBuffer([10, 10], 4)
		print('obsBuf.buffer   ', type(obsBuf.buffer), len(obsBuf.buffer))
		print('obsBuf.buffer[0]', type(obsBuf.buffer[0]), obsBuf.buffer[0].shape)
		print('obsBuf.get	  ', type(obsBuf.get()), obsBuf.get().shape)
		print()
		print('sum			', np.sum(obsBuf.get()))
		obsBuf.add(obs)
		obsBuf.add(obs)
		obsBuf.add(obs)
		obsBuf.add(obs)
		obsBuf.add(obs)
		print('sum after add  ', np.sum(obsBuf.get()))
		obsBuf.reset()
		print('sum after reset', np.sum(obsBuf.get()))
		print()


class Preprocessor_2d():	
	""" Preprocessor intended for images.

		Assumptions:
		* Images are either RGB or grayscale. Thus valid shapes are 
			[w, h, 1] and [w, h, 3]

		Args:
		* out_shape: List with [width, height] that the obs should be stretched to fit

		Methods:
	"""
	def __init__(self, out_shape, gray=False):
		self.out_shape = out_shape
		self.gray = gray
		self.n_channels = 1 if self.gray else 3
		
	def _make_gray(self, img):
		if len(img.shape)==3:
			img = np.mean(img, -1)
			 # pre = pre.convert('L') # Alternative way, more fancy, but probably worse
		if len(img.shape)==2:
			img = np.expand_dims(img, axis=-1)
		return img
		
	def process(self, obs):
		pre = toimage(obs)
		pre = pre.resize(self.out_shape)
		pre = fromimage(pre)
		if self.gray:
			pre = self._make_gray(pre)
		# For consistency an image ALWAYS has dimensions [w, h, c]!
		assert len(pre.shape)==3, "ERROR: Preprocessor_2d: pre has dim: " + str(pre.shape)
		return pre

	def test(self):
		print('\n##############################################################################')
		print('TEST: Preprocessor_2d\n')
		# get an observation
		import gym
		env = gym.make('Breakout-v0')
		obs = env.reset()
		preprocessor = Preprocessor_2d(out_shape=[84,84], gray=True)

		print('Test preprocessing step')
		pre = preprocessor.process(obs)
		print('obs', type(obs), obs.shape, np.product(obs.shape))
		print('pre', type(pre), end=' ')
		print(pre.shape, np.product(pre.shape))
		print((1.*np.product(pre.shape))/np.product(obs.shape), '% of original')
		print()

		# Visualize preprocessing
		fig, ax = plt.subplots(1,2)
		fig.suptitle('Preprocessor_2d test')
		ax[0].imshow(obs)
		ax[0].set_title('Original')
		ax[1].imshow(np.squeeze(pre), cmap='gray')
		ax[1].set_title('After Preprocessing')
		plt.draw()


class EnvironmentInterface():
	def __init__(self, preprocessor=None, action_repeats=1, merge_frames=False, clip_reward=None):
		""" 
			Arguments:
			* clip_rewards: None --> Don't clip. Otherwise list of [low, high]
				List can contain None, to remove clippage.
		"""
		self.preprocessor = preprocessor
		self.action_repeats = action_repeats
		self.merge_frames = merge_frames
		
		if clip_reward is None:
			self.clip = lambda x: x
		else:
			self.clip = lambda x: np.clip(x, clip_reward[0], clip_reward[1])

		if self.merge_frames:
			assert self.action_repeats > 1, 'ERROR: EnvironmentInterface: '\
				+ 'Cannot merge frames with action_repeats !> 1.'\
				+ 'action_repeats = ' + str(self.action_repeats)

	def take_action(self, action, env):
		""" Take an action self.action_repeats times, and return an
			(optionally) preprocessed obs
		"""
		obs = None
		prev_obs = obs
		total_reward = 0
		done = False
		info = ''

		# Repeat action self.action_repeats times
		for i in range(self.action_repeats):
			prev_obs = obs
			obs, reward, done, info = env.step(action)
			reward = self.clip(reward) # Only clips if self.clip is defined! See __init__
			total_reward += reward
			if done: break

		if self.merge_frames:
			obs = np.maximum(obs, prev_obs)

		if self.preprocessor is not None:
			obs = self.preprocessor.process(obs)

		return obs, total_reward, done, info
	
	def reset_env(self, env):
		"""Simple wrapper that restarts the environment"""
		obs = env.reset()
		episode_time_step = 0
		episode_reward = 0
		
		if self.preprocessor is not None:
			obs = self.preprocessor.process(obs)
		return obs, episode_time_step, episode_reward

	def test(self):
		print('\n##############################################################################')
		print('TEST: EnvironmentInterface\n')
		import gym
		env = gym.make('Breakout-v0')
		preprocessor = Preprocessor_2d([84, 84], gray=True)

		envInter = EnvironmentInterface(preprocessor=preprocessor, action_repeats=4, merge_frames=True, clip_reward=[-1,1])

		obs, eps_r, eps_t = envInter.reset_env(env)
		print('obs from reset_env: ', type(obs), obs.shape)
		fig, ax = plt.subplots(2,2)
		fig.suptitle('EnvironmentInterface test')
		for i in range(2):
			for j in range(2):
				obs, reward, done, info = envInter.take_action(1, env)
				ax[i][j].imshow(np.squeeze(obs), cmap='gray')
				ax[i][j].set_title('t = ' + str(5 - i*2 - j))
				ax[i][j].axis('off')
		fig.tight_layout()

		print('obs from take_action: ', type(obs), obs.shape)

		plt.draw()


if __name__=='__main__':
	Experience_buffer().test()
	ObsBuffer([1], 1).test()
	Preprocessor_2d(None).test()
	EnvironmentInterface().test()
	plt.show()


