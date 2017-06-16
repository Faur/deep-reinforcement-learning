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
			Tests the following features:
				__str__
				buffer_size
				add
				sample
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
		print('obsBuf.get      ', type(obsBuf.get()), obsBuf.get().shape)
		print()
		print('sum            ', np.sum(obsBuf.get()))
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
		fig.suptitle('Preprocessor test')
		ax[0].imshow(obs)
		ax[0].set_title('Original')
		ax[1].imshow(np.squeeze(pre), cmap='gray')
		ax[1].set_title('After Preprocessing')
		plt.show()


if __name__=='__main__':
	Experience_buffer().test()
	ObsBuffer([1], 1).test()
	Preprocessor_2d(None).test()



