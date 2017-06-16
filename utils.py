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
		print('Testing Experience_buffer')

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

class DataHandler():	
	def __init__(self):
		self.preprocessing_2d_defined = False
		self.buffer = []

	def _make_gray(self, img):
		if len(img.shape)==3:
			img = np.mean(img, -1)
			 # pre = pre.convert('L') # Alternative way, more fancy, but probably worse
		if len(img.shape)==2:
			img = np.expand_dims(img, axis=-1)
		# For consistency an image ALWAYS has dimensions [w, h, c]!
		assert len(img.shape)==3, "WARNING: _make_gray: img has dim: " + str(img.shape)
		return img
		
	def define_preprocess_2d(self, target_size, num_frames=1, gray=False):
		"""
			Args:
				target_size: List with [width, height] that the game should be stretched to fit
		"""
		self.num_frames = num_frames
		self.target_size = target_size
		self.gray = gray
		self.n_channels = 1 if self.gray else 3
		
		 # Fill the buffer with zeros
		self.reset_buffer_2d()

		self.preproces_2d_defined = True		
	
	def preprocess_2d(self, obs):
		assert self.preproces_2d_defined, "ERROR: You must run define_preprocessing_2d before running preprocessing_2d!"
		pre = toimage(obs)
		pre = pre.resize(self.target_size)
		pre = fromimage(pre)
		if self.gray:
			pre = self._make_gray(pre)
		return pre
	
	def add_2d(self, obs):
		""" Buffer holding self.num_frames"""
		self.buffer.pop(0)
		pre = self.preprocess_2d(obs)
		self.buffer.append(pre)

	def get_buffer_2d(self):
		"""Returns buffer as a vertically stacked numpy array"""
		return np.dstack(self.buffer)
	
	def reset_buffer_2d(self):
		self.buffer = [np.zeros(self.target_size+[self.n_channels]) \
			for i in range(self.num_frames)]


	def test(self):
		# get an observation
		import gym
		env = gym.make('Breakout-v0')
		obs = env.reset()
		
		# Create preprocessor
		preprocessor = DataHandler()
		preprocessor.define_preprocess_2d(target_size=[84,84], num_frames=4, gray=True)

		print('Test preprocessing step')
		pre = preprocessor.preprocess_2d(obs)
		print('obs', type(obs), obs.shape, np.product(obs.shape))
		print('pre', type(pre), end=' ')
		print(pre.shape, np.product(pre.shape))
		print((1.*np.product(pre.shape))/np.product(obs.shape), '% of original')
		print()

		print('Test buffer')
		preprocessor.add_2d(obs)
		preprocessor.add_2d(obs)
		pre_buffer = preprocessor.get_buffer_2d()
		print('env_handler.buffer', type(preprocessor.buffer), len(preprocessor.buffer))
		print('env_handler.buffer[0]', type(preprocessor.buffer[0]), preprocessor.buffer[0].shape)
		print('pre_buffer', type(pre_buffer), pre_buffer.shape)
		print()

		# Visualize preprocessing
		fig, ax = plt.subplots(1,2)
		fig.suptitle('Preprocessor test')
		ax[0].imshow(obs)
		ax[0].set_title('Original')
		ax[1].imshow(np.squeeze(pre), cmap='gray')
		ax[1].set_title('After Preprocessing')
		# plt.show()
		plt.draw()

if __name__=='__main__':
	print('##############################################################################')
	print('TEST: DataHandler\n')
	DataHandler().test()

	print('\n##############################################################################')
	print('TEST: Experience_buffer\n')
	Experience_buffer().test()


