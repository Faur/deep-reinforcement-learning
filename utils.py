from __future__ import absolute_import, division, print_function, unicode_literals

import os
import datetime, time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import toimage, fromimage
import gym

def time_str():
	return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-(%H-%M-%S)')

def print_attributes(C):
	print(C.__name__, 'attribures:')
	for v in vars(C):
		if not v.startswith('__'):
			print('\t', v)
			
def discount_rewards(r, gamma):
	""" Takes the rewards of an entire episoe, and discounts them."""
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(len(r))):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add

	# TODO: assert that all the values aren't the same.. this messes things up for some reason
	return discounted_r

def gif_from_figs(figs, target, duration=None):
	import imageio
	with imageio.get_writer(target, duration=None) as writer:
		for i, f in enumerate(figs):
			f.canvas.draw()
			data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
			data = data.reshape(f.canvas.get_width_height()[::-1] + (3,))
			writer.append_data(data)

def num_trainable_param():
	""" Counts the number of trainable parameters in the current graph
		From: https://stackoverflow.com/a/38161314/3747801
	"""
	total_parameters = 0
	for variable in tf.trainable_variables():
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parametes = 1
		for dim in shape:
			variable_parametes *= dim.value
		total_parameters += variable_parametes
	return total_parameters

class Annealer():
	"""Simple class that helps with annealing"""
	def __init__(self, initial_value, end_value, period):
		self.initial_value = initial_value
		self.epsilon = initial_value
		self.end_value = end_value
		self.period = float(period)


	def linear(self, step):
		slope = (self.end_value - self.initial_value)/self.period
		epsilon = slope*step + self.initial_value
		return max(epsilon, self.end_value)

class Experience_buffer():
	""" Consists of a list of 'experience', each of which is a dict.
		The keys should be the same for experience, but are not set in advance
	"""
	def __init__(self, buffer_capacity=int(1e6)):
		self.buffer = []
		self.buffer_capacity = buffer_capacity
	
	def __str__(self):
		""" Simply the content of the buffer"""
		tostr = ''
		for item in self.buffer:
			tostr += item.__str__()
			tostr += '\n'
		return tostr

	def buffer_size(self):
		""" Return the current number of experiences in the buffer."""
		return len(self.buffer)
	
	def is_full(self):
		if self.buffer_size() >= self.buffer_capacity:
			return True
		else:
			return False

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
		obs = None
		for i in range(5):
			obs = np.array([i])
			experience = {'obs':obs, 'action':i, 'reward':i, 'next_obs':obs, 'done':i}
			print(i, buffer.buffer_size())
			buffer.add(experience)
		print('\nContent of buffer')
		print(buffer)

		print('Test 1d obs')
		print('obs.shape', obs.shape)
		batch = buffer.sample(5)
		print(type(batch['obs']))
		print('Should not have superflouous dimensions')
		print(batch['obs'].shape)
		assert len(batch['obs'].shape) == 1, "batch['obs'] has wrong len!"
		buffer.clear()
		print('\nTest buffer.clear: len = ' + str(buffer.buffer_size()))

		print('\nTest 1d obs')
		for i in range(5):
			obs = np.array([i,i])
			experience = {'obs':[obs], 'action':i, 'reward':i, 'next_obs':[obs], 'done':i}
			buffer.add(experience)
		batch = buffer.sample(5)
		print('obs.shape', obs.shape)
		print(type(batch['obs'])) 
		print(batch['obs'].shape)
		assert len(batch['obs'].shape) == 2, "batch['obs'] has wrong len!"
		buffer.clear()

		print("NB: 2d obs shouldn't exist!!")
		print()

		print('\nTest 3d obs')
		for i in range(5):
			obs = np.array([[[i,i], [i,i]], [[i,i], [i,i]]])
			experience = {'obs':[obs], 'action':i, 'reward':i, 'next_obs':[obs], 'done':i}
			buffer.add(experience)
		batch = buffer.sample(5)
		print('obs.shape', obs.shape)
		print(type(batch['obs']))
		print(batch['obs'].shape)
		assert len(batch['obs'].shape) == 4, "batch['obs'] has wrong len!"
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
	def __init__(self, obs_shape, buffer_size=1):
		self.obs_shape = obs_shape
		self.obs_dim = len(obs_shape)
		self.buffer_size = buffer_size
		if self.obs_dim != 3:
			assert self.buffer_size == 1, 'ERROR: ObsBuffer: buffer_size '\
				+ 'must be 1 when obs_dim != 3.\n'+ 'buffer_size: ' \
				+ str(self.buffer_size) + '. obs_dim: ' + str(self.obs_dim)

		# Initialize an empty buffer
		self.buffer = []
		self.reset()
	
	def add(self, obs):
		assert type(obs) is np.ndarray, 'ERROR: ObsBuffer: obs must be an np.ndarray'
		self.buffer.append(obs)
		# Remove excess obs
		while len(self.buffer) > self.buffer_size:
			self.buffer.pop(0)

	def get(self):
		""" for 1D obs output is
				obs_shape
			for 2D obs output is
				obs_shape + [buffer_size]
		"""
		# TODO: This is kinda shit?
		if self.obs_dim == 1: # 1D input
			assert self.buffer_size == 1, "ERROR: ObsBuffer: self.buffer_size != 1."\
				+ ' For 1D obs'
			return self.buffer[0]
		elif self.obs_dim == 3: # 2D input
			return np.dstack(self.buffer)
		else:
			raise Exception('ERROR: ObsBuffer.get(): self.obs_dim == ' + str(self.obs_dim))
	
	def reset(self):
		""" Fill the buffer with zeros
		"""
		self.buffer = [np.zeros(self.obs_shape) \
			for i in range(self.buffer_size)]

	def test(self):
		print('\n##############################################################################')
		print('TEST: ObsBuffer\n')

		print('1D obs')
		obs_shape = [10]
		buf_size = 1
		obs = np.ones(obs_shape)
		obsBuf = ObsBuffer(obs_shape, buf_size)
		print('obsBuf.buffer   ', type(obsBuf.buffer), len(obsBuf.buffer))
		print('obsBuf.buffer[0]', type(obsBuf.buffer[0]), obsBuf.buffer[0].shape)
		print('obsBuf.get	   ', type(obsBuf.get()), obsBuf.get().shape)
		print('^^^ shoudl be:  ', obs_shape[:-1] + [obs_shape[-1]*buf_size])
		print('sum before add  ', np.sum(obsBuf.get()))
		obsBuf.add(obs)
		print('sum after add   ', np.sum(obsBuf.get()))
		obsBuf.reset()
		print('sum after reset', np.sum(obsBuf.get()))
		print()

		print('2D obs')
		obs_shape = [10, 10, 2]
		buf_size = 4
		obs = np.ones(obs_shape)
		obsBuf = ObsBuffer(obs_shape, buf_size)
		print('obsBuf.buffer   ', type(obsBuf.buffer), len(obsBuf.buffer))
		print('obsBuf.buffer[0]', type(obsBuf.buffer[0]), obsBuf.buffer[0].shape)
		print('obsBuf.get	   ', type(obsBuf.get()), obsBuf.get().shape)
		print('^^^ shoudl be:  ', obs_shape[:-1] + [obs_shape[-1]*buf_size])
		print('sum before add  ', np.sum(obsBuf.get()))
		obsBuf.add(obs)
		obsBuf.add(obs)
		obsBuf.add(obs)
		obsBuf.add(obs)
		obsBuf.add(obs)
		print('sum after add   ', np.sum(obsBuf.get()))
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
		return img
		
	def process(self, obs):
		pre = toimage(obs)
		pre = pre.resize(self.out_shape[:2])
		pre = fromimage(pre)
		if self.gray:
			pre = self._make_gray(pre)
		# For consistency an image ALWAYS has dimensions [w, h, c]!
		if len(pre.shape)==2:
			pre = np.expand_dims(pre, axis=-1)
		assert len(pre.shape)==3, "ERROR: Preprocessor_2d: pre has dim: " + str(pre.shape)
		pre = pre/255.
		return pre

	# def test(self):
	# 	print('\n##############################################################################')
	# 	print('TEST: Preprocessor_2d\n')
	# 	# get an observation
	# 	import gym
	# 	env = gym.make('Breakout-v0')
	# 	obs = env.reset()
	# 	preprocessor = Preprocessor_2d(out_shape=[84,84], gray=True)

	# 	print('Test preprocessing step')
	# 	pre = preprocessor.process(obs)
	# 	print('obs', type(obs), obs.shape, np.product(obs.shape))
	# 	print('pre', type(pre), end=' ')
	# 	print(pre.shape, np.product(pre.shape))
	# 	print((1.*np.product(pre.shape))/np.product(obs.shape), '% of original')
	# 	print()

	# 	# Visualize preprocessing
	# 	fig, ax = plt.subplots(1,2)
	# 	fig.suptitle('Preprocessor_2d test')
	# 	ax[0].imshow(obs)
	# 	ax[0].set_title('Original')
	# 	ax[1].imshow(np.squeeze(pre), cmap='gray')
	# 	ax[1].set_title('After Preprocessing')
	# 	plt.draw()


class EnvironmentInterface():
    def __init__(self, config, preprocessor=None, action_repeats=1, obs_buffer_size=1):
        #, merge_frames=False):
        """ 
        """
        self.env_name = config.env_name
        self.env = gym.make(config.env_name)
        self.preprocessor = preprocessor
        self.action_repeats = action_repeats
        self.obs_buffer_size = obs_buffer_size
        
        self.obs_dim = config.num_state
        self.single_obs_dim = config.num_state[:2] + [1]
        
        self.obsBuf = ObsBuffer(self.single_obs_dim, self.obs_buffer_size)
#         self.merge_frames = merge_frames
#         if self.merge_frames:
#             assert self.action_repeats > 1, 'ERROR: EnvironmentInterface: '\
#                 + 'Cannot merge frames with action_repeats !> 1.'\
#                 + 'action_repeats = ' + str(self.action_repeats)

    def step(self, action):
        """ Perform action self.action_repeats times times.
            Preprocess the last observation, and add it to the observation buffer.
            Return an observation from the observation buffer, cumulative reward
        """
        ## repeat action 
        obs = None
        total_reward = 0
        done = False
        infos = []

        # Repeat action self.action_repeats times
        for i in range(self.action_repeats):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            infos.append(info)
            if done: break

        ## Preprocess last obs
        if self.preprocessor is not None:
            obs = self.preprocessor.process(obs)
        
        ## Add processed obs to obsBuf
        self.obsBuf.add(obs)
        
        ## Return
        return self.obsBuf.get(), total_reward, done, infos
        
    def render(self, close=False, mode='human'):
        self.env.render(close=close, mode=mode)
        
    def reset(self):
        """Simple wrapper that restarts the environment"""
        obs = self.env.reset()

        if self.preprocessor is not None:
            obs = self.preprocessor.process(obs)
        
        self.obsBuf.reset()
        self.obsBuf.add(obs)
        
        return self.obsBuf.get()
    


if __name__=='__main__':
	print('Experiment start' + current_time)
	Experience_buffer().test()
	ObsBuffer([1], 1).test()
	Preprocessor_2d(None).test()
	EnvironmentInterface().test()
	plt.show()


