
import numpy as np


def clip_rewards():
	""" Clip rewards between [-1, 1]"""

def crop():
	pass

def remove_color():
	"""
	Input: [w, h, c]

	Output: [w, h, 1]
	"""

def max_frames():
	"""Combines two frames into one by elementwise max. Used because some sprites don't show on all frames.
	"""

def frame_skip(k=4):
	"""The agent only observes every k'th frame, and every action is repeated k times"""

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


if __name__ =='__main__':
	Experience_buffer().test()


