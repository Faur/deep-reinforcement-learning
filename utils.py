
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

def frame_scip(k=4):
	"""The agent only observes every k'th frame, and every action is repeated k times"""

class Annealer():
	"""Simple class that helps with annealing"""
	def __init__(self, initial_value, end_value, period):
		self.initial_value = initial_value
		self.epsilon = initial_value
		self.end_value = end_value
		self.period = float(period)

	def linear(self, step):
		if self.epsilon < self.end_value:
			return self.end_value
		self.epsilon = self.initial_value - self.initial_value*step/self.period
		return self.epsilon

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
        """ Add ONE experience to the experience buffer"""
        self.buffer.append(experience)
        while len(self.buffer) > self.buffer_capacity:
            self.buffer.pop(0)
    
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
        return batch
                
    def test():
        """ Test the following features:
                __str__
                add
                sample
        """
        print('Testing Experience_buffer')

        
        buffer = Experience_buffer(5)
        for i in range(8):
            experience = {'obs':i, 'action':i, 'reward':i, 'next_obs':i, 'done':i}
            print(i, buffer.buffer_size())
            buffer.add(experience)
        
        print('\nContent of buffer')
        print(buffer)
        
        print('A sample')
        batch = buffer.sample(5)
        print(type(batch['obs'])) # should retunr a list
        print(batch)



