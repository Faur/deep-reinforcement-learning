



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
