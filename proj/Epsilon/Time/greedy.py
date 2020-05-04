import numpy as np

class epsilon():

  def __init__(self, init_value = 1.0, min_value = 0.01, decay = 0.9998):

    self.epsilon = init_value
    self.min = min_value
    self.dim = len(points)
    self.max_prev = 0
    self.k = 0
    self.take_random = True
    
    
  def take_random_action(self, state):
    return numpy.random.uniform() < self.epsilon
        
  def step_update(self, state, value):
    if self.epsilon > self.min:
        self.epsilon *= decay
    
  def episode_update(self, states, value):
    pass
  