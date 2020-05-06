import numpy as np

class epsilon():

  def __init__(self, init_value = 0.5, min_value = 0.01, decay = 0.99):

    self.epsilon = init_value
    self.min = min_value
    self.decay = decay
    
  def take_random_action(self, state):
    return np.random.uniform() < self.epsilon
        
  def step_update(self, state, value):
    pass
    
  def episode_update(self, states, value):
    if self.epsilon > self.min:
        self.epsilon *= self.decay
  