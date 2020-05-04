import numpy as np

class epsilon():

  def __init__(self, init_value = 1.0):

    self.epsilon = init_value
    self.max_prev = 0
    self.k = 0
    self.m = 0
    self.take_random = True
    self.episode_count = 0
    
  def take_random_action(self, state):
    return self.take_random
        
  def step_update(self, state, value):
    f, l= 7, 10#10 # kommer från pappret, de testade sig fram så de kanske är konstiga för oss
    
    if np.random.uniform() <= self.epsilon:
        max_curr = value
        self.k += 1 
        if self.k == l:
            delta = (value - self.max_prev) * f
            if delta > 0:
                # sigmoid update
                self.epsilon = 1 / (1 + np.exp(-2*delta))
            else:
                if delta < 0:
                    # if the new value is not better then the last value after
                    # l or more visits, then reset it. 
                    # new_val = min(0.5, self.__call__(point) *  1.5)
                    if self.m == 10:
                        self.epsilon = 0.5
                        self.m = 0
                    else:
                        self.episode = self.epsilon ** (self.episode_count * 4)
                        self.m += 1 
            self.max_prev = max_curr
            self.k = 0
        self.take_random = True
    else:
        self.take_random = False
    
  def episode_update(self, states, value):
    self.episode_count += 1
  