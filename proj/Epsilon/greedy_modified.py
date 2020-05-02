import numpy as np
from scipy.interpolate import RegularGridInterpolator

class epsilon(RegularGridInterpolator):

  def __init__(self, n_nodes, lows, highs, init_value = 1.0, min_value = 0.01, decay = 0.9):

    points, values = mesh(n_nodes, lows, highs, init_value)

    super().__init__(points, values)
    self.points = points
    self.dim = len(points)
    self.min = min_value
    self.decay = decay
    self.best_value = 0
    self.prev_value = 0
    self.update_counter = np.zeros(n_nodes)
    
    
  def take_random_action(self, state):
    return np.random.uniform() < self.__call__(state)
  
  def set_values(self, values):
    self.values = copy.copy(values)
  
  def step_update(self, state):
    epsilon = self.__call__(state)
    if  epsilon > self.min:
        nearest_node = self.get_nearest_node(state)
        self.update_counter[nearest_node] += 1
        new_epsilon = self.min \
                       + (0.5 - self.min) \
                       * np.exp( - (1 - self.decay) \
                                * self.update_counter[nearest_node])
        #new_epsilon = self.__call__(state) * self.decay
        self.update_nearest(state, new_epsilon)
    
  def episode_update(self, states, value):
    
    for state in states:
        if value - self.prev_value*2 > 0: # 2x improvment
            new_val = self.__call__(state) * self.decay * self.decay
            self.update_nearest(state, max(self.min,new_val))
        else:
            new_val = self.__call__(state) * (2-self.decay)
            self.update_nearest(state, min(0.5,new_val))
        
        delta = (value - self.best_value) * 7            
        if delta > 0:
            self.best_value = value
            new_value = max(self.min, 1 / (1 + np.exp(-2 * delta)))
            self.update_nearest(state, new_value)
        else: 
            self.update_nearest(state, min(0.5, self.__call__(state) * 1.5))
    self.prev_value = value
        
  def __call__(self, xi):
    # RegularGridInteroplators take arrays of points as inputs. Here I assume
    # that we will only call GridEpsilon on single points, which is why xi is
    # placed in a singleton array.
    for i in range(self.dim):
        xi[i] = min(self.points[i][-1], max(self.points[i][0], xi[i]))
    
    return super().__call__(np.array([xi]))
  
  def get_nearest_node(self, xi):
    xi_indices = np.zeros(self.dim, dtype = int)
    for i in range(self.dim):

      dim_min = self.points[i][0]
      dim_max = self.points[i][-1]
      dim_len = len(self.points[i]) - 1
      index = (xi[i] - dim_min) / (dim_max - dim_min) * dim_len
      index = np.round(index)
      xi_indices[i] = min(dim_len, max(0, index))
    return tuple(xi_indices)
    
  def update_nearest(self, xi, value):

    node = self.get_nearest_node(xi)
    # self.values[tuple(xi_indices)] = value
    self.values[node] = value

def mesh(n_nodes, lows, highs, init_value):
    points = []
    for i in range(len(n_nodes)):
        points.append(np.linspace(lows[i], highs[i], n_nodes[i]))
        values = init_value * np.ones(n_nodes)
    return points, values

