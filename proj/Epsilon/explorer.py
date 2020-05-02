import numpy as np
from scipy.interpolate import RegularGridInterpolator

class epsilon(RegularGridInterpolator):

  def __init__(self, n_nodes, lows, highs, init_value = 1.0, min_value = 0.01, decay = 0.9):

    points, values = mesh(n_nodes, lows, highs, init_value)

    super().__init__(points, values)
    self.dim = len(points)
    self.points = points
    self.min = min_value
    self.decay = decay
    self.best_value = 0
    self.prev_value = 0
    self.k = 0
    self.num_explore = 100
    self.update_counter = np.zeros(n_nodes)
    
    
  def take_random_action(self, state):
    node = self.get_nearest_node(state)
    return self.update_counter[node] > self.num_explore # np.random.uniform() < self.__call__(state)
  
  def set_values(self, values):
    #self.values = copy.copy(values)
   pass
   
  def step_update(self, state):
        # only updates itself if a node has been visited more than a certain number of times
    node = self.get_nearest_node(state)
    self.update_counter[node] += 1
    if self.update_counter[node] > self.num_explore:
        self.values[node] = self.values[node] * self.decay
    
  def episode_update(self, states, value):
    
    pass
        
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
