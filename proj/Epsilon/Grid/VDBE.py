import numpy as np
from scipy.interpolate import RegularGridInterpolator

class epsilon(RegularGridInterpolator):

  def __init__(self, n_nodes, lows, highs, init_value = 1.0, min_value = 0.01, decay = 0.9):

    points, values = mesh(n_nodes, lows, highs, init_value)

    super().__init__(points, values)
    self.dim = len(points)
    self.points = points
    self.prev_value = np.zeros(n_nodes)
    self.k = np.zeros(n_nodes)
    self.take_random = np.ones(n_nodes, dtype=bool)
    self.sigma = 0.5
    self.alpha = 0.6
    
    
  def take_random_action(self, state):
    node = self.get_nearest_node(state)
    return self.take_random[node]
        
  def step_update(self, state, value):
    node = self.get_nearest_node(state)
    delta = value - self.prev_value[node]
    f = (1 - np.exp(- abs(self.alpha * delta))) / \
        (1 + np.exp(- abs(self.alpha * delta)))
    self.values[node] = self.sigma * f + (1 - self.sigma) * self.values[node]
    self.prev_value[node] = value
    
  def episode_update(self, states, value):
    pass

  
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
 
def mesh(n_nodes, lows, highs, init_value):
    points = []
    for i in range(len(n_nodes)):
        points.append(np.linspace(lows[i], highs[i], n_nodes[i]))
        values = init_value * np.ones(n_nodes)
    return points, values