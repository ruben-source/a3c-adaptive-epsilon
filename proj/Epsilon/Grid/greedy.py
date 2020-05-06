import numpy as np
from scipy.interpolate import RegularGridInterpolator


class epsilon(RegularGridInterpolator):
  
  def __init__(self, n_nodes, lows, highs, init_value = 1.0, min_value = 0.01, decay = 0.99):
    points, values = mesh(n_nodes, lows, highs, init_value)
    super().__init__(points, values)
    self.dim = len(points)
    self.points = points
    self.min_value = min_value
    self.decay = decay

  def __call__(self, xi):
    # RegularGridInteroplators take arrays of points as inputs. Here I assume
    # that we will only call GridEpsilon on single points, which is why xi is
    # placed in a singleton array.
    for i in range(self.dim):
      xi[i] = min(self.points[i][-1], max(self.points[i][0], xi[i]))

    return super().__call__(np.array([xi]))

  def take_random_action(self, state):
    return np.random.uniform() < self.__call__(state)

  def step_update(self, state, value):
    value = self.__call__(state)
    if value > self.min_value:
      self.update_nearest(state, value * self.decay)
  
  def episode_update(self, states, value):
    pass
    
  def update_nearest(self, xi, value):

    xi_indices = np.zeros(self.dim, dtype = int)
    for i in range(self.dim):

      dim_min = self.points[i][0]
      dim_max = self.points[i][-1]
      dim_len = len(self.points[i]) - 1
      index = (xi[i] - dim_min) / (dim_max - dim_min) * dim_len
      index = round(index)
      xi_indices[i] = min(dim_len, max(0, index))
  
    self.values[tuple(xi_indices)] = value

def mesh(n_nodes, lows, highs, init_value):
    points = []
    for i in range(len(n_nodes)):
        points.append(np.linspace(lows[i], highs[i], n_nodes[i]))
        values = init_value * np.ones(n_nodes)
    return points, values



