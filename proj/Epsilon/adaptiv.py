import numpy as np
from scipy.interpolate import RegularGridInterpolator

class epsilon(RegularGridInterpolator):

  def __init__(self, n_nodes, lows, highs, init_value = 1.0, min_value = 0.01, decay = 0.9):

    points, values = mesh(n_nodes, lows, highs, init_value)

    super().__init__(points, values)
    self.dim = len(points)
    self.points = points
    self.max_prev = np.zeros(n_nodes)
    self.k = np.zeros(n_nodes)
    self.take_random = np.ones(n_nodes, dtype=bool)
    
    
  def take_random_action(self, state):
    node = self.get_nearest_node(state)
    return self.take_random[node]
        
  def step_update(self, state):
    pass
    
  def episode_update(self, states, value):
    
    f, l= 7, 3#10 # kommer från pappret, de testade sig fram så de kanske är konstiga för oss
    for point in states:
        node = self.get_nearest_node(point)
        if np.random.uniform() <= self.values[node]:
            max_curr = value
            self.k[node] += 1 
            if self.k[node] == l:
                delta = (value - self.max_prev[node]) * f
                if delta > 0:
                    # sigmoid update
                    self.values[node] = 1 / (1 + np.exp(-2*delta))
                    
                else:
                    if delta < 0:
                        # if the new value is not better then the last value after
                        # l or more visits, then reset it. 
                        # new_val = min(0.5, self.__call__(point) *  1.5)
                        self.update_nearest(point, 0.5)
                self.max_prev[node] = max_curr
                self.k[node] = 0
            self.take_random[node] = True
        else:
            self.take_random[node] = False

  def __call__(self, xi):
    # RegularGridInteroplators take arrays of points as inputs. Here I assume
    # that we will only call GridEpsilon on single points, which is why xi is
    # placed in a singleton array.
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