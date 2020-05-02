import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt


SCORES_DIR = os.path.join('.', 'plot_scores')
SAVE_DIR   = os.path.join('.', 'plots')

def plot_moving_average(length):
  files = os.listdir(SCORES_DIR)
  xs = None

  for file in os.listdir(SCORES_DIR):
    scores = np.loadtxt(os.path.join(SCORES_DIR, file))
    moving_avgs = moving_averages(scores, length)
    if xs is None:
      xs = moving_avgs
    else:
      xs = xs + moving_avgs

  xs = xs / len(files)

  plt.figure(figsize = (8, 4))
  plt.plot(xs)
  plt.xlabel("Episode")
  plt.ylabel("Moving average episode score (length %d)" % length)

  fname = dt.datetime.today().strftime("%Y-%m-%d-%X") \
      + ":moving_averages_%d" % length \
      + ".pdf"
  fpath = os.path.join(SAVE_DIR, fname)

  plt.savefig(fpath)
  print("Saved figure as", fpath)
    
    
def moving_averages(xs, length):
  ys = np.zeros(len(xs))
  for i in range(len(xs)):
    l = min(i, length)
    ys[i] = np.mean(xs[i - l: i + 1])

  return ys


if __name__ == '__main__':
  plot_moving_average(50)
