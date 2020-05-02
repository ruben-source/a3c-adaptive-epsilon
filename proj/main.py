import os, stat
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import matplotlib.pyplot as plt
from A3C import train, test
import numpy as np
import datetime as dt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ENVIRONMENT = 'LunarLander-v2'
ENVIRONMENT = 'CartPole-v1'

parser = argparse.ArgumentParser()
parser.add_argument('--test', dest = 'test_model', type = str)
parser.add_argument('--freq', dest = 'freq', default = 20, type = int)
parser.add_argument('--avg', dest = 'num_trainings', default = 1, type = int)
parser.add_argument('--stochastic', dest = 'stoc', action = 'store_true')
parser.add_argument('--Tmax', dest = 'global_T_max', default = 200, type = int)
parser.add_argument('--eps', dest = 'epsilons', default = 'epsilon_greedy', type=str)
parser.add_argument('--num_agents', dest = 'num_agents', default = 4, type = int)
args = parser.parse_args()

# Setup for saving models during training
save_dir = os.path.join('.', ENVIRONMENT)
print(save_dir)
try:
  os.mkdir(save_dir)
  print("Made directory %s" % save_dir)
except FileExistsError:
  print("Directory %s already exists, skipping creation" % save_dir)
try:
  os.mkdir(os.path.join(save_dir, 'training_episode_scores'))
except FileExistsError:
  print("Directory exists")
try:
  os.mkdir(os.path.join(save_dir, 'figures'))
  print("Made directory %s" % save_dir)
except FileExistsError:
  print("Directory %s already exists, skipping creation" % os.path.join(save_dir, 'figures'))

  
  
if __name__ == '__main__':
  
  if args.test_model is not None:
    # test(args.test_model)
    test()

  else:
    for _ in range(args.num_trainings):
      
      assert args.num_agents >= 2 #"should be asyncrounus training with atleast 2 agents"
      
      print("Starting training")
      score_history = train(args, save_dir, args.num_agents)

      moving_avg_scores = []
      moving_avg_len    = 50
      for i in range(len(score_history)):
        l = min(i, moving_avg_len)
        moving_avg_scores.append( np.mean(score_history[i - l: i + 1]) )

      fname = dt.datetime.today().strftime("%Y-%m-%d-%X") \
          + ":training_scores"
      np.savetxt(os.path.join(save_dir, 'training_episode_scores', fname), score_history)

      plt.figure(figsize = (8, 4))
      plt.plot(score_history)
      plt.xlabel("Episode")
      plt.ylabel("Episode score")
      plt.savefig(os.path.join(save_dir, 'figures', 'scores.pdf'))

      plt.figure(figsize = (8, 4))
      plt.plot(moving_avg_scores)
      plt.xlabel("Episode")
      plt.ylabel("Moving average episode score (length %d)" % moving_avg_len)
      plt.savefig(os.path.join(save_dir, 'figures', 'moving_avg_scores.pdf'))
      print("Saved figures in", os.path.join(save_dir, 'figures'))

  print("Done")