

import tensorflow as tf
import gym

import numpy as np
import multiprocessing as mp
import time

from worker import worker_main
from coordinator import coordinator
from Epsilon.Grid import greedy, greedy_modified, explorer, exploiter, adaptiv, VDBE
from Epsilon.Time import greedy as tgreedy, adaptive as tadaptive

def train(args,save_dir, num_agents):
  
  score_history_pipe = mp.Pipe()
  score_history_conn = score_history_pipe[0]

  gradient_queue =  mp.Queue()
  scores_queue   =  mp.Queue()
  exit_queue     =  mp.Queue()
  sync_pipes     = [mp.Pipe() for _ in range(num_agents)]
  connections_0  = [c[0] for c in sync_pipes]
  connections_1  = [c[1] for c in sync_pipes]

  global_T = mp.Value('i', 0)

  global_process = mp.Process(
      target = coordinator,
      args   = (num_agents, gradient_queue, 
                scores_queue, exit_queue, 
                connections_0, score_history_pipe[1],
                global_T)
  )
  global_process.start()
  print("Started global process")
  
  n_nodes = (50, 50, 50, 50)
  lows    = [-4.8, -0.5, -0.42, -0.5]
  highs   = [ 4.8,  3.0,  0.42,  1.0]
  num_agents = 4
  eps = [] # (n_nodes, lows, highs, init_value= 0.5)
  eps.append(explorer.epsilon(n_nodes, lows, highs, init_value= 0.5))
  eps.append(exploiter.epsilon(n_nodes, lows, highs, init_value= 0.5))
  eps.append(explorer.epsilon(n_nodes, lows, highs, init_value= 0.5))
  eps.append(exploiter.epsilon(n_nodes, lows, highs, init_value= 0.5))
    
  processes = []
  for id in range(num_agents):
    proc = mp.Process(
        target = worker_main,
        args   = (id,args,save_dir, gradient_queue,
                  scores_queue, exit_queue,
                  connections_1[id], global_T,
                  eps[id])
    )
    processes.append(proc)
    proc.start()
    print("Started process %d" % id)

  for id in range(num_agents):
    processes[id].join()
    print("Joined process %d" % id)

  training_score_history = score_history_conn.recv()
  global_process.join()
  print("Joined global process")

  return training_score_history

def test():
  env = gym.make(ENVIRONMENT)
  print("Globl agent made environment")

  num_actions = env.action_space.n
  num_states  = env.observation_space.shape[0]
  global_agent = Agent(
      num_actions,
      num_states
  )
  global_agent.combined_model.load_weights(os.path.join(save_dir, './tmpa3c'))
  for _ in range(4):
    score = 0
    state = env.reset()
    state = np.reshape(state, (1, num_states))
    terminated = False
    while not terminated:
      predictions = global_agent.combined_model(tf.convert_to_tensor(state))
      logits      = predictions[1]
      probs       = tf.nn.softmax(logits)
      env.render(mode = 'close')
      time.sleep(1/50)
      if args.stoc:
        action = np.random.choice(num_actions, p = probs.numpy()[0])
      else:
        action = np.argmax(probs.numpy())
      next_state, reward, terminated, _ = env.step(action)
      score = score + reward
      state = next_state
      state = np.reshape(state, (1, num_states))
    print("Final test done with score %f" % score)





