import gym
from Agent import Agent
import time
# from A3C import ENVIRONMENT
#import multiprocessing
ENVIRONMENT = 'CartPole-v1'
def coordinator(num_agents, gradient_queue, scores_queue, exit_queue, sync_connections, score_history_conn, global_T):

  env = gym.make(ENVIRONMENT)
  print("Globl agent made environment")
  # env.seed(0)

  # T = 0  # TODO global, shared between all processes
  gamma = 0.99
  save_freq = 20
  save_counter = 0

  num_actions = env.action_space.n
  num_states  = env.observation_space.shape[0]
  global_agent = Agent(
      num_actions,
      num_states
  )
  print("Global agent made agent")

  print("\n=======================================")
  print("   environment action space.n = ", env.action_space.n)
  print("   environment observation space.shape = ", env.observation_space.shape)
  print("=======================================\n")

  # Array keeping track of which local agents have finished all work
  has_exited = [False for _ in range(num_agents)]
  while not all(has_exited):

    for i in range(num_agents):
      if sync_connections[i].poll():
        _ = sync_connections[i].recv()

        weights = global_agent.combined_model.get_weights()
        sync_connections[i].send(weights)

    # Queue.empty() is unreliable according to docs, may cause bugs
    while not gradient_queue.empty():
      grads = gradient_queue.get()
      global_agent.apply_combined_gradients(grads)
      save_counter = save_counter + 1

    while not exit_queue.empty():
      exited_id = exit_queue.get()
      has_exited[exited_id] = True
      print("Global agent found agent %d has exited" % exited_id)

    time.sleep(0.2)

  scores = []
  while not scores_queue.empty():
    local_score = scores_queue.get()
    scores.append(local_score)

  score_history_conn.send(scores)
  print("Global agent is done")


