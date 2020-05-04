import tensorflow as tf
import gym
import os
import numpy as np
from Agent import Agent
from Epsilon import epsilon as epsilon
# from A3C import ENVIRONMENT
ENVIRONMENT = 'CartPole-v1'
def worker_main(id, args, save_dir,
                gradient_queue, scores_queue,
                exit_queue, sync_connection,
                global_T, epsilon):

  # epsilon_min   = 0.01
  # epsilon_decay = 0.99
  # # epsilon_decay = 1.0
  # epsilon_init = 1.0
  # eps = epsilon_init
  # epsilon_decay_time = 5000
  eps = epsilon

  gamma = 0.99

  t_max = args.freq
  max_episode_length = 2000  # NOTE not used

  best_episode_score = 0

  env = gym.make(ENVIRONMENT)
  env._max_episode_steps = 2000
  print("Agent %d made environment" % id)
  # env.seed(0)
  num_actions = env.action_space.n
  num_states = env.observation_space.shape[0]

  agent = Agent(
      num_actions,
      num_states
  )
  print("Agent %d made agent" % id)

  # Reset local gradients
  combined_gradients = \
      [tf.zeros_like(tw) for tw in agent.combined_model.trainable_weights]

  while global_T.value < args.global_T_max:
    with global_T.get_lock():
      global_T.value += 1
      current_episode = global_T.value

    state = env.reset()
    state = np.reshape(state, (1, num_states))

    state_buffer  = []
    action_buffer = []
    reward_buffer = []
    score = 0

    t = 0  # local time step counter
    terminated = False

    while not terminated:

      predictions = agent.combined_model(tf.convert_to_tensor(state))
      value       = predictions[0]
      logits      = predictions[1]
      probs       = tf.nn.softmax(logits)

      # if id == 0:
        # env.render(mode = 'close')

      if args.stoc:
        action = np.random.choice(num_actions, p = probs.numpy()[0])
      else:
        # if np.random.rand() <= eps:
          # action = np.random.randint(num_actions)
        # else:
          # action = np.argmax(probs.numpy())
        # if eps > eps_min:
          # eps = eps * eps_decay
          # eps = eps - (epsilon_init - epsilon_min) / epsilon_decay_time
        if eps.take_random_action(state[0]):
          action = np.random.randint(num_actions)
        else:
          action = np.argmax(probs.numpy())
        
      next_state, reward, terminated, _ = env.step(action)

      #if terminated:
       # reward = -1
      new_state = np.reshape(next_state, (1, num_states))
      value = agent.combined_model(tf.convert_to_tensor(new_state))[0]
      eps.step_update(state[0], value)

                
      state_buffer.append(state)
      action_buffer.append(action)
      reward_buffer.append(reward)
      score = score + reward

      t = t + 1

      if t == t_max or terminated:

        # Compute and send gradients
        cum_reward = \
            0 if terminated \
              else agent.combined_model(tf.convert_to_tensor(state))[0]

        for i in reversed(range(t)):
          cum_reward = reward_buffer[i] + gamma * cum_reward

          combined_gradients = add_gradients(
              combined_gradients,
              agent.combined_gradients(
                  state_buffer[i],
                  action_buffer[i],
                  cum_reward
              )
          )

        combined_gradients = mult_gradients(combined_gradients, 1 / t)
        # Send local gradients to global agent
        gradient_queue.put(combined_gradients)

        # Reset local gradients
        combined_gradients = \
            [tf.zeros_like(tw) for tw in agent.combined_model.trainable_weights]

        # Request weights from global agent. 1 is a dummy
        sync_connection.send(1)
        
        eps.episode_update(state_buffer[0], cum_reward)
        
        if  terminated:
          #print("Agent %d, episode %d/%d, got score %f" % (id, current_episode, args.global_T_max, score))
          scores_queue.put(score)
          
          # Save model if better than previous
          # (may be false due to frequent updating)
          if score > best_episode_score:
            print("Agent %d at episode %d/%d saving local model with score %.0f" % (id, current_episode, args.global_T_max, score))
            agent.combined_model.save_weights(
                os.path.join(save_dir, 'tmpa3c')
            )
            # agent.combined_model.save_weights(
                # os.path.join(save_dir, 'model_score_%.0f' % score)
            # )
            best_episode_score = score

        # Synchronise local and global parameters
        weights = sync_connection.recv()
        agent.combined_model.set_weights(weights)

        state_buffer  = []
        action_buffer = []
        reward_buffer = []
        
        # Reset update timer
        t = 0

      state = next_state
      state = np.reshape(state, (1, num_states))

  exit_queue.put(id)
  print("Agent %d quit" % id)

# Elementwise addition of list of tensors
def add_gradients(xs, ys):
  return list( map(lambda pair: pair[0] + pair[1], zip(xs, ys)) )

def mult_gradients(xs, fac):
  return list( map(lambda x: x * fac, xs) )