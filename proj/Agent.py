import tensorflow as tf
# import gym
# from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, linear # , softmax

from tensorflow.keras.backend import manual_variable_initialization
manual_variable_initialization(True)
tf.keras.backend.set_floatx('float64')

class Agent:

  def __init__(
      self,
      num_actions,
      num_states,
      learning_rate = 0.001):

    manual_variable_initialization(True)  # TODO is this needed?

    self.num_actions   = num_actions
    self.num_states    = num_states
    self.optimizer     = Adam(learning_rate = learning_rate)


    ins    = Input(shape = (num_states,))
    l1     = Dense(100,          activation = relu   )(ins)
    l2     = Dense(100,          activation = relu   )(ins)
    value  = Dense(1,            activation = linear )(l1)
    # NOTE policy output is unnormalised probabilities; use softmax explicitly
    policy = Dense(num_actions,  activation = linear)(l2)
    
    self.combined_model = Model(inputs = ins, outputs = [value, policy])

  def combined_gradients(self, state, action, cum_reward):
    with tf.GradientTape() as tape:
      tape.watch(self.combined_model.trainable_weights)

      predictions = self.combined_model(tf.convert_to_tensor(state, dtype = tf.float64))
      value       = predictions[0]
      policy      = predictions[1]
      probs       = tf.nn.softmax(policy)

      advantage = tf.convert_to_tensor(cum_reward, dtype = tf.float64) - value

      v_loss = tf.math.pow(advantage, 2)

      actions_index = tf.one_hot(
          tf.convert_to_tensor(action),
          self.num_actions
      )

      # non-sparse takes index vector; sparse takes a single index
      p_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels = tf.convert_to_tensor([action]),
          logits = policy
      )
      p_loss = p_loss * tf.stop_gradient(advantage)

      loss = 0.5 * p_loss + 0.5 * v_loss
    grad = tape.gradient(loss, self.combined_model.trainable_weights)
    return grad

  # Should only be used on the global agent.
  def apply_combined_gradients(self, gradients):
    self.optimizer.apply_gradients(
        zip(gradients, self.combined_model.trainable_weights)
    )
