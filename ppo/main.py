
"""
    Implementations:

    https://keras.io/examples/rl/actor_critic_cartpole/
    https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
    The latter seems to be clearer.
"""

import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
import time
import numpy as np
from tensorflow.keras.models import clone_model
import sys

EPSILON = sys.float_info.epsilon # smallest number such that 1+EPSILON = 1

# Keras model
num_inputs = 4
num_actions = 2
num_hidden = 32

# Could be pretty much any architecture as long as it has the same number of inputs and outputs
inputs = Input(shape=(num_inputs,))
common1 = Dense(64, activation='relu')(inputs)
common2 = Dense(32, activation='relu')(common1)
common3 = Dense(16, activation='relu')(common1)
actor = Dense(num_actions, activation='softmax')(common3)
critic = Dense(1)(common2)

model = Model(inputs=inputs, outputs=[actor, critic])

actor_model = Model(inputs=inputs, outputs=[actor])
critic_model = Model(inputs=inputs, outputs=[critic])

optimizer = Adam(learning_rate=0.001)

actor_optimizer = Adam(learning_rate=0.01)
critic_optimizer = Adam(learning_rate=0.001)

mse = MeanSquaredError()

def predict(model, obs):
    obs = tf.convert_to_tensor(obs)
    obs = tf.expand_dims(obs, 0)

    return model(obs)

alpha = 0.05
gamma = 0.999
epsilon = 0.05

# https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
coeff_vf = 0.1 # Value function coefficient
coeff_entr = 0.03 # Entropy coefficient

"""
    --- ALPHA ---
    A smaller value of alpha means that the agent will make smaller updates to its estimates,
    which can make the learning process slower but more stable.
    A larger value of alpha means that the agent will make larger updates to its estimates,
    which can make the learning process faster but less stable.

    --- GAMMA ---
    A value of 0 means that the agent is only interested in the immediate rewards,
    while a value of 1 means that the agent is only interested in the long-term rewards.

    --- EPSILON ---
    A value that determines the maximum deviation allowed between
    the old and new policy probability distributions during the policy update step.
    This defines the clipping range.
"""

history = [] # Filled with tuples representing episode... => (reward, critic_loss, actor_loss, td_error, td_target, ms)

max_episodes = 200

env = gym.make("CartPole-v1", render_mode='human')
# env = gym.make("CartPole-v1") # no render

times = []

old_agent = clone_model(model)
old_agent.set_weights(model.get_weights())

for episode in range(max_episodes):

    observation, info = env.reset()
    terminated = False
    episode_reward = 0
    
    start = time.time()
    while not terminated:
        with tf.GradientTape(persistent=True) as tape:
            actor_probs, critic_value = predict(model, observation)
            action = np.random.choice(num_actions, p=np.squeeze(actor_probs))

            old_actor_probs, old_critic_value = predict(old_agent, observation)
            old_agent.set_weights(model.get_weights())

            observation, reward, terminated, truncated, info = env.step(action)
            next_actor_probs, next_critic_value = predict(model, observation)

            episode_reward += reward
            
            # TD(0) update rule: critic_value + alpha * (reward + gamma * next_critic_value - critic_value)
            # Simplified: (1 - alpha)*critic_value+ alpha*(reward + gamma*next_critic_value)
            td_target = reward + gamma*(0 if terminated else next_critic_value) # TD target is the value that the agent's estimate of the value function is being updated towards. 
            td_error = td_target - critic_value # Advantage = Aπ(s,a) = Qπ(s,a)-Vπ(s,a) # https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#advantage-functions
            # value = critic_value + alpha*td_error
            
            # https://stackoverflow.com/a/50663200/10603899
            ratio = actor_probs/(old_actor_probs+EPSILON) # Instead of negative probabilities we divide previous and actual actor_probs
            clipped_ratio = np.clip(ratio, 1-epsilon, 1+epsilon)
            
            critic_loss = mse([td_target], [critic_value])

            l_entropy = -actor_probs[0]*tf.math.log(actor_probs[0]+EPSILON)
            advantage = td_error
            l_clip = tf.minimum(ratio*advantage, clipped_ratio*advantage)

            # https://pylessons.com/PPO-reinforcement-learning
            # We don't use the VF with CLIP because actor and critic gradients are applied separatedly
            # l_value_function = critic_loss
            # l_clip - coeff_vf*l_value_function + coeff_entr*l_entropy
            clipped_surrogate_objective = l_clip + coeff_entr*l_entropy
            actor_loss = tf.reduce_mean(clipped_surrogate_objective)
            
        # grads = tape.gradient(model_loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))

        actor_grads = tape.gradient(actor_loss, actor_model.trainable_variables)
        critic_grads = tape.gradient(critic_loss, critic_model.trainable_variables)
        
        critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))
        actor_optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))

    history.append((
        episode_reward,
        critic_loss.numpy(),
        np.squeeze(actor_loss),
        np.squeeze(td_error),
        td_target,
        time.time()-start
    ))