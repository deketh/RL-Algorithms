# Implementations:

# https://keras.io/examples/rl/actor_critic_cartpole/
# https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
# The latter seems to be clearer.

import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
import time
import numpy as np

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

# --- ALPHA ---
# A smaller value of alpha means that the agent will make smaller updates to its estimates,
# which can make the learning process slower but more stable.
# A larger value of alpha means that the agent will make larger updates to its estimates,
# which can make the learning process faster but less stable.

# --- GAMMA ---
# A value of 0 means that the agent is only interested in the immediate rewards,
# while a value of 1 means that the agent is only interested in the long-term rewards.

history = [] # Filled with tuples representing episode... => (reward, critic_loss, actor_loss, td_error, td_target, ms)

max_episodes = 100

env = gym.make("CartPole-v1", render_mode='human')
# env = gym.make("CartPole-v1") # no render

times = []

# 0.20s per episode approx.
for episode in range(max_episodes):

    observation, info = env.reset()
    terminated = False
    episode_reward = 0
    
    start = time.time()
    while not terminated:
        with tf.GradientTape(persistent=True) as tape:
            actor_probs, critic_value = predict(model, observation)
            action = np.random.choice(num_actions, p=np.squeeze(actor_probs))

            observation, reward, terminated, truncated, info = env.step(action)
            next_actor_probs, next_critic_value = predict(model, observation)

            episode_reward += reward
            
            # TD(0) update rule: critic_value + alpha * (reward + gamma * next_critic_value - critic_value)
            # Simplified: (1 - alpha)*critic_value+ alpha*(reward + gamma*next_critic_value)
            td_target = reward + gamma*(0 if terminated else next_critic_value) # TD target is the value that the agent's estimate of the value function is being updated towards. 
            td_error = td_target - critic_value # Advantage = Aπ(s,a) = Qπ(s,a)-Vπ(s,a) # https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#advantage-functions
            # value = critic_value + alpha*td_error
            
            critic_loss = mse([td_target], [critic_value])

            # https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic#the_actor_loss
            log_prob = tf.math.log(actor_probs[0, action]) # sliced because of expanded dims
            advantage = td_error
            actor_loss = -log_prob*advantage

            # https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic#3_the_actor-critic_loss
            # model_loss = actor_loss + critic_loss
            
        # grads = tape.gradient(model_loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))

        actor_grads = tape.gradient(actor_loss, actor_model.trainable_variables)
        critic_grads = tape.gradient(critic_loss, critic_model.trainable_variables)
        
        optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))
        optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))

    history.append((
        episode_reward,
        critic_loss.numpy(),
        np.squeeze(actor_loss),
        np.squeeze(td_error),
        td_target,
        time.time()-start
    ))