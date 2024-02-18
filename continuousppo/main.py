# Initialization

import gym
from torch import from_numpy, no_grad
from continuousppo.buffer import ContinuousPPOBuffer
from continuousppo.model import ContinuousPPOModel
from continuousppo.trainer import ContinuousPPOTrainer

env = gym.make('LunarLander-v2', continuous=True) # render_mode='human'

BUFFER_SIZE = 2048

NUM_INPUTS = 8
NUM_ACTIONS = 2

train_reward_history = []
model = ContinuousPPOModel(num_inputs=NUM_INPUTS, num_actions=NUM_ACTIONS)

trainer = ContinuousPPOTrainer(
  model = model,
  num_inputs=NUM_INPUTS,
  num_actions=NUM_ACTIONS
)

buffer = ContinuousPPOBuffer(
  size=BUFFER_SIZE,
  num_inputs=NUM_INPUTS,
  num_actions=NUM_ACTIONS
)

# main.py
import builtins
from typing import Any

trainer.epsilon = 0.2
trainer.entropy_coeff = 0.0007
# trainer.value_function_coeff = 0.9
trainer.gamma = 0.99

trainer.set_means_lr(0.00025)
trainer.set_devs_lr(0.00025)
trainer.set_critic_lr(0.00025)

BATCH_SIZE = 512
EPOCHS = 3
TRAIN_TIMES = 29*6

def env_step(state: Any, buffer: ContinuousPPOBuffer, trainer: ContinuousPPOTrainer):
  actor, critic = trainer.model.predict(from_numpy(state))
  actions = trainer.model.sample_actions(actor)
  
  next_state, reward, terminated, truncated, _ = env.step(actions.numpy())

  means, devs = actor
  done = terminated or truncated

  buffer.append(state, next_state, means, devs, actions, critic, reward, done)

  return next_state, done

# Training loop

for i in range(TRAIN_TIMES):
  done = True
  buffer.initialize()

  # Populate buffer
  with no_grad():
    for i in range(buffer.size):

      if done:
        state, _ = env.reset()

      state, done = env_step(state, buffer, trainer)

    _, last_critic = trainer.model.predict(from_numpy(state), actor=False) # Inside no grad
  
  trainer.train(
    buffer=buffer,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    last_critic=last_critic, 
    normalize=True
  )
