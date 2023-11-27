import gymnasium as gym
from torch import from_numpy, no_grad
from continuous_ppo.model import ContinuousPPOModel

from continuous_ppo.trainer import ContinuousPPOTrainer

env = gym.make('LunarLander-v2', continuous=True) # render_mode='human'

BUFFER_SIZE = 10000

NUM_INPUTS = 8
NUM_ACTIONS = 2

train_reward_history = []
model = ContinuousPPOModel(num_inputs=NUM_INPUTS, num_actions=NUM_ACTIONS)
trainer = ContinuousPPOTrainer(
    model = model,
    buffer_size=BUFFER_SIZE,
    num_inputs=NUM_INPUTS,
    num_actions=NUM_ACTIONS
  )

import builtins

trainer.epsilon = 0.2
trainer.entropy_coeff = 0.0007
# trainer.value_function_coeff = 0.9
trainer.gamma = 0.99

trainer.set_means_lr(0.00025)
trainer.set_devs_lr(0.00025)
trainer.set_critic_lr(0.00025)

trainer.actor_losses = []
trainer.critic_losses = []

BATCH_SIZE = 1000
EPOCHS = 10
TRAIN_TIMES = 32*5
episode_rewards = []

for i in range(TRAIN_TIMES):
  done = True

  # Populate buffer
  with no_grad():
    for i in range(trainer.buffer_size):
      if done:
        state, _ = env.reset()

      actor, critic = trainer._predict(model, from_numpy(state))
      actions, actor_log_probs = trainer._sample_actions(actor)

      next_state, reward, terminated, truncated, _ = env.step(actions.numpy())
      
      means, devs = actor
      done = terminated or truncated

      trainer.buffer_append(state, next_state, means, devs, actions, critic, reward, done)
      episode_rewards.append(reward)

      state = next_state

    _, last_critic = trainer._predict(model, from_numpy(state), actor=False) # Inside no grad
    state, _ = env.reset()


  train_reward_history.append(builtins.sum(episode_rewards)/len(episode_rewards))

  trainer.train(batch_size=BATCH_SIZE, epochs=EPOCHS, last_critic=last_critic, normalize=True)