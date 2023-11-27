from dataclasses import dataclass
from torch import Tensor, optim, from_numpy, clamp, minimum, log, tensor, sum, stack, mean, no_grad, mean, empty, float64, randperm, cat
from torch.distributions import Normal
from torch.nn.functional import mse_loss
import sys
from numpy import ndarray

from continuous_ppo.model import ContinuousPPOModel

EPSILON = sys.float_info.epsilon

@dataclass
class ContinuousPPOTrainer():
    model: ContinuousPPOModel
    num_inputs: int
    num_actions: int
    buffer_size: int

    gamma: float = 0.999
    lamda: float = 0.95
    epsilon: float = 0.3
    entropy_coeff: float = 0.0007
    value_function_coeff: float = 1

    means_lr: float = 0.0003
    devs_lr: float = 0.0001
    critic_lr: float = 0.0007

    def _initialize_buffers(self):
      self._buffer_counter = 0

      self._states = empty(self.buffer_size, self.num_inputs)
      self._next_states = empty(self.buffer_size, self.num_inputs)
      self._means = empty(self.buffer_size, self.num_actions)
      self._devs = empty(self.buffer_size, self.num_actions)
      self._actions = empty(self.buffer_size, self.num_actions)
      self._critics = empty(self.buffer_size, 1)
      self._rewards = empty(self.buffer_size)
      self._dones = empty(self.buffer_size)
    
    def __post_init__(self):
      self.num_inputs = self.model.num_inputs
      self.num_actions = self.model.num_actions

      self._EPSILON = sys.float_info.epsilon

      self._means_optimizer = optim.Adam(self.model.means.parameters(), lr=self.means_lr)
      self._devs_optimizer = optim.Adam(self.model.devs.parameters(), lr=self.devs_lr)
      self._critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=self.critic_lr)

      self.critic_losses = []
      self.actor_losses = []

      self._initialize_buffers()

    def set_means_lr(self, lr: float):
      for param_group in self._means_optimizer.param_groups:
        param_group['lr'] = lr

    def set_devs_lr(self, lr: float):
      for param_group in self._devs_optimizer.param_groups:
        param_group['lr'] = lr

    def set_critic_lr(self, lr: float):
      for param_group in self._critic_optimizer.param_groups:
        param_group['lr'] = lr

    def buffer_append(self, state, next_state, means, devs, actions, critic, reward, done):
      self._states[self._buffer_counter] = from_numpy(state)
      self._next_states[self._buffer_counter] = from_numpy(next_state)
      self._means[self._buffer_counter] = means
      self._devs[self._buffer_counter] = devs
      self._actions[self._buffer_counter] = actions
      self._critics[self._buffer_counter] = critic
      self._rewards[self._buffer_counter] = reward
      self._dones[self._buffer_counter] = done
      
      self._buffer_counter += 1
    
    def buffer_clear(self):
      self._initialize_buffers()
    
    def _sample_actions(self, actor: tuple[Tensor, Tensor]):
      means, devs = actor

      dist = Normal(means, devs)

      actions = dist.rsample()
      log_probs = dist.log_prob(actions)

      return actions, log_probs

    def _predict(self, model: ContinuousPPOModel, model_input: Tensor, critic=True, actor=True) -> tuple[tuple[Tensor, Tensor], Tensor]:
      return model(model_input, critic=critic, actor=actor)

    def _train_batch(self, states, returns, actors, actions, critics, rewards, advantages):
      means, devs = actors
      actor_log_probs = Normal(means, devs).log_prob(actions)

      new_actors, new_critics = self._predict(self.model, states)
      
      new_means, new_devs = new_actors
      new_actor_dists = Normal(new_means, new_devs)
      new_actor_actions = new_actor_dists.rsample()
      new_actor_log_probs = new_actor_dists.log_prob(new_actor_actions)
      entropies = new_actor_dists.entropy()

      value_function_loss = mse_loss(new_critics, returns)

      ratios = new_actor_log_probs/(actor_log_probs+EPSILON)
      clipped_ratio = clamp(ratios, 1-self.epsilon, 1+self.epsilon)

      l_clip = minimum(ratios*advantages, clipped_ratio*advantages)

      clip_loss = l_clip + self.entropy_coeff*entropies # - self.value_function_coeff*value_function_loss

      batch_means = -mean(clip_loss, 1) # dim = 1
      return mean(batch_means), value_function_loss
    
    def _calculate_gae(self, values: list[Tensor], done: list[bool], rewards: list[ndarray], last_value, normalize = False) -> ndarray:
      returns = []
      gae = 0
      
      for i in reversed(range(len(rewards))):
        try:
            value_next = values[i + 1]
        except IndexError:
            value_next = last_value

        value = values[i]

        mask = 0 if done[i] else 1

        delta = rewards[i] + 1*value_next*mask - value
        gae = delta + 1*1*gae*mask
            
        returns.insert(0, gae+value)

      stacked_returns = stack(returns)

      if normalize:
        stacked_returns = (stacked_returns - stacked_returns.mean()) / stacked_returns.std()
      
      return stacked_returns

    def train(self, batch_size, epochs, last_critic, normalize=False):
      returns = self._calculate_gae(self._critics, self._dones, self._rewards, last_critic, normalize=normalize)
      advantages = returns - self._critics

      for _ in range(epochs):
        
        indexes = randperm(self.buffer_size)

        shuffled_states = self._states[indexes]
        shuffled_means = self._means[indexes]
        shuffled_devs = self._devs[indexes]
        shuffled_actions = self._actions[indexes]
        shuffled_critics = self._critics[indexes]
        shuffled_rewards = self._rewards[indexes]

        shuffled_returns = returns[indexes]
        shuffled_advantages = advantages[indexes]

        for i in range(0, self.buffer_size, batch_size):

          batch_states = shuffled_states[i:i+batch_size]
          batch_means = shuffled_means[i:i+batch_size]
          batch_devs = shuffled_devs[i:i+batch_size]
          batch_actions = shuffled_actions[i:i+batch_size]
          batch_critics = shuffled_critics[i:i+batch_size]
          batch_rewards = shuffled_rewards[i:i+batch_size]
          batch_returns = shuffled_returns[i:i+batch_size]
          batch_advantages = shuffled_advantages[i:i+batch_size]

          batch_actor = (batch_means, batch_devs)

          self._means_optimizer.zero_grad()
          self._devs_optimizer.zero_grad()
          self._critic_optimizer.zero_grad()

          actor_loss, critic_loss = self._train_batch(
            batch_states,
            batch_returns,
            batch_actor,
            batch_actions,
            batch_critics,
            batch_rewards,
            batch_advantages
          )

          self.actor_losses.append(actor_loss)
          self.critic_losses.append(critic_loss)

          actor_loss.backward()
          critic_loss.backward()

          self._means_optimizer.step()
          self._devs_optimizer.step()
          self._critic_optimizer.step()     

      self._initialize_buffers()