# trainer.py

from dataclasses import dataclass
from torch import optim, clamp, minimum, mean, mean, randperm
from torch.distributions import Normal
from torch.nn.functional import mse_loss
import sys
from continuousppo.buffer import ContinuousPPOBuffer

from continuousppo.model import ContinuousPPOModel

EPSILON = sys.float_info.epsilon

@dataclass
class ContinuousPPOTrainer():
    model: ContinuousPPOModel
    num_inputs: int
    num_actions: int
    # buffer_size: int

    gamma: float = 0.999
    lamda: float = 0.95
    epsilon: float = 0.3
    entropy_coeff: float = 0.0007
    value_function_coeff: float = 1

    means_lr: float = 0.0003
    devs_lr: float = 0.0001
    critic_lr: float = 0.0007
    
    def __post_init__(self):
      self.num_inputs = self.model.num_inputs
      self.num_actions = self.model.num_actions

      self._EPSILON = sys.float_info.epsilon

      self._means_optimizer = optim.Adam(self.model.means.parameters(), lr=self.means_lr)
      self._devs_optimizer = optim.Adam(self.model.devs.parameters(), lr=self.devs_lr)
      self._critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=self.critic_lr)

      # History

      self.critic_losses = []
      self.actor_losses = []

      # self._initialize_buffers()

    def set_means_lr(self, lr: float):
      for param_group in self._means_optimizer.param_groups:
        param_group['lr'] = lr

    def set_devs_lr(self, lr: float):
      for param_group in self._devs_optimizer.param_groups:
        param_group['lr'] = lr

    def set_critic_lr(self, lr: float):
      for param_group in self._critic_optimizer.param_groups:
        param_group['lr'] = lr

    # actual PPO algo
    def _train_batch(self, states, returns, actors, actions, advantages):
      means, devs = actors
      actor_log_probs = Normal(means, devs).log_prob(actions)

      new_actors, new_critics = self.model.predict(states)
      
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
    
    def train(self, buffer: ContinuousPPOBuffer, batch_size, epochs, last_critic, normalize=False):
      returns, advantages = buffer.gae(last_value=last_critic, normalize=normalize)

      for _ in range(epochs):
        
        indexes = randperm(buffer.size)

        shuffled_states = buffer.states[indexes]
        shuffled_means = buffer.means[indexes]
        shuffled_devs = buffer.devs[indexes]
        shuffled_actions = buffer.actions[indexes]
        # shuffled_critics = buffer.critics[indexes]
        # shuffled_rewards = buffer.rewards[indexes]

        shuffled_returns = returns[indexes]
        shuffled_advantages = advantages[indexes]

        for i in range(0, buffer.size, batch_size):

          batch_states = shuffled_states[i:i+batch_size]
          batch_means = shuffled_means[i:i+batch_size]
          batch_devs = shuffled_devs[i:i+batch_size]
          batch_actions = shuffled_actions[i:i+batch_size]
          # batch_critics = shuffled_critics[i:i+batch_size]
          # batch_rewards = shuffled_rewards[i:i+batch_size]
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
            # batch_critics,
            # batch_rewards,
            batch_advantages
          )

          self.actor_losses.append(actor_loss)
          self.critic_losses.append(critic_loss)

          actor_loss.backward()
          critic_loss.backward()

          self._means_optimizer.step()
          self._devs_optimizer.step()
          self._critic_optimizer.step()