# buffer.py

from dataclasses import dataclass
from torch import from_numpy, stack, empty

@dataclass
class ContinuousPPOBuffer():
   size: int
   num_inputs: int
   num_actions: int

   def __post_init__(self):
      self.initialize()

   def initialize(self):
      self._buffer_counter = 0

      self.states = empty(self.size, self.num_inputs)
      self.next_states = empty(self.size, self.num_inputs)
      self.means = empty(self.size, self.num_actions)
      self.devs = empty(self.size, self.num_actions)
      self.actions = empty(self.size, self.num_actions)
      self.critics = empty(self.size, 1)
      self.rewards = empty(self.size)
      self.dones = empty(self.size)
   
   def clear(self):
      self.initialize()
   
   def append(self, state, next_state, means, devs, actions, critic, reward, done):
      self.states[self._buffer_counter] = from_numpy(state)
      self.next_states[self._buffer_counter] = from_numpy(next_state)
      self.means[self._buffer_counter] = means
      self.devs[self._buffer_counter] = devs
      self.actions[self._buffer_counter] = actions
      self.critics[self._buffer_counter] = critic
      self.rewards[self._buffer_counter] = reward
      self.dones[self._buffer_counter] = done
      
      self._buffer_counter += 1
   
   def gae(self, last_value, normalize = False):
      values = self.critics
      done = self.dones
      rewards = self.rewards

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

      advantages = stacked_returns - values
      
      return stacked_returns, advantages
