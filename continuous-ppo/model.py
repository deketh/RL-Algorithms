from torch import nn, Tensor, exp

class ContinuousPPOModel(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ContinuousPPOModel, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions

        # Architecture
        self.actor1 = nn.Linear(num_inputs, 128)
        self.critic1 = nn.Linear(num_inputs, 64)

        self.actor2 = nn.Linear(128, 64)
        self.critic2 = nn.Linear(64, 32)
        
        self.means = nn.Linear(64, num_actions)
        self.devs = nn.Linear(64, num_actions)

        self.critic = nn.Linear(32, 1)

    def forward(self, x, critic=True, actor=True) -> tuple[tuple[Tensor, Tensor], Tensor]:
        critic_tensor = None
        actor_tuple = None

        # x = nn.functional.leaky_relu(self.common1(x))
        # x = nn.functional.leaky_relu(self.common2(x))

        if actor:
          xactor = nn.functional.leaky_relu(self.actor1(x))
          xactor = nn.functional.leaky_relu(self.actor2(xactor))

          actor_means = nn.functional.tanh(self.means(xactor))
          actor_devs = exp(self.devs(xactor))

          actor_tuple = (actor_means, actor_devs)

        if critic:
          xcritic = nn.functional.leaky_relu(self.critic1(x))
          xcritic = nn.functional.leaky_relu(self.critic2(xcritic))

          critic_tensor = self.critic(xcritic)

        return actor_tuple, critic_tensor