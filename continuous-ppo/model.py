from torch import nn

class ContinuousPPOAgent(nn.Module):
    def __init__(self, num_inputs=2, num_actions=1):
        super(ContinuousPPOAgent, self).__init__()

        # Architecture
        self.common1 = nn.Linear(num_inputs, 64)
        self.common2 = nn.Linear(64, 32)
        
        self.means = nn.Linear(32, num_actions)
        self.devs = nn.Linear(32, num_actions)

        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.functional.relu(self.common1(x))
        x = nn.functional.relu(self.common2(x))

        actor_means = nn.functional.tanh(self.means(x))
        actor_devs = nn.functional.leaky_relu(self.devs(x), negative_slope=0.2)

        critic = self.critic(x)

        return (actor_means, actor_devs), critic
