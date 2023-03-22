from torch import nn

class ContinuousPPOAgent(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ContinuousPPOAgent, self).__init__()

        # Architecture
        self.common1 = nn.Linear(num_inputs, 32)
        self.common2 = nn.Linear(32, 16)
        
        self.means = nn.Linear(16, num_actions)
        self.devs = nn.Linear(16, num_actions)

        self.critic = nn.Linear(16, 1)

    def forward(self, x):
        x = nn.functional.relu(self.common1(x))
        x = nn.functional.relu(self.common2(x))

        actor_means = nn.functional.tanh(self.means(x))
        actor_devs = nn.functional.sigmoid(self.devs(x))

        critic = self.critic(x)

        return (actor_means, actor_devs), critic
