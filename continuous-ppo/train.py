import sys
import math
from numpy import ndarray

from torch import normal, clamp, log, minimum, from_numpy, no_grad
from torch.optim import Adam
from gymnasium import Env

from model import ContinuousPPOAgent

EPSILON = sys.float_info.epsilon

def sample_actions(actor: tuple[int, int]):
    mean, dev = actor
    return normal(mean=mean, std=max(0, dev))

def predict(model: ContinuousPPOAgent, input: ndarray):
    return model(from_numpy(input))

# FIXME: Grads can't be calculated correctly right now.

def train(
    model: ContinuousPPOAgent,
    env: Env,
    episodes: int,
    means_optimizer: Adam,
    devs_optimizer: Adam,
    critic_optimizer: Adam,
    gamma=0.999,
    epsilon=0.2,
    entropy_coeff=0.01
):
    old_model = ContinuousPPOAgent()
    old_model.load_state_dict(model.state_dict())

    for _ in range(episodes):
        observation, _ = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            actor, critic = predict(model, observation)
            actions = sample_actions(actor)

            old_actor, _ = predict(old_model, observation)
            old_actions = sample_actions(old_actor)
            
            with no_grad():
                observation, reward, terminated, truncated, _ = env.step(actions)

            end = terminated or truncated

            _, critic_next = predict(model, observation)
            _, old_critic_next = predict(model, observation)

            td_target = reward + gamma*(0 if end else critic_next)
            td_error = td_target - critic
            advantage = reward + gamma*(0 if end else old_critic_next)

            ratio = actions/(old_actions+EPSILON)
            clipped_ratio = clamp(ratio, 1-epsilon, 1+epsilon)

            critic_loss = td_error**2

            _, devs = actor
            l_entropy = sum(0.5*log((2*math.pi*math.e*devs**2)+EPSILON))
            l_clip = minimum(ratio*advantage, clipped_ratio*advantage)

            actor_loss = l_clip + entropy_coeff*l_entropy

            old_model.load_state_dict(model.state_dict())

            critic_loss.backward()
            actor_loss.backward()

            means_optimizer.step()
            devs_optimizer.step()
            critic_optimizer.step()