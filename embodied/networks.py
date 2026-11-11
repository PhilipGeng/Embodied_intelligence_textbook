import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, outputs))


class GaussianActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(6, 4)

    def distribution(self, obs):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        return Normal(mean, log_std.clamp(-5, 1).exp())

    def forward(self, obs, deterministic=False):
        dist = self.distribution(obs)
        z = dist.mean if deterministic else dist.rsample()
        action = z.tanh()
        logp = (dist.log_prob(z) - torch.log(1 - action.square() + 1e-6)).sum(-1)
        return action, logp

    def log_prob(self, obs, action):
        action = action.clamp(-1 + 1e-5, 1 - 1e-5)
        z = .5 * (torch.log1p(action) - torch.log1p(-action))
        return (self.distribution(obs).log_prob(z) - torch.log(1 - action.square() + 1e-6)).sum(-1)


class DeterministicActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(6, 2)

    def forward(self, obs):
        return self.net(obs).tanh()


class TwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.first, self.second = mlp(8, 1), mlp(8, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.first(x).squeeze(-1), self.second(x).squeeze(-1)


def action_function(actor, device, stochastic=False):
    def policy(obs):
        with torch.no_grad():
            tensor = torch.as_tensor(obs, device=device)
            if isinstance(actor, GaussianActor):
                action, _ = actor(tensor, deterministic=not stochastic)
            else:
                action = actor(tensor)
            return action.cpu().numpy()
    return policy


@torch.no_grad()
def soft_update(target, source, tau=.01):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.mul_(1 - tau).add_(sp, alpha=tau)
