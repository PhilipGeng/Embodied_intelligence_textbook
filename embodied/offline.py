import numpy as np
import torch
from torch import nn
from .env import DockingEnv, pd_action, evaluate
from .networks import GaussianActor, TwinQ, mlp, action_function, soft_update
from warehouse.neural import setup


def collect_dataset(episodes=100, seed=7, include_task_goal=True):
    env = DockingEnv(seed=seed, random_goals=True)
    rng = np.random.default_rng(seed)
    data = []
    for episode in range(episodes):
        obs = env.reset()
        if include_task_goal and episode % 4 == 0:
            # Cover the actual deployment task as well as random goals.
            env.goal = np.array([.7, .7])
            obs = env.observation()
        for _ in range(env.horizon):
            # Programmatic PD demonstrations plus noisy/low-quality behavior.
            if episode % 5 == 0:
                action = rng.uniform(-1, 1, 2)
            else:
                action = np.clip(pd_action(obs) + rng.normal(0, .1, 2), -1, 1)
            tr, info = env.step(action)
            data.append((tr.observation, tr.action, tr.reward, tr.next_observation,
                         tr.terminated, episode))
            obs = tr.next_observation
    return {key: np.asarray([row[i] for row in data], dtype=np.float32)
            for i, key in enumerate(("observations", "actions", "rewards", "next_observations", "terminated", "episode_id"))}


def train_offline(algorithm, dataset, updates=2000, seed=7, device="auto"):
    device = setup(seed, device)
    rng = np.random.default_rng(seed)
    actor = GaussianActor().to(device)
    critic = TwinQ().to(device)
    target_q = TwinQ().to(device)
    target_q.load_state_dict(critic.state_dict())
    value = mlp(6, 1).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=5e-4)
    q_opt = torch.optim.Adam(critic.parameters(), lr=5e-4)
    value_opt = torch.optim.Adam(value.parameters(), lr=5e-4)
    logs = []
    for update in range(1, updates + 1):
        indices = rng.integers(len(dataset["rewards"]), size=128)
        s, a, r, ns, done = [torch.as_tensor(dataset[k][indices], device=device) for k in
                             ("observations", "actions", "rewards", "next_observations", "terminated")]
        if algorithm == "iql":
            with torch.no_grad():
                q1, q2 = target_q(s, a)
                q = torch.minimum(q1, q2)
            residual = q - value(s).squeeze(-1)
            weights = torch.where(residual > 0, .7, .3)
            v_loss = (weights * residual.square()).mean()
            value_opt.zero_grad(); v_loss.backward(); value_opt.step()
            with torch.no_grad():
                target = r + .98 * (1 - done) * value(ns).squeeze(-1)
            q1, q2 = critic(s, a)
            q_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
            q_opt.zero_grad(); q_loss.backward(); q_opt.step()
            with torch.no_grad():
                advantage = q - value(s).squeeze(-1)
                behavior_weights = (3 * advantage).exp().clamp(max=100)
            actor_loss = -(behavior_weights * actor.log_prob(s, a)).mean()
            soft_update(target_q, critic)
        else:
            predicted, _ = actor(s, deterministic=True)
            actor_loss = (predicted - a).square().mean()
        actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 10)
        actor_opt.step()
        if update % max(100, updates // 10) == 0 or update == updates:
            score, _ = evaluate(action_function(actor, device), episodes=15)
            logs.append(dict(updates=update, actor_loss=float(actor_loss.detach()), **score))
    return actor, critic, value, logs
