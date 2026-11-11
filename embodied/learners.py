from copy import deepcopy
import numpy as np
import torch
from torch import nn
from .env import DockingEnv, evaluate
from .networks import GaussianActor, DeterministicActor, TwinQ, action_function, soft_update
from .replay import GoalReplay
from warehouse.neural import setup


def train_online(algorithm, steps, seed=7, device="auto", sparse=False, randomize=False):
    device = setup(seed, device)
    rng = np.random.default_rng(seed)
    env = DockingEnv(seed=seed, sparse=sparse, randomize=randomize)
    is_td3 = algorithm == "td3"
    her = algorithm == "her"
    if her and not sparse:
        raise ValueError("HER requires the same sparse reward for original and relabeled goals.")
    actor = (DeterministicActor() if is_td3 else GaussianActor()).to(device)
    target_actor = deepcopy(actor) if is_td3 else None
    critic = TwinQ().to(device)
    target_critic = deepcopy(critic)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    log_alpha = torch.tensor(-1., device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)
    replay = GoalReplay(seed)
    policy = action_function(actor, device)
    obs, logs, updates = env.reset(), [], 0
    last_q_loss = last_actor_loss = 0.0
    log_every = max(500, steps // 10)
    for step in range(1, steps + 1):
        if step < 500:
            action = rng.uniform(-1, 1, 2)
        elif is_td3:
            action = np.clip(policy(obs) + rng.normal(0, .15, 2), -1, 1)
        else:
            action = action_function(actor, device, stochastic=True)(obs)
        tr, info = env.step(action)
        replay.add(tr)
        obs = env.reset() if tr.truncated or tr.terminated else tr.next_observation
        if step >= 500 and step % 2 == 0:
            s, a, r, ns, done = [torch.as_tensor(x, device=device) for x in replay.sample(128, her=her)]
            with torch.no_grad():
                if is_td3:
                    noise = (torch.randn_like(a) * .2).clamp(-.5, .5)
                    na = (target_actor(ns) + noise).clamp(-1, 1)
                    nq1, nq2 = target_critic(ns, na)
                    future = torch.minimum(nq1, nq2)
                else:
                    na, nlogp = actor(ns)
                    nq1, nq2 = target_critic(ns, na)
                    future = torch.minimum(nq1, nq2) - log_alpha.exp().detach() * nlogp
                target = r + .98 * (1 - done) * future
            q1, q2 = critic(s, a)
            q_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
            critic_opt.zero_grad(); q_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 10)
            critic_opt.step()
            last_q_loss = float(q_loss.detach())
            updates += 1
            if not is_td3 or updates % 2 == 0:
                # Freeze parameters, but retain dQ/da for the Actor.
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
                if is_td3:
                    qa, _ = critic(s, actor(s))
                    actor_loss = -qa.mean()
                else:
                    sampled, logp = actor(s)
                    qa, qb = critic(s, sampled)
                    actor_loss = (log_alpha.exp().detach() * logp - torch.minimum(qa, qb)).mean()
                actor_opt.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 5)
                actor_opt.step()
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)
                last_actor_loss = float(actor_loss.detach())
                if not is_td3:
                    # Continuous differential-entropy target = -action_dimension.
                    alpha_loss = -(log_alpha * (logp.detach() - 2.0)).mean()
                    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
                    with torch.no_grad():
                        log_alpha.clamp_(-8, 2)
                soft_update(target_critic, critic)
                if is_td3:
                    soft_update(target_actor, actor)
        if step % log_every == 0 or step == steps:
            score, _ = evaluate(policy, episodes=15, sparse=sparse)
            logs.append(dict(steps=step, critic_loss=last_q_loss, actor_loss=last_actor_loss,
                             temperature=float(log_alpha.exp().detach()) if not is_td3 else 0., **score))
            print(f"{algorithm}: {step}/{steps}, success={score['success_rate']:.0%}, distance={score['final_distance']:.3f}", flush=True)
    return actor, critic, logs
