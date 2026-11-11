"""Small explicit PyTorch learners. No external RL framework hides the updates."""
import numpy as np
from functools import partial
import torch
from torch import nn
from torch.distributions import Categorical
from .env import Warehouse, evaluate
from .replay import Replay, EpisodeStore
from .device import resolve_device


def setup(seed, device="auto"):
    device = resolve_device(device)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return device


class Network(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(70, 64), nn.Tanh(), nn.Linear(64, output))

    def forward(self, states):
        x = torch.nn.functional.one_hot(states.long(), num_classes=70).float()
        return self.body(x)


def categorical_gae(rewards, values, next_values, terminated, boundaries, gamma, lam):
    """[time, env]. Bootstrap at timeouts, but never recurse into reset episodes."""
    delta = rewards + gamma * (1 - terminated) * next_values - values
    advantages = torch.zeros_like(delta)
    last = delta.new_zeros(delta.shape[1])
    for t in reversed(range(len(delta))):
        last = delta[t] + gamma * lam * (1 - boundaries[t]) * last
        advantages[t] = last
    return advantages


def dqn(total_steps, seed, shaping=True, device="auto"):
    device = setup(seed, device)
    tensor = partial(torch.as_tensor, device=device)
    rng = np.random.default_rng(seed)
    env = Warehouse(seed=seed, shaping=shaping)
    q, target_q = Network(4).to(device), Network(4).to(device)
    target_q.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)
    replay, store = Replay(seed=seed), EpisodeStore(seed=seed)
    state, episode, logs = env.reset(), [], []
    log_every = max(1000, total_steps // 10)
    for step in range(1, total_steps + 1):
        epsilon = max(0.05, 1 - step / max(1, total_steps * 0.55))
        with torch.no_grad():
            action = int(rng.integers(4)) if rng.random() < epsilon else int(q(tensor(state)).argmax())
        tr = env.step(action)
        replay.add(tr)
        episode.append(tr)
        state = tr.next_state
        if tr.terminated or tr.truncated:
            store.append(episode)
            episode = []
            state = env.reset()
        if step >= 512 and step % 4 == 0:
            batch = replay.sample(64)
            s = tensor([x.state for x in batch])
            a = tensor([x.action for x in batch])
            r = tensor([x.reward for x in batch])
            ns = tensor([x.next_state for x in batch])
            done = tensor([x.terminated for x in batch], dtype=torch.float32)
            with torch.no_grad():
                y = r + env.gamma * (1 - done) * target_q(ns).max(dim=1).values
            prediction = q(s).gather(1, a[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10)
            optimizer.step()
        if step % 500 == 0:
            target_q.load_state_dict(q.state_dict())
        if step % log_every == 0 or step == total_steps:
            policy = lambda s: int(q(tensor(s)).argmax())
            with torch.no_grad():
                logs.append(dict(steps=step, **evaluate(policy, episodes=50, shaping=shaping)))
    if episode:
        store.append(episode)
    return q, logs, store


def reinforce(episodes, seed, shaping=True, device="auto"):
    device = setup(seed, device)
    tensor = partial(torch.as_tensor, device=device)
    env = Warehouse(seed=seed, shaping=shaping, max_steps=10000)
    actor = Network(4).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=2e-3)
    logs, store, pending = [], EpisodeStore(seed=seed), []
    steps, baseline = 0, 0.0
    for ep in range(episodes):
        state, trajectory = env.reset(), []
        while True:
            with torch.no_grad():
                action = int(Categorical(logits=actor(tensor(state))).sample())
            tr = env.step(action)
            trajectory.append(tr)
            state = tr.next_state
            steps += 1
            if tr.terminated:
                break
            if tr.truncated:
                raise RuntimeError("REINFORCE needs a complete return; increase its collection guard.")
        store.append(trajectory)
        returns, g = [], 0.0
        for tr in reversed(trajectory):
            g = tr.reward + env.gamma * g
            returns.append(g)
        returns.reverse()
        pending.append((trajectory, returns))
        if len(pending) == 8 or ep == episodes - 1:
            # Constant baseline from PREVIOUS batches: no critic, no data-dependent
            # action baseline. Divide by episodes, not by variable trajectory length.
            loss = tensor(0.0)
            all_returns = []
            for trajectory, returns in pending:
                states = tensor([x.state for x in trajectory])
                actions = tensor([x.action for x in trajectory])
                advantages = (tensor(returns) - baseline) / 10.0
                discount = env.gamma ** torch.arange(len(returns), device=device)
                logp = Categorical(logits=actor(states)).log_prob(actions)
                loss = loss - (logp * advantages * discount).sum()
                all_returns.extend(returns)
            loss = loss / len(pending)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 5)
            optimizer.step()
            baseline = 0.9 * baseline + 0.1 * float(np.mean(all_returns))
            pending = []
        if (ep + 1) % max(8, episodes // 10) == 0:
            with torch.no_grad():
                logs.append(dict(steps=steps, episode=ep + 1,
                                 **evaluate(lambda s: int(actor(tensor(s)).argmax()), episodes=50,
                                            shaping=shaping)))
    return actor, logs, store


def sac(total_steps, seed, shaping=True, device="auto"):
    """Discrete SAC: exact action sums, twin Q, replay and learned temperature.

    Continuous SAC needs a different actor distribution/reparameterization;
    a grid action is categorical, so no tanh-Gaussian is used here.
    """
    device = setup(seed, device)
    tensor = partial(torch.as_tensor, device=device)
    rng = np.random.default_rng(seed)
    env = Warehouse(seed=seed, shaping=shaping)
    actor, q1, q2 = Network(4).to(device), Network(4).to(device), Network(4).to(device)
    target1, target2 = Network(4).to(device), Network(4).to(device)
    target1.load_state_dict(q1.state_dict()); target2.load_state_dict(q2.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
    log_alpha = tensor(0.0).requires_grad_()
    alpha_opt = torch.optim.Adam([log_alpha], lr=1e-3)
    target_entropy = 0.5 * float(np.log(4))
    replay, store = Replay(seed=seed), EpisodeStore(seed=seed)
    state, episode, logs = env.reset(), [], []
    log_every = max(1000, total_steps // 10)
    for step in range(1, total_steps + 1):
        with torch.no_grad():
            action = (int(rng.integers(4)) if step < 512 else
                      int(Categorical(logits=actor(tensor(state))).sample()))
        tr = env.step(action)
        replay.add(tr); episode.append(tr)
        state = tr.next_state
        if tr.terminated or tr.truncated:
            store.append(episode); episode = []
            state = env.reset()
        if step >= 512 and step % 4 == 0:
            batch = replay.sample(64)
            s = tensor([x.state for x in batch])
            a = tensor([x.action for x in batch])
            r = tensor([x.reward for x in batch])
            ns = tensor([x.next_state for x in batch])
            done = tensor([x.terminated for x in batch], dtype=torch.float32)
            alpha = log_alpha.exp().detach()
            with torch.no_grad():
                next_logp = actor(ns).log_softmax(-1)
                next_p = next_logp.exp()
                next_q = torch.minimum(target1(ns), target2(ns))
                next_value = (next_p * (next_q - alpha * next_logp)).sum(-1)
                y = r + env.gamma * (1 - done) * next_value
            pred1 = q1(s).gather(1, a[:, None]).squeeze(-1)
            pred2 = q2(s).gather(1, a[:, None]).squeeze(-1)
            q_loss = nn.functional.mse_loss(pred1, y) + nn.functional.mse_loss(pred2, y)
            q_opt.zero_grad(); q_loss.backward()
            nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10)
            q_opt.step()

            logp = actor(s).log_softmax(-1)
            p = logp.exp()
            with torch.no_grad():
                minimum_q = torch.minimum(q1(s), q2(s))
            # Sum exactly over the four discrete actions; do not differentiate Q params.
            actor_loss = (p * (alpha * logp - minimum_q)).sum(-1).mean()
            actor_opt.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 5)
            actor_opt.step()

            entropy = -(p.detach() * logp.detach()).sum(-1).mean()
            # Entropy above target -> reduce alpha; below target -> increase alpha.
            alpha_loss = log_alpha * (entropy - target_entropy)
            alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
            with torch.no_grad():
                log_alpha.clamp_(-8, 3)
                for target, source in ((target1, q1), (target2, q2)):
                    for tp, sp in zip(target.parameters(), source.parameters()):
                        tp.mul_(0.99).add_(sp, alpha=0.01)
        if step % log_every == 0 or step == total_steps:
            with torch.no_grad():
                score = evaluate(lambda s: int(actor(tensor(s)).argmax()),
                                 episodes=50, shaping=shaping)
            logs.append(dict(steps=step, temperature=float(log_alpha.exp().detach()),
                             target_entropy=target_entropy, **score))
    if episode:
        store.append(episode)
    return actor, q1, q2, logs, store


def actor_critic(name, total_steps, seed, shaping=True, device="auto"):
    device = setup(seed, device)
    tensor = partial(torch.as_tensor, device=device)
    actor, critic = Network(4).to(device), Network(1).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=2e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=2e-3)
    n_envs, rollout_length = 16, 32
    envs = [Warehouse(seed=seed + i * 100, shaping=shaping) for i in range(n_envs)]
    gamma = envs[0].gamma
    states = np.array([env.reset() for env in envs])
    episodes = [[] for _ in envs]
    store, logs, steps = EpisodeStore(seed=seed), [], 0
    last_log = 0
    while steps < total_steps:
        ss, aa, rr, vv, nv, tt, bb, lp = [], [], [], [], [], [], [], []
        for _ in range(rollout_length):
            with torch.no_grad():
                st = tensor(states)
                dist = Categorical(logits=actor(st))
                actions = dist.sample()
                values = critic(st).squeeze(-1)
                logp = dist.log_prob(actions)
            transitions = [env.step(a) for env, a in zip(envs, actions.cpu().tolist())]
            next_states = tensor([tr.next_state for tr in transitions])
            with torch.no_grad():
                next_values = critic(next_states).squeeze(-1)
            ss.append(st); aa.append(actions); vv.append(values); nv.append(next_values); lp.append(logp)
            rr.append(tensor([tr.reward for tr in transitions]))
            tt.append(tensor([tr.terminated for tr in transitions], dtype=torch.float32))
            bb.append(tensor([tr.terminated or tr.truncated for tr in transitions], dtype=torch.float32))
            for i, tr in enumerate(transitions):
                episodes[i].append(tr)
                states[i] = tr.next_state
                if tr.terminated or tr.truncated:
                    store.append(episodes[i]); episodes[i] = []
                    states[i] = envs[i].reset()
            steps += n_envs
        s, a, rewards, values, next_values, terms, boundaries, old_logp = map(
            torch.stack, (ss, aa, rr, vv, nv, tt, bb, lp))
        # AC uses one-step TD (lambda=0); GAE and PPO use lambda=.95.
        advantages = categorical_gae(rewards, values, next_values, terms, boundaries,
                                     gamma, 0.0 if name == "ac" else 0.95)
        returns = (advantages + values).flatten().detach()
        s, a, old_logp = s.flatten(), a.flatten(), old_logp.flatten()
        advantages = advantages.flatten().detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        epochs = 4 if name == "ppo" else 1
        for _ in range(epochs):
            for idx in torch.randperm(len(s), device=device).split(128):
                dist = Categorical(logits=actor(s[idx]))
                new_logp = dist.log_prob(a[idx])
                if name == "ppo":
                    ratio = (new_logp - old_logp[idx]).exp()
                    objective = torch.minimum(ratio * advantages[idx],
                                              ratio.clamp(0.8, 1.2) * advantages[idx])
                    actor_loss = -objective.mean()
                else:
                    actor_loss = -(new_logp * advantages[idx]).mean()
                actor_loss = actor_loss - 0.01 * dist.entropy().mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_opt.step()
                value_loss = 0.5 * (critic(s[idx]).squeeze(-1) - returns[idx]).square().mean()
                critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()
        if steps - last_log >= max(512, total_steps // 10) or steps >= total_steps:
            with torch.no_grad():
                logs.append(dict(steps=steps, **evaluate(lambda st: int(actor(tensor(st)).argmax()),
                                                        episodes=50, shaping=shaping)))
            last_log = steps
    for episode in episodes:
        if episode:
            store.append(episode)
    return actor, critic, logs, store
