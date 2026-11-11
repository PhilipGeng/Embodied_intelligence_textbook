import numpy as np
from .env import DockingEnv


class GoalReplay:
    """Keep episode boundaries so future HER goals never cross resets."""
    def __init__(self, seed=7, capacity=100000):
        self.rng = np.random.default_rng(seed)
        self.episodes, self.current = [], []
        self.capacity = capacity

    def add(self, transition):
        self.current.append(transition)
        if transition.terminated or transition.truncated:
            self.episodes.append(self.current)
            self.current = []
            while sum(len(ep) for ep in self.episodes) > self.capacity:
                self.episodes.pop(0)

    def sample(self, size, her=False):
        available = self.episodes + ([self.current] if self.current else [])
        observations, actions, rewards, next_observations, dones = [], [], [], [], []
        lengths = np.array([len(ep) for ep in available])
        ends = lengths.cumsum()
        flat_indices = self.rng.integers(int(ends[-1]), size=size)
        episode_indices = np.searchsorted(ends, flat_indices, side="right")
        starts = np.r_[0, ends[:-1]]
        for ep_index, index in zip(episode_indices, flat_indices):
            episode = available[int(ep_index)]
            t = int(index - starts[ep_index])
            tr = episode[t]
            obs, ns, reward = tr.observation.copy(), tr.next_observation.copy(), tr.reward
            # Future can include the achieved position immediately after this action.
            if her and self.rng.random() < .8:
                future = episode[int(self.rng.integers(t, len(episode)))]
                new_goal = future.next_observation[:2]
                obs[4:6] = ns[4:6] = new_goal
                reward = float(DockingEnv.compute_reward(ns[:2], new_goal, sparse=True))
            observations.append(obs); actions.append(tr.action); rewards.append(reward)
            next_observations.append(ns); dones.append(tr.terminated)
        return [np.asarray(x, dtype=np.float32) for x in (observations, actions, rewards, next_observations, dones)]
