"""Both independent transitions and episode-aware sequences for future world models."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .env import Transition


class Replay:
    def __init__(self, capacity=100000, seed=0):
        self.capacity, self.data, self.position = capacity, [], 0
        self.rng = np.random.default_rng(seed)

    def add(self, transition):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, size):
        return [self.data[i] for i in self.rng.integers(len(self.data), size=size)]


class EpisodeStore:
    """Never joins two episodes into one sequence; masks padded positions.

    observations have length L+1, actions/rewards/flags length L.
    """
    def __init__(self, episodes=None, seed=0):
        self.episodes = episodes or []
        self.rng = np.random.default_rng(seed)

    def append(self, episode):
        if not episode:
            return
        for first, second in zip(episode, episode[1:]):
            if first.next_state != second.state or first.terminated or first.truncated:
                raise ValueError("Sequence crosses an episode boundary.")
        self.episodes.append(list(episode))

    def sample_sequences(self, batch_size, length):
        states = np.zeros((batch_size, length + 1), dtype=np.int64)
        actions = np.zeros((batch_size, length), dtype=np.int64)
        rewards = np.zeros((batch_size, length), dtype=np.float32)
        terminated = np.zeros((batch_size, length), dtype=bool)
        truncated = np.zeros_like(terminated)
        mask = np.zeros((batch_size, length), dtype=np.float32)
        is_first = np.zeros((batch_size, length + 1), dtype=bool)
        for b in range(batch_size):
            episode = self.episodes[int(self.rng.integers(len(self.episodes)))]
            start = int(self.rng.integers(len(episode)))
            segment = episode[start:start + length]
            states[b, 0] = segment[0].state
            is_first[b, 0] = start == 0
            for t, tr in enumerate(segment):
                states[b, t + 1] = tr.next_state
                actions[b, t], rewards[b, t] = tr.action, tr.reward
                terminated[b, t], truncated[b, t] = tr.terminated, tr.truncated
                mask[b, t] = 1
            states[b, len(segment) + 1:] = segment[-1].next_state
        return dict(states=states, actions=actions, rewards=rewards,
                    terminated=terminated, truncated=truncated, mask=mask, is_first=is_first)

    def save(self, path):
        Path(path).write_text(json.dumps([[asdict(tr) for tr in ep] for ep in self.episodes]), encoding="utf-8")

    @classmethod
    def load(cls, path, seed=0):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([[Transition(**tr) for tr in ep] for ep in data], seed)
