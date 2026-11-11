from dataclasses import dataclass
import numpy as np

MAP = ("S..#..D", ".#...#.", "...X...", ".#...#.", "P......")
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("up", "right", "down", "left")
START, PICKUP, DELIVERY, HAZARD = (0, 0), (4, 0), (0, 6), (2, 3)


@dataclass
class Transition:
    state: int
    action: int
    reward: float
    base_reward: float
    next_state: int
    terminated: bool
    truncated: bool


class Warehouse:
    """MDP: (row, column, carrying). Horizon truncates collection, not the MDP.

    slip chooses a perpendicular action. Model-based methods may call outcomes();
    model-free learners must use step() samples only.
    """
    n_states, n_actions = 70, 4

    def __init__(self, seed=0, slip=0.10, gamma=0.97, shaping=True, max_steps=120):
        # Separate the environment stream from a policy using default_rng(seed).
        # Reusing identical streams correlates exploratory actions with slip!
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 41031]))
        self.slip, self.gamma, self.shaping, self.max_steps = slip, gamma, shaping, max_steps
        self.valid_states = [self.encode(r, c, p) for p in (0, 1)
                             for r, row in enumerate(MAP) for c, cell in enumerate(row)
                             if cell not in "#X" and not (p == 1 and (r, c) == DELIVERY)]
        self.reset()

    @staticmethod
    def encode(r, c, carrying):
        return int(carrying * 35 + r * 7 + c)

    @staticmethod
    def decode(state):
        return state % 35 // 7, state % 7, state // 35

    def reset(self):
        self.state, self.elapsed = self.encode(*START, 0), 0
        self.ended = False
        return self.state

    def potential(self, state):
        r, c, phase = self.decode(state)
        target = DELIVERY if phase else PICKUP
        distance = abs(r - target[0]) + abs(c - target[1])
        if not phase:
            distance += abs(PICKUP[0] - DELIVERY[0]) + abs(PICKUP[1] - DELIVERY[1])
        return -0.5 * distance

    def deterministic(self, state, actual_action):
        r, c, phase = self.decode(state)
        dr, dc = ACTIONS[actual_action]
        nr, nc = r + dr, c + dc
        base = -0.2
        if not (0 <= nr < 5 and 0 <= nc < 7) or MAP[nr][nc] == "#":
            nr, nc = r, c
            base -= 0.6
        terminated = (nr, nc) == HAZARD
        if terminated:
            base -= 8.0
        if (nr, nc) == PICKUP and phase == 0:
            phase = 1
            base += 3.0
        if (nr, nc) == DELIVERY and phase == 1:
            terminated = True
            base += 20.0
        ns = self.encode(nr, nc, phase)
        reward = base
        if self.shaping:
            # Potential is zero only on genuine terminal outcomes, not truncations.
            reward += self.gamma * (0.0 if terminated else self.potential(ns)) - self.potential(state)
        return ns, reward, base, terminated

    def outcomes(self, state, action):
        """Exact distribution. For planning/reference checks only."""
        choices = ((1 - self.slip, action), (self.slip / 2, (action - 1) % 4),
                   (self.slip / 2, (action + 1) % 4))
        return [(prob, *self.deterministic(state, a)) for prob, a in choices if prob > 0]

    def step(self, action):
        if self.ended:
            raise RuntimeError("Reset after termination or truncation.")
        choices = self.outcomes(self.state, action)
        k = self.rng.choice(len(choices), p=[x[0] for x in choices])
        _, ns, reward, base, terminated = choices[k]
        self.elapsed += 1
        truncated = self.elapsed >= self.max_steps and not terminated
        tr = Transition(self.state, action, reward, base, ns, terminated, truncated)
        self.state, self.ended = ns, terminated or truncated
        return tr

    def observation(self, state=None, mode="onehot"):
        state = self.state if state is None else state
        r, c, phase = self.decode(state)
        if mode == "onehot":
            x = np.zeros(self.n_states, dtype=np.float32)
            x[state] = 1
            return x
        if mode == "features":
            target = DELIVERY if phase else PICKUP
            return np.array([r / 4, c / 6, phase, (target[0] - r) / 4,
                             (target[1] - c) / 6, (DELIVERY[0] - r) / 4,
                             (DELIVERY[1] - c) / 6], dtype=np.float32)
        if mode == "rgb":
            image = np.full((5, 7, 3), 235, dtype=np.uint8)
            for rr, row in enumerate(MAP):
                for cc, cell in enumerate(row):
                    image[rr, cc] = {"#": (45, 45, 45), "X": (220, 65, 60),
                                     "P": (230, 180, 35), "D": (65, 180, 100)}.get(cell, (235, 235, 235))
            image[r, c] = (150, 70, 210) if phase else (40, 110, 220)
            return np.repeat(np.repeat(image, 8, axis=0), 8, axis=1)
        raise ValueError(mode)


def evaluate(policy, seed=10000, episodes=100, shaping=True, gamma=0.97):
    env = Warehouse(seed=seed, shaping=shaping, gamma=gamma)
    records = []
    for _ in range(episodes):
        state = env.reset()
        base_return = shaped_return = 0.0
        for t in range(env.max_steps):
            tr = env.step(int(policy(state)))
            base_return += gamma ** t * tr.base_reward
            shaped_return += gamma ** t * tr.reward
            state = tr.next_state
            if tr.terminated or tr.truncated:
                break
        r, c, phase = env.decode(state)
        records.append((float((r, c) == DELIVERY and phase == 1),
                        float((r, c) == HAZARD), t + 1, base_return, shaped_return))
    values = np.mean(records, axis=0)
    return dict(zip(("success_rate", "hazard_rate", "mean_steps", "base_discounted_return",
                     "training_discounted_return"), map(float, values)))
