from dataclasses import dataclass
import numpy as np


@dataclass
class ReachTransition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


class DockingEnv:
    """2D point-mass docking subskill: continuous forces, inertia and friction.

    This is a pedagogical dynamics simulator, not articulated-robot physics.
    A fixed 60-step collection window does not terminate the underlying task.
    Success is measured at the final step, encouraging stable goal holding.
    """
    observation_size, action_size, horizon, dt = 6, 2, 60, 0.12
    success_radius = .12

    def __init__(self, seed=7, sparse=False, randomize=False, shifted=False, random_goals=False):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 982451653]))
        self.sparse, self.randomize, self.shifted = sparse, randomize, shifted
        self.random_goals = random_goals
        self.reset()

    def reset(self):
        self.mass, self.drag, self.delay = 1.0, .35, 0
        if self.randomize:
            self.mass = float(self.rng.uniform(.65, 1.45))
            self.drag = float(self.rng.uniform(.15, .65))
            self.delay = int(self.rng.integers(2))
        if self.shifted:
            self.mass, self.drag, self.delay = 1.6, .7, 2
        self.physical = np.r_[np.array([-.7, -.7]) + self.rng.uniform(-.04, .04, 2), np.zeros(2)]
        self.goal = self.rng.uniform(-.8, .8, 2) if self.random_goals else np.array([.7, .7])
        self.queue = [np.zeros(2) for _ in range(self.delay)]
        self.steps, self.ended = 0, False
        return self.observation()

    def observation(self):
        return np.r_[self.physical, self.goal].astype(np.float32)

    @classmethod
    def compute_reward(cls, achieved, desired, action=None, sparse=True):
        distance = np.linalg.norm(np.asarray(achieved) - np.asarray(desired), axis=-1)
        if sparse:
            return np.where(distance < cls.success_radius, 0.0, -1.0).astype(np.float32)
        penalty = 0 if action is None else .01 * np.square(action).sum(axis=-1)
        return np.asarray(-distance - penalty, dtype=np.float32)

    def step(self, action):
        if self.ended:
            raise RuntimeError("Reset after the collection horizon.")
        previous = self.observation()
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        if action.shape != (2,):
            raise ValueError("Expected two continuous forces.")
        self.queue.append(action)
        applied = self.queue.pop(0)
        position, velocity = self.physical[:2].copy(), self.physical[2:].copy()
        velocity = np.clip(velocity + self.dt * (applied / self.mass - self.drag * velocity), -1.5, 1.5)
        proposed = position + self.dt * velocity
        collision = bool(np.any(np.abs(proposed) > 1))
        position = np.clip(proposed, -1, 1)
        velocity[np.abs(proposed) > 1] = 0
        self.physical = np.r_[position, velocity]
        self.steps += 1
        truncated = self.steps >= self.horizon
        self.ended = truncated
        reward = float(self.compute_reward(position, self.goal, action, self.sparse))
        tr = ReachTransition(previous, action.astype(np.float32), reward, self.observation(), False, truncated)
        info = dict(distance=float(np.linalg.norm(position - self.goal)),
                    success=bool(np.linalg.norm(position - self.goal) < self.success_radius),
                    collision=collision, mass=self.mass, drag=self.drag, delay=self.delay)
        return tr, info

    def render(self):
        image = np.full((256, 256, 3), 245, dtype=np.uint8)
        yy, xx = np.ogrid[:256, :256]
        for point, radius, color in ((self.goal, 14, (44, 170, 110)),
                                     (self.physical[:2], 8, (45, 100, 220))):
            x, y = (point + 1) * 127.5
            image[(xx - x) ** 2 + (yy - (255 - y)) ** 2 < radius ** 2] = color
        return image


def pd_action(observation):
    position, velocity, goal = observation[:2], observation[2:4], observation[4:6]
    return np.clip(3.0 * (goal - position) - 2.0 * velocity, -1, 1).astype(np.float32)


def evaluate(policy, seed=20000, episodes=30, sparse=False, shifted=False):
    env = DockingEnv(seed=seed, sparse=sparse, shifted=shifted)
    records, route = [], []
    for ep in range(episodes):
        obs, total, energy, collisions, reached = env.reset(), 0., 0., 0, False
        for t in range(env.horizon):
            action = policy(obs)
            tr, info = env.step(action)
            total += tr.reward
            energy += float(np.square(action).sum())
            collisions += info["collision"]
            reached |= info["success"]
            if ep == 0:
                route.append(dict(step=t, position=obs[:2].tolist(), velocity=obs[2:4].tolist(),
                                  goal=obs[4:6].tolist(), action=np.asarray(action).tolist(),
                                  next_position=tr.next_observation[:2].tolist(), reward=tr.reward,
                                  distance=info["distance"], success=info["success"]))
            obs = tr.next_observation
        records.append([info["success"], reached, info["distance"], total, energy, collisions])
    values = np.mean(records, axis=0)
    return dict(zip(("success_rate", "ever_reached_rate", "final_distance", "episode_return",
                     "action_energy", "boundary_collisions"), map(float, values))), route
