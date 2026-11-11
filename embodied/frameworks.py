"""Gymnasium adapters. External algorithms share the same teaching environments."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from warehouse.env import Warehouse
from .env import DockingEnv


class WarehouseGym(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.env = Warehouse()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 1, shape=(70,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env = Warehouse(seed=seed)
        self.env.reset()
        return self.env.observation(), {}

    def step(self, action):
        tr = self.env.step(int(action))
        r, c, phase = self.env.decode(tr.next_state)
        return self.env.observation(), tr.reward, tr.terminated, tr.truncated, {"is_success": (r, c, phase) == (0, 6, 1)}

    def render(self):
        return self.env.observation(mode="rgb")


class DockingGym(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}
    def __init__(self, goal_dict=False, randomize=False, partial=False, render_mode=None):
        self.goal_dict, self.randomize, self.partial, self.render_mode = goal_dict, randomize, partial, render_mode
        self.env = DockingEnv(sparse=goal_dict, randomize=randomize)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        if goal_dict:
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(-1.5, 1.5, shape=(4,), dtype=np.float32),
                "achieved_goal": spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
                "desired_goal": spaces.Box(-1, 1, shape=(2,), dtype=np.float32)})
        else:
            self.observation_space = spaces.Box(-1.5, 1.5, shape=(6,), dtype=np.float32)

    def _observation(self):
        obs = self.env.observation()
        if self.partial:
            obs[2:4] = 0  # hide velocity; creates a real memory requirement
        if self.goal_dict:
            return {"observation": obs[:4], "achieved_goal": obs[:2].copy(), "desired_goal": obs[4:].copy()}
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env = DockingEnv(seed=seed, sparse=self.goal_dict, randomize=self.randomize)
        self.env.reset()
        return self._observation(), {}

    def step(self, action):
        tr, info = self.env.step(action)
        info["is_success"] = info["success"]
        return self._observation(), tr.reward, tr.terminated, tr.truncated, info

    def render(self):
        return self.env.render()


class GoalDockingGym(DockingGym):
    def __init__(self, **kwargs):
        super().__init__(goal_dict=True, **kwargs)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return DockingEnv.compute_reward(achieved_goal, desired_goal, sparse=True)


def make_env(name, partial=False, randomize=False):
    if name == "warehouse":
        return WarehouseGym()
    if name in ("reach", "reach-sparse"):
        cls = GoalDockingGym if name == "reach-sparse" else DockingGym
        return cls(partial=partial, randomize=randomize)
    # Standard Gymnasium/MuJoCo task IDs; optional dependencies are explicit.
    return gym.make(name)
