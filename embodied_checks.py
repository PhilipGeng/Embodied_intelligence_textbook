"""Physics, goal relabeling, continuous probabilities and framework semantics."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
import unittest
import numpy as np
import torch
from embodied.env import DockingEnv, ReachTransition, pd_action, evaluate
from embodied.networks import GaussianActor
from embodied.replay import GoalReplay
from embodied.offline import collect_dataset
from embodied.planning import LinearWorldModel


class EmbodiedChecks(unittest.TestCase):
    def test_force_mass_and_semi_implicit_integration(self):
        env = DockingEnv(seed=7)
        old = env.reset()
        tr, _ = env.step([1., 0.])
        self.assertAlmostEqual(float(tr.next_observation[2]), env.dt, places=6)
        self.assertAlmostEqual(float(tr.next_observation[0] - old[0]), env.dt ** 2, places=6)
        self.assertFalse(tr.terminated)

    def test_delay_is_actually_in_the_dynamics(self):
        env = DockingEnv(seed=7, shifted=True)
        start = env.reset()
        for _ in range(2):
            tr, _ = env.step([1., 0.])
            self.assertTrue(np.allclose(tr.next_observation[:4], start[:4]))
        tr, _ = env.step([1., 0.])
        self.assertGreater(tr.next_observation[0], start[0])

    def test_her_goals_are_future_positions_in_same_episode(self):
        replay = GoalReplay(seed=7)
        original_goal = np.array([.9, .9], dtype=np.float32)
        episodes = []
        for offset in (-.7, .3):
            episode = []
            for i in range(3):
                obs = np.r_[offset + i * .1, offset, 0, 0, original_goal].astype(np.float32)
                ns = obs.copy(); ns[0] += .1
                tr = ReachTransition(obs, np.zeros(2, dtype=np.float32), -1, ns, False, i == 2)
                episode.append(tr); replay.add(tr)
            episodes.append(episode)
        s, a, r, ns, d = replay.sample(200, her=True)
        for old, new, reward in zip(s, ns, r):
            self.assertTrue(np.allclose(old[4:], new[4:]))
            self.assertEqual(reward, float(DockingEnv.compute_reward(new[:2], new[4:])))
            if not np.allclose(old[4:], original_goal):
                ep = episodes[0] if old[1] < 0 else episodes[1]
                candidates = [tr.next_observation[:2] for tr in ep if tr.observation[0] >= old[0] - 1e-6]
                self.assertTrue(any(np.allclose(old[4:], goal) for goal in candidates))
        self.assertTrue(np.all(d == 0))

    def test_gaussian_tanh_density_consistency(self):
        torch.manual_seed(7)
        actor = GaussianActor()
        obs = torch.randn(64, 6)
        action, logp = actor(obs)
        rebuilt = actor.log_prob(obs, action)
        self.assertTrue(torch.allclose(logp, rebuilt, atol=2e-4))
        loss = -(logp - action.square().sum(-1)).mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))

    def test_pd_baseline_and_learned_dynamics_are_independent(self):
        metrics, _ = evaluate(pd_action, episodes=5)
        self.assertEqual(metrics["success_rate"], 1)
        dataset = collect_dataset(30)
        model = LinearWorldModel()
        scores = model.fit(dataset)
        self.assertGreater(scores["validation_transitions"], 0)
        self.assertLess(scores["heldout_state_rmse"], .1)

    def test_gymnasium_goal_and_partial_observation_interfaces(self):
        try:
            from embodied.frameworks import make_env
            from stable_baselines3.common.env_checker import check_env
        except ImportError:
            self.skipTest("Optional framework dependencies not installed")
        for name in ("warehouse", "reach", "reach-sparse"):
            env = make_env(name)
            check_env(env, warn=True)
            env.close()
        env = make_env("reach", partial=True)
        obs, _ = env.reset(seed=7)
        obs, *_ = env.step(np.ones(2, dtype=np.float32))
        self.assertTrue(np.all(obs[2:4] == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
