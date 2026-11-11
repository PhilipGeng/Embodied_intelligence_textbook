"""Meaningful environment, Bellman and trajectory-boundary checks (unittest)."""
from pathlib import Path
import sys
deps = Path(__file__).parent / ".deps"
if deps.exists():
    sys.path.insert(0, str(deps))
import unittest
import numpy as np
import torch
from warehouse.env import Warehouse, Transition
from warehouse.tabular import value_iteration, bellman_q, policy_evaluation, prediction
from warehouse.neural import categorical_gae
from warehouse.replay import EpisodeStore


class Checks(unittest.TestCase):
    def test_pickup_changes_markov_state(self):
        env = Warehouse(slip=0, shaping=False)
        ns, reward, base, done = env.deterministic(env.encode(3, 0, 0), 2)
        self.assertEqual(env.decode(ns), (4, 0, 1))
        self.assertAlmostEqual(reward, 2.8)
        self.assertFalse(done)

    def test_delivery_requires_pickup(self):
        env = Warehouse(slip=0, shaping=False)
        self.assertFalse(env.deterministic(env.encode(0, 5, 0), 1)[-1])
        self.assertTrue(env.deterministic(env.encode(0, 5, 1), 1)[-1])
        self.assertTrue(env.deterministic(env.encode(2, 2, 0), 1)[-1])

    def test_observations(self):
        env = Warehouse()
        self.assertEqual(env.observation(mode="rgb").shape, (40, 56, 3))
        self.assertEqual(env.observation(mode="features").shape, (7,))
        self.assertEqual(env.observation().sum(), 1)

    def test_slip_distribution(self):
        env = Warehouse(seed=9, slip=0.2)
        state = env.encode(2, 1, 0)
        outcomes = env.outcomes(state, 1)
        self.assertAlmostEqual(sum(x[0] for x in outcomes), 1)
        counts = {}
        for _ in range(10000):
            env.state, env.elapsed, env.ended = state, 0, False
            ns = env.step(1).next_state
            counts[ns] = counts.get(ns, 0) + 1
        for p, ns, *_ in outcomes:
            expected = sum(o[0] for o in outcomes if o[1] == ns)
            self.assertLess(abs(counts[ns] / 10000 - expected), 0.02)

    def test_bellman_residual_and_shaping_invariance(self):
        env = Warehouse()
        value, q = value_iteration(env)
        raw = Warehouse(shaping=False)
        base_value, base_q = value_iteration(raw)
        for s in env.valid_states:
            self.assertLess(abs(value[s] - max(bellman_q(env, value, s, a) for a in range(4))), 1e-8)
            self.assertAlmostEqual(value[s], base_value[s] - env.potential(s), places=7)
        probabilities = np.eye(4)[q.argmax(axis=1)]
        evaluated = policy_evaluation(env, probabilities)
        self.assertTrue(np.allclose(evaluated, value, atol=1e-7))

    def test_gae_numbers_and_timeout_boundary(self):
        reward = torch.tensor([[0.], [0.], [10.]])
        value = torch.tensor([[6.], [7.], [8.]])
        next_v = torch.tensor([[7.], [8.], [999.]])
        terminal = torch.tensor([[0.], [0.], [1.]])
        result = categorical_gae(reward, value, next_v, terminal, terminal, 0.9, 0.95)
        self.assertTrue(torch.allclose(result[:, 0], torch.tensor([1.93305, 1.91, 2.]), atol=1e-5))
        # t=0 timeout: bootstrap its true next observation, but not the reset episode.
        result = categorical_gae(torch.tensor([[1.], [100.]]), torch.zeros(2, 1),
                                 torch.tensor([[5.], [0.]]), torch.tensor([[0.], [1.]]),
                                 torch.ones(2, 1), 0.9, 0.95)
        self.assertAlmostEqual(result[0, 0].item(), 5.5)

    def test_sequences_do_not_cross_and_have_padding_masks(self):
        store = EpisodeStore([[Transition(0, 1, 1, 1, 1, False, False),
                               Transition(1, 2, 2, 2, 2, True, False)]], seed=7)
        batch = store.sample_sequences(16, 4)
        for states, mask in zip(batch["states"], batch["mask"]):
            n = int(mask.sum())
            self.assertIn(n, (1, 2))
            self.assertEqual(states[n], 2)
            self.assertTrue(np.all(mask[n:] == 0))
        with self.assertRaises(ValueError):
            store.append([Transition(0, 1, 0, 0, 1, True, False),
                          Transition(1, 1, 0, 0, 2, False, False)])

    def test_monte_carlo_matches_exact_policy_expectation(self):
        # Also detects accidental correlation of action and environment RNGs.
        _, logs, _ = prediction("mc", 4000, 7)
        self.assertLess(abs(logs[-1]["start_estimate"] - logs[-1]["start_exact"]), 0.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
