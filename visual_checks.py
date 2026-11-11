"""Validate exported diagnostic values against the real saved checkpoints."""
from pathlib import Path
import sys
root = Path(__file__).parent
if (root / ".deps").exists():
    sys.path.insert(0, str(root / ".deps"))
import unittest
import numpy as np
from warehouse.inspection import Inspector, bellman_update_demo, add_world_predictions, build_dashboard_data


class VisualChecks(unittest.TestCase):
    def test_saved_ppo_trace_bellman_and_gae_are_consistent(self):
        inspector = Inspector(root / "output" / "ppo_seed7")
        episode = inspector.rollout(seed=123)
        self.assertTrue(episode["success"])
        for i, f in enumerate(episode["frames"]):
            self.assertAlmostEqual(f["target"], f["reward"] + inspector.gamma * f["next_value"])
            self.assertAlmostEqual(f["delta"], f["target"] - f["value"])
            direct = sum((inspector.gamma * .95) ** j * x["delta"]
                         for j, x in enumerate(episode["frames"][i:]))
            self.assertAlmostEqual(f["advantage"], direct)
            self.assertAlmostEqual(sum(f["probabilities"]), 1, places=6)
        self.assertEqual(episode["frames"][-1]["next_value"], 0)

    def test_terminal_q_target_ignores_spurious_next_q(self):
        inspector = Inspector(root / "output" / "q_seed7")
        inspector.q[41] = 999  # delivery state (0,6,carrying=1)
        episode = inspector.rollout(seed=123)
        last = episode["frames"][-1]
        self.assertTrue(last["terminated"])
        self.assertEqual(last["target"], last["reward"])

    def test_sac_soft_value_matches_exact_action_sum(self):
        inspector = Inspector(root / "output" / "sac_seed7")
        frame = inspector.rollout()["frames"][0]
        ns = frame["next_state"]
        p = inspector.probabilities[ns]
        q = np.minimum(inspector.q1[ns], inspector.q2[ns])
        alpha = inspector.report["temperature"]
        expected = np.sum(p * (q - alpha * np.log(p.clip(1e-12))))
        self.assertAlmostEqual(frame["next_value"], float(expected))

    def test_bellman_demo_only_changes_one_state_value_per_update(self):
        demo = bellman_update_demo()
        for u in demo["updates"]:
            self.assertAlmostEqual(u["target"], u["reward"] + demo["gamma"] * u["next_max"])
            self.assertAlmostEqual(u["new_q"], u["old_q"] + demo["alpha"] * u["delta"])
            changed = np.flatnonzero(np.array(u["values_after"]) != np.array(u["values_before"]))
            self.assertTrue(all(s == u["state"] for s in changed))

    def test_world_prediction_distribution_and_export_structure(self):
        episode = Inspector(root / "output" / "ppo_seed7").rollout()
        add_world_predictions(root / "output" / "world-model_seed7", [episode])
        for frame in episode["frames"]:
            self.assertAlmostEqual(sum(frame["world"]["grid_probabilities"]), 1, places=5)
            self.assertGreaterEqual(frame["world"]["continuation"], 0)
            self.assertLessEqual(frame["world"]["continuation"], 1)
        data = build_dashboard_data(root, include_world=False)
        self.assertTrue(any(r["algorithm"] == "ppo" for r in data["runs"]))
        self.assertEqual({p["report"]["algorithm"] for p in data["predictions"]}, {"mc", "td"})
        self.assertEqual(data["warnings"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
