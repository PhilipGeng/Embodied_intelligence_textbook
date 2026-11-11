"""CPU/CUDA forward, gradient, boundary and checkpoint integration checks."""
from pathlib import Path
import sys
root = Path(__file__).parent
if (root / ".deps").exists():
    sys.path.insert(0, str(root / ".deps"))
import tempfile
import unittest
from unittest.mock import patch
import torch
from warehouse.device import resolve_device
from warehouse.neural import Network, dqn, reinforce, actor_critic, sac, categorical_gae
from warehouse.world_model import experiment


class DeviceChecks(unittest.TestCase):
    def test_explicit_cuda_never_silently_falls_back(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto").type, "cpu")
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                resolve_device("cuda")

    def test_all_learners_move_parameters_batches_and_gradients(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for name in ("dqn", "reinforce", "ac", "gae", "ppo", "sac"):
                with self.subTest(device=device, learner=name):
                    if name == "dqn":
                        model = dqn(512, 19, device=device)[0]
                    elif name == "reinforce":
                        model = reinforce(8, 19, device=device)[0]
                    elif name == "sac":
                        model = sac(512, 19, device=device)[0]
                    else:
                        model = actor_critic(name, 512, 19, device=device)[0]
                    self.assertEqual(next(model.parameters()).device.type, device)
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
            with tempfile.TemporaryDirectory(dir=root / "output") as folder:
                metrics = experiment(20, 2, 19, True, Path(folder), device=device)
                self.assertTrue(metrics["heldout_next_state_nll"] > 0)
                weights = torch.load(Path(folder) / "world_model.pt", map_location="cpu", weights_only=True)
                self.assertTrue(all(t.device.type == "cpu" for t in weights.values()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_gpu_predictions_gae_and_checkpoint_portability(self):
        cpu = Network(4)
        gpu = Network(4).cuda()
        gpu.load_state_dict(cpu.state_dict())
        states = torch.tensor([0, 10, 35, 69])
        self.assertTrue(torch.allclose(cpu(states), gpu(states.cuda()).cpu(), atol=1e-6))
        reward = torch.tensor([[0.], [0.], [10.]])
        values = torch.tensor([[6.], [7.], [8.]])
        next_values = torch.tensor([[7.], [8.], [999.]])
        done = torch.tensor([[0.], [0.], [1.]])
        a_cpu = categorical_gae(reward, values, next_values, done, done, .9, .95)
        a_gpu = categorical_gae(reward.cuda(), values.cuda(), next_values.cuda(),
                                done.cuda(), done.cuda(), .9, .95)
        self.assertTrue(torch.allclose(a_cpu, a_gpu.cpu(), atol=1e-6))
        with tempfile.TemporaryDirectory(dir=root / "output") as folder:
            path = Path(folder) / "cuda_actor.pt"
            torch.save(gpu.state_dict(), path)
            restored = Network(4)
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.assertTrue(torch.allclose(cpu(states), restored(states), atol=1e-6))


if __name__ == "__main__":
    (root / "output").mkdir(exist_ok=True)
    unittest.main(verbosity=2)
