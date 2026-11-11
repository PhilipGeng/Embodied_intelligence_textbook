"""Run every lesson with explicit budgets; single-seed runs are teaching demos."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--world-model", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    root = Path(__file__).parent
    for seed in args.seeds:
        for name in ("bellman", "mc", "td", "q", "sarsa", "dqn", "reinforce", "ac", "gae", "ppo", "sac"):
            episodes = min(args.episodes, 2000) if name == "reinforce" else args.episodes
            cmd = [sys.executable, str(root / "run.py"), name, "--seed", str(seed),
                   "--episodes", str(episodes), "--steps", str(args.steps), "--device", args.device]
            if args.no_shaping:
                cmd += ["--no-shaping"]
            subprocess.run(cmd, check=True)
        if args.world_model:
            cmd = [sys.executable, str(root / "run.py"), "world-model", "--seed", str(seed),
                   "--episodes", "300", "--steps", "2000", "--device", args.device]
            if args.no_shaping:
                cmd += ["--no-shaping"]
            subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, str(root / "summarize.py")], check=True)


if __name__ == "__main__":
    main()
