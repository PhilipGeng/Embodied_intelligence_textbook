"""Replay a saved learner step-by-step: state, action, reward and optional TD error."""
import argparse
from pathlib import Path
import sys
import numpy as np
root = Path(__file__).parent
if (root / ".deps").exists():
    sys.path.insert(0, str(root / ".deps"))
from warehouse.env import Warehouse, MAP, ACTION_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("algorithm", choices=["q", "sarsa", "bellman", "dqn", "reinforce", "ac", "gae", "ppo", "sac"])
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--sample", action="store_true", help="Sample neural actor instead of greedy evaluation")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    if args.algorithm in ("q", "sarsa", "bellman"):
        device = "cpu"
    else:
        from warehouse.device import resolve_device
        try:
            device = resolve_device(args.device)
        except (RuntimeError, ValueError) as exc:
            p.error(str(exc))
    default_folder = root / "output" / f"{args.algorithm}_seed7"
    cuda_folder = root / "output" / f"{args.algorithm}_seed7_cuda"
    if str(device).startswith("cuda") and cuda_folder.exists():
        default_folder = cuda_folder
    folder = args.checkpoint or default_folder
    print("Checkpoint:", folder.resolve())
    import json
    metadata = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    env = Warehouse(seed=args.seed, shaping=metadata["shaping"], gamma=metadata["gamma"])
    critic = None
    if args.algorithm in ("q", "sarsa", "bellman"):
        q = np.load(folder / "q.npy")
        def decide(state):
            return int(q[state].argmax()), "Q=" + np.array2string(q[state], precision=2)
    else:
        import torch
        from warehouse.neural import Network, setup
        try:
            device = setup(args.seed, device)
        except (RuntimeError, ValueError) as exc:
            p.error(str(exc))
        print("Inference device:", device)
        actor = Network(4).to(device)
        actor.load_state_dict(torch.load(folder / "actor.pt", map_location=device, weights_only=True))
        actor.eval()
        if (folder / "critic.pt").exists():
            critic = Network(1).to(device)
            critic.load_state_dict(torch.load(folder / "critic.pt", map_location=device, weights_only=True))
            critic.eval()
        def decide(state):
            with torch.no_grad():
                values = actor(torch.tensor(state, device=device))
                if args.algorithm == "dqn":
                    return int(values.argmax()), "Q=" + str(values.cpu().numpy().round(2))
                distribution = torch.distributions.Categorical(logits=values)
                action = int(distribution.sample()) if args.sample else int(values.argmax())
                return action, "pi=" + str(distribution.probs.cpu().numpy().round(3))
    state = env.reset()
    total, base_total = 0.0, 0.0
    print("\n".join(MAP))
    print("State=(row,column,carrying); actions indexed [up,right,down,left]")
    for t in range(env.max_steps):
        action, explanation = decide(state)
        tr = env.step(action)
        detail = ""
        if critic is not None:
            with torch.no_grad():
                v = float(critic(torch.tensor(state, device=device)).item())
                next_v = 0.0 if tr.terminated else float(critic(torch.tensor(tr.next_state, device=device)).item())
            delta = tr.reward + env.gamma * next_v - v
            detail = f" V={v:.3f} target={tr.reward+env.gamma*next_v:.3f} delta={delta:+.3f}"
        print(f"{t:02d} {env.decode(state)} --{ACTION_NAMES[action]}--> {env.decode(tr.next_state)}"
              f" base_r={tr.base_reward:+.2f} train_r={tr.reward:+.2f}{detail}")
        print("   ", explanation)
        total += env.gamma ** t * tr.reward
        base_total += env.gamma ** t * tr.base_reward
        state = tr.next_state
        if tr.terminated or tr.truncated:
            break
    print(f"Ended: terminated={tr.terminated}, truncated={tr.truncated},"
          f" base_return={base_total:.3f}, training_return={total:.3f}")
    print("This is one sampled episode, not a policy-quality estimate.")


if __name__ == "__main__":
    main()
