"""Shared command line for the entire warehouse RL course."""
import argparse
import csv
import json
from pathlib import Path
import sys

# Optional project-local dependency repair; never changes global site-packages.
local_deps = Path(__file__).parent / ".deps"
if local_deps.exists():
    sys.path.insert(0, str(local_deps))

import numpy as np
from warehouse.env import Warehouse, evaluate, MAP
from warehouse.tabular import value_iteration, control, prediction


def save_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("algorithm", choices=["bellman", "mc", "td", "q", "sarsa", "dqn",
                                        "reinforce", "ac", "gae", "ppo", "sac", "world-model"])
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--steps", type=int, default=65536)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-shaping", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Neural compute device; table algorithms always use CPU.")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.episodes < 1 or args.steps < 1:
        p.error("budgets must be positive")
    shaping = not args.no_shaping
    neural = args.algorithm in ("dqn", "reinforce", "ac", "gae", "ppo", "sac", "world-model")
    if neural:
        from warehouse.device import resolve_device, device_metadata
        try:
            device = resolve_device(args.device)
        except (RuntimeError, ValueError) as exc:
            p.error(str(exc))
        metadata = device_metadata(device, args.device)
    else:
        device = "cpu"
        metadata = dict(requested_device=args.device, device="cpu", device_name="CPU (NumPy table)")
    suffix = "_cuda" if str(device).startswith("cuda") else ""
    task_suffix = "_sparse" if args.no_shaping else ""
    out = args.output or Path(__file__).parent / "output" / f"{args.algorithm}{task_suffix}_seed{args.seed}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    env = Warehouse(seed=args.seed, shaping=shaping)
    print("\n".join(MAP), flush=True)
    print(f"Compute device: {metadata['device']} ({metadata['device_name']})", flush=True)
    logs, store = [], None
    result = dict(algorithm=args.algorithm, seed=args.seed, shaping=shaping, gamma=env.gamma, **metadata,
                  evaluation="greedy policy, independent seed, 200 episodes, cap=120 steps")
    if args.algorithm == "bellman":
        value, q = value_iteration(env)
        policy = lambda s: int(q[s].argmax())
        result["exact_start_training_value"] = float(value[env.reset()])
        np.save(out / "q.npy", q)
    elif args.algorithm in ("q", "sarsa"):
        q, logs = control(args.algorithm, args.episodes, args.seed, shaping)
        policy = lambda s: int(q[s].argmax())
        np.save(out / "q.npy", q)
    elif args.algorithm in ("mc", "td"):
        value, logs, probabilities = prediction(args.algorithm, args.episodes, args.seed, shaping)
        np.save(out / "value.npy", value)
        rng = np.random.default_rng(12345)
        policy = lambda s: int(rng.choice(4, p=probabilities[s]))
        result["evaluation"] = "fixed 10%-exploring reference policy; prediction only, no actor improvement"
        result["prediction_error"] = logs[-1]
    elif args.algorithm == "world-model":
        from warehouse.world_model import experiment
        result.update(experiment(args.episodes, args.steps, args.seed, shaping, out, device=device))
        policy = None
    else:
        import torch
        from warehouse.neural import dqn, reinforce, actor_critic, sac
        if args.algorithm == "dqn":
            actor, logs, store = dqn(args.steps, args.seed, shaping, device=device)
        elif args.algorithm == "reinforce":
            actor, logs, store = reinforce(args.episodes, args.seed, shaping, device=device)
        elif args.algorithm == "sac":
            actor, q1, q2, logs, store = sac(args.steps, args.seed, shaping, device=device)
            torch.save(q1.state_dict(), out / "q1.pt")
            torch.save(q2.state_dict(), out / "q2.pt")
            result["variant"] = "discrete SAC; exact four-action expectation; automatic temperature"
            result["temperature"] = logs[-1]["temperature"]
        else:
            actor, critic, logs, store = actor_critic(args.algorithm, args.steps, args.seed, shaping, device=device)
            torch.save(critic.state_dict(), out / "critic.pt")
        torch.save(actor.state_dict(), out / "actor.pt")
        def policy(s):
            with torch.no_grad():
                return int(actor(torch.tensor(s, device=device)).argmax())
    if policy is not None:
        result["evaluation_result"] = evaluate(policy, episodes=200, seed=20000, shaping=shaping)
        route, state = [], env.reset()
        for _ in range(120):
            action = policy(state)
            tr = env.step(action)
            route.append(dict(state=env.decode(state), action=action, reward=tr.reward,
                              base_reward=tr.base_reward, next_state=env.decode(tr.next_state),
                              terminated=tr.terminated, truncated=tr.truncated))
            state = tr.next_state
            if tr.terminated or tr.truncated:
                break
        (out / "example_route.json").write_text(json.dumps(route, indent=2), encoding="utf-8")
    if store:
        store.save(out / "episodes.json")
        result["stored_episodes"] = len(store.episodes)
    if logs:
        result["environment_steps"] = args.steps if args.algorithm in ("dqn", "sac") else logs[-1].get("steps")
    result["requested_budget"] = {"episodes": args.episodes, "steps": args.steps}
    save_csv(out / "learning.csv", logs)
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print("Saved:", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
