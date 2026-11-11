"""Continuous embodied-RL lessons: SAC, TD3, HER, BC, IQL and learned MPC."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
import numpy as np
from embodied.env import DockingEnv, pd_action, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("algorithm", choices=["sac", "td3", "her", "bc", "iql", "mpc", "pd"])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--episodes", type=int, default=100, help="Programmatic demonstration collection episodes")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--sparse", action="store_true")
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--random-goals-only", action="store_true", help="Dataset coverage ablation: omit fixed deployment-goal demos")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if min(args.steps, args.updates, args.episodes, args.eval_episodes) < 1:
        p.error("Budgets must be positive")
    if args.sparse and args.algorithm in ("bc", "iql", "mpc", "pd"):
        p.error("Offline/MPC/PD lessons use dense docking rewards; sparse is an online/HER comparison.")
    if args.randomize and args.algorithm not in ("sac", "td3", "her"):
        p.error("--randomize is an online training intervention; choose sac, td3 or her")
    sparse = args.sparse or args.algorithm == "her"
    neural = args.algorithm in ("sac", "td3", "her", "bc", "iql")
    from warehouse.device import resolve_device, device_metadata
    device = resolve_device(args.device) if neural else "cpu"
    metadata = device_metadata(device, args.device)
    suffix = ("_sparse" if sparse and args.algorithm != "her" else "") + ("_randomized" if args.randomize else "")
    folder = f"{args.algorithm}{suffix}_seed{args.seed}" + ("_cuda" if str(device).startswith("cuda") else "")
    output = args.output or ROOT / "output" / "embodied" / folder
    output.mkdir(parents=True, exist_ok=True)
    report = dict(algorithm=args.algorithm, family="continuous_docking", seed=args.seed,
                  sparse=sparse, randomized_training=args.randomize, gamma=.98, **metadata,
                  scope="2D point-mass docking teaching simulator; not articulated robotics or real-robot sim-to-real")
    print(f"{args.algorithm}: {metadata['device_name']}", flush=True)
    logs, dataset = [], None
    if args.algorithm in ("sac", "td3", "her"):
        from embodied.learners import train_online
        from embodied.networks import action_function
        import torch
        actor, critic, logs = train_online(args.algorithm, args.steps, args.seed, device, sparse, args.randomize)
        torch.save(actor.state_dict(), output / "actor.pt")
        torch.save(critic.state_dict(), output / "critic.pt")
        policy_factory = lambda: action_function(actor, device)
        report["training_environment_steps"] = args.steps
    elif args.algorithm in ("bc", "iql", "mpc"):
        from embodied.offline import collect_dataset, train_offline
        if args.dataset:
            with np.load(args.dataset) as raw:
                dataset = {key: raw[key] for key in raw.files}
            report["dataset_origin"] = str(args.dataset.resolve())
        else:
            dataset = collect_dataset(args.episodes, args.seed, include_task_goal=not args.random_goals_only)
            report["dataset_origin"] = "programmatic PD + noise + random behavior; not human demonstrations"
            report["dataset_coverage"] = "random goals only" if args.random_goals_only else "random goals + 25% deployment-goal episodes"
        np.savez_compressed(output / "dataset.npz", **dataset)
        report["dataset_transitions"] = len(dataset["rewards"])
        report["dataset_hash"] = hashlib.sha256(dataset["observations"].tobytes() + dataset["actions"].tobytes()).hexdigest()
        report["training_environment_steps"] = 0
        if args.algorithm == "mpc":
            from embodied.planning import LinearWorldModel, MPC
            model = LinearWorldModel()
            report["model_validation"] = model.fit(dataset)
            np.save(output / "dynamics_weights.npy", model.weights)
            policy_factory = lambda: MPC(model, seed=args.seed)
        else:
            from embodied.networks import action_function
            import torch
            actor, critic, value, logs = train_offline(args.algorithm, dataset, args.updates, args.seed, device)
            torch.save(actor.state_dict(), output / "actor.pt")
            if args.algorithm == "iql":
                torch.save(critic.state_dict(), output / "critic.pt")
                torch.save(value.state_dict(), output / "value.pt")
            policy_factory = lambda: action_function(actor, device)
            report["gradient_updates"] = args.updates
    else:
        policy_factory = lambda: pd_action
        report["training_environment_steps"] = 0
        report["scope"] = "handcrafted PD controller reference; no learning"
    report["evaluation"] = {}
    for domain in ("nominal", "shifted"):
        score, route = evaluate(policy_factory(), episodes=args.eval_episodes, sparse=sparse, shifted=domain == "shifted")
        report["evaluation"][domain] = score
        (output / f"trajectory_{domain}.json").write_text(json.dumps(route, indent=2), encoding="utf-8")
    if logs:
        with (output / "learning.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(logs[0])); writer.writeheader(); writer.writerows(logs)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("Saved:", output.resolve(), flush=True)


if __name__ == "__main__":
    main()
