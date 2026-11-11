"""Run a real external SB3 implementation on this project or a Gymnasium task."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("algorithm", choices=["ppo", "sac", "td3", "her-sac", "recurrent-ppo"])
    p.add_argument("--env", default="reach")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--partial", action="store_true")
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    try:
        import torch
        import stable_baselines3 as sb3
        from stable_baselines3 import PPO, SAC, TD3, HerReplayBuffer
        from stable_baselines3.common.env_checker import check_env
        from embodied.frameworks import make_env
        from warehouse.device import resolve_device, device_metadata
    except ImportError as exc:
        p.error(f"Optional framework dependency missing: {exc}. Install requirements-frameworks.txt")
    if args.steps < 1:
        p.error("--steps must be positive")
    if args.algorithm == "her-sac" and args.env != "reach-sparse":
        p.error("Use --env reach-sparse for the GoalEnv HER demonstration")
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    env = make_env(args.env, args.partial, args.randomize)
    check_env(env, warn=True)
    if args.algorithm not in ("ppo", "recurrent-ppo") and not hasattr(env.action_space, "low"):
        p.error("SAC/TD3 need continuous Box actions; warehouse has discrete actions")
    options = dict(seed=args.seed, device=str(device), verbose=0,
                   policy_kwargs=dict(net_arch=[64, 64]))
    policy = "MultiInputPolicy" if args.env == "reach-sparse" else "MlpPolicy"
    if args.algorithm in ("ppo", "recurrent-ppo"):
        cls = PPO
        if args.algorithm == "recurrent-ppo":
            try:
                from sb3_contrib import RecurrentPPO
            except ImportError:
                p.error("Install sb3-contrib to use recurrent-ppo")
            cls, policy = RecurrentPPO, ("MultiInputLstmPolicy" if args.env == "reach-sparse" else "MlpLstmPolicy")
            options["policy_kwargs"].update(lstm_hidden_size=64)
        model = cls(policy, env, n_steps=128, batch_size=64, n_epochs=4, gamma=.98, **options)
    else:
        extra = {}
        if args.algorithm == "her-sac":
            extra = dict(replay_buffer_class=HerReplayBuffer,
                         replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"))
        cls = TD3 if args.algorithm == "td3" else SAC
        model = cls(policy, env, learning_starts=256, buffer_size=30000, batch_size=64,
                    train_freq=4, gradient_steps=1, gamma=.98, **extra, **options)
    model.learn(total_timesteps=args.steps)
    folder = args.output or ROOT / "output" / "frameworks" / f"sb3_{args.algorithm}_{args.env}_seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder / "model")
    test_env = make_env(args.env, args.partial, args.randomize)
    returns, successes = [], []
    for episode in range(10):
        obs, info = test_env.reset(seed=20000 + episode)
        recurrent_state, episode_start, total = None, True, 0.
        for _ in range(1000):
            if args.algorithm == "recurrent-ppo":
                import numpy as np
                action, recurrent_state = model.predict(obs, state=recurrent_state,
                                                        episode_start=np.array([episode_start]), deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total += float(reward)
            episode_start = False
            if terminated or truncated:
                break
        returns.append(total)
        if "is_success" in info:
            successes.append(float(info["is_success"]))
    report = dict(framework="stable-baselines3", version=sb3.__version__, algorithm=args.algorithm,
                  environment=args.env, partial_observation=args.partial, randomized=args.randomize,
                  requested_steps=args.steps, actual_steps=model.num_timesteps,
                  mean_return=sum(returns) / len(returns), **device_metadata(device, args.device))
    if successes:
        report["success_rate"] = sum(successes) / len(successes)
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close(); test_env.close()
    print(json.dumps(report, indent=2), flush=True)
    print("Saved:", folder.resolve(), flush=True)


if __name__ == "__main__":
    main()
