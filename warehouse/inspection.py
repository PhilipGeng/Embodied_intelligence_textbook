"""Read-only model inspection for teaching dashboards; never updates weights."""
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .env import Warehouse, MAP, DELIVERY, HAZARD

ACTOR_ALGORITHMS = {"reinforce", "ac", "gae", "ppo", "sac"}
Q_ALGORITHMS = {"q", "sarsa", "dqn", "bellman"}


def read_run(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    learning = []
    if (folder / "learning.csv").exists():
        with (folder / "learning.csv").open(encoding="utf-8") as f:
            learning = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]
    return report, learning


class Inspector:
    def __init__(self, folder, device="cpu"):
        self.folder = Path(folder)
        self.report, self.learning = read_run(folder)
        self.name = self.report["algorithm"]
        self.gamma = self.report.get("gamma", 0.97)
        self.values = self.q = self.probabilities = self.q1 = self.q2 = None
        if self.name in ("q", "sarsa", "bellman"):
            self.q = np.load(self.folder / "q.npy")
        elif self.name in ("mc", "td"):
            self.values = np.load(self.folder / "value.npy")
        else:
            import torch
            from .neural import Network
            from .device import resolve_device
            selected = resolve_device(device)
            states = torch.arange(70, device=selected)
            def load(filename, outputs):
                model = Network(outputs).to(selected)
                weights = torch.load(self.folder / filename, map_location=selected, weights_only=True)
                model.load_state_dict(weights)
                model.eval()
                with torch.no_grad():
                    return model(states).cpu().numpy()
            output = load("actor.pt", 4)
            if self.name == "dqn":
                self.q = output
            else:
                shifted = output - output.max(axis=1, keepdims=True)
                exp = np.exp(shifted)
                self.probabilities = exp / exp.sum(axis=1, keepdims=True)
            if (self.folder / "critic.pt").exists():
                self.values = load("critic.pt", 1)[:, 0]
            if self.name == "sac":
                self.q1, self.q2 = load("q1.pt", 4), load("q2.pt", 4)

    def action(self, state, rng, sampled=False):
        if self.probabilities is not None:
            return int(rng.choice(4, p=self.probabilities[state])) if sampled else int(self.probabilities[state].argmax())
        return int(self.q[state].argmax())

    def heatmap(self):
        if self.values is not None:
            return self.values.tolist(), "Critic V(s)"
        if self.q is not None:
            return self.q.max(axis=1).tolist(), "max Q(s,a)"
        if self.q1 is not None:
            return np.minimum(self.q1, self.q2).max(axis=1).tolist(), "max min(Q1,Q2)"
        return None, None

    def rollout(self, seed=123, sampled=False):
        env = Warehouse(seed=seed, gamma=self.gamma, shaping=self.report.get("shaping", True))
        rng = np.random.default_rng(seed + 17713)
        state, frames = env.reset(), []
        total = base_total = 0.0
        for t in range(env.max_steps):
            action = self.action(state, rng, sampled)
            tr = env.step(action)
            frame = asdict(tr)
            frame.update(index=t, position=list(env.decode(state)), next_position=list(env.decode(tr.next_state)))
            frame["probabilities"] = self.probabilities[state].tolist() if self.probabilities is not None else None
            frame["q_values"] = self.q[state].tolist() if self.q is not None else None
            total += self.gamma ** t * tr.reward
            base_total += self.gamma ** t * tr.base_reward
            frame.update(training_return_so_far=total, base_return_so_far=base_total)
            if self.values is not None:
                value = float(self.values[state])
                next_value = 0.0 if tr.terminated else float(self.values[tr.next_state])
                target = tr.reward + self.gamma * next_value
                frame.update(value=value, next_value=next_value, target=target, delta=target - value,
                             target_kind="V-TD", current_estimate=value)
            elif self.q is not None:
                next_action = None
                if self.name == "sarsa":
                    next_action = self.action(tr.next_state, rng) if not tr.terminated else 0
                    next_value = float(self.q[tr.next_state, next_action])
                else:
                    next_value = float(self.q[tr.next_state].max())
                next_value = 0.0 if tr.terminated else next_value
                estimate = float(self.q[state, action])
                target = tr.reward + self.gamma * next_value
                frame.update(next_value=next_value, target=target, delta=target - estimate,
                             current_estimate=estimate, target_kind="SARSA-greedy" if self.name == "sarsa" else "Q-max")
            elif self.q1 is not None:
                alpha = self.report.get("temperature", 1.0)
                next_p = self.probabilities[tr.next_state]
                next_q = np.minimum(self.q1[tr.next_state], self.q2[tr.next_state])
                next_value = 0.0 if tr.terminated else float(np.sum(next_p * (next_q - alpha * np.log(next_p.clip(1e-12)))))
                estimate = float(min(self.q1[state, action], self.q2[state, action]))
                target = tr.reward + self.gamma * next_value
                frame.update(q1=float(self.q1[state, action]), q2=float(self.q2[state, action]),
                             temperature=alpha, next_value=next_value, target=target, delta=target - estimate,
                             target_kind="SAC-soft", current_estimate=estimate)
            frames.append(frame)
            state = tr.next_state
            if tr.terminated or tr.truncated:
                break
        # Finite-window rewards are not a complete Monte Carlo return on timeout.
        future = 0.0
        for frame in reversed(frames):
            future = frame["reward"] + self.gamma * future
            frame["observed_return_to_go"] = future
            frame["return_is_complete"] = frames[-1]["terminated"]
        if self.values is not None:
            last = 0.0
            lam = 0.0 if self.name == "ac" else 0.95
            for frame in reversed(frames):
                boundary = frame["terminated"] or frame["truncated"]
                last = frame["delta"] + self.gamma * lam * (not boundary) * last
                frame["advantage"] = last
                frame["lambda"] = lam
        r, c, phase = env.decode(state)
        status = "送达成功" if (r, c) == DELIVERY and phase else "进入危险区" if (r, c) == HAZARD else "采样时间截断"
        return dict(frames=frames, status=status, success=status == "送达成功", steps=len(frames),
                    base_return=base_total, training_return=total)


def add_world_predictions(model_folder, episodes, device="cpu"):
    import torch
    from .world_model import StateWorldModel
    from .device import resolve_device
    selected = resolve_device(device)
    model = StateWorldModel().to(selected)
    model.load_state_dict(torch.load(Path(model_folder) / "world_model.pt", map_location=selected, weights_only=True))
    model.eval()
    all_frames = [frame for episode in episodes for frame in episode["frames"]]
    if not all_frames:
        return
    with torch.no_grad():
        states = torch.tensor([f["state"] for f in all_frames], device=selected)
        actions = torch.tensor([f["action"] for f in all_frames], device=selected)
        logits, rewards, continuation = model(states, actions)
        probs = logits.softmax(-1).cpu().numpy()
        rewards, continuation = rewards.cpu().numpy(), continuation.sigmoid().cpu().numpy()
    for frame, p, r, cont in zip(all_frames, probs, rewards, continuation):
        top = np.argsort(p)[-4:][::-1]
        frame["world"] = dict(model_id=Path(model_folder).name, reward=float(r), continuation=float(cont),
                              actual_next_probability=float(p[frame["next_state"]]),
                              grid_probabilities=(p[:35] + p[35:]).tolist(),
                              top=[dict(state=int(s), position=list(Warehouse.decode(int(s))), probability=float(p[s])) for s in top])


def bellman_update_demo(seed=123):
    """Actual Q updates on an explicitly labeled, reverse-ordered teaching replay."""
    env = Warehouse(seed=seed, slip=0.0)
    state, samples = env.reset(), []
    for action in [2] * 4 + [1] * 6 + [0] * 4:
        tr = env.step(action)
        samples.append(tr)
        state = tr.next_state
    q, updates = np.zeros((70, 4)), []
    for tr in reversed(samples):
        old = float(q[tr.state, tr.action])
        next_max = 0.0 if tr.terminated else float(q[tr.next_state].max())
        target = tr.reward + env.gamma * next_max
        delta = target - old
        new = old + 0.5 * delta
        before = q.max(axis=1).tolist()
        q[tr.state, tr.action] = new
        updates.append(dict(state=tr.state, action=tr.action, next_state=tr.next_state,
                            position=list(env.decode(tr.state)), next_position=list(env.decode(tr.next_state)),
                            reward=tr.reward, base_reward=tr.base_reward, terminated=tr.terminated,
                            old_q=old, next_max=next_max, target=target, delta=delta, new_q=new,
                            values_before=before, values_after=q.max(axis=1).tolist()))
    return dict(alpha=0.5, gamma=env.gamma, updates=updates,
                explanation="独立教学回放：零打滑的一条成功轨迹，按终点到起点的顺序做真实 Q-learning 更新；不是历史训练日志，也不是一次在线回合的时间顺序。")


def build_dashboard_data(project, seed=123, device="cpu", include_world=True):
    project = Path(project)
    runs, predictions, model_reports, warnings = [], [], [], []
    model_folders = {}
    for folder in sorted((project / "output").glob("*")):
        if not folder.is_dir() or not (folder / "report.json").exists():
            continue
        report, learning = read_run(folder)
        name = report.get("algorithm")
        if name == "world-model":
            model_reports.append(dict(id=folder.name, report=report))
            if (folder / "world_model.pt").exists():
                model_folders[(report.get("shaping", True), report.get("gamma", .97))] = folder
            continue
        if name in ("mc", "td"):
            predictions.append(dict(id=folder.name, report=report, learning=learning))
            continue
        if name not in ACTOR_ALGORITHMS | Q_ALGORITHMS:
            continue
        try:
            inspector = Inspector(folder, device=device)
            values, value_label = inspector.heatmap()
            entry = dict(id=folder.name, algorithm=name, report=report, learning=learning,
                         gamma=inspector.gamma, values=values, value_label=value_label,
                         greedy=inspector.rollout(seed), sampled=None)
            if name in ACTOR_ALGORITHMS:
                entry["sampled"] = inspector.rollout(seed, sampled=True)
            runs.append(entry)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            warnings.append(f"{folder.name}: {exc}")
    if not runs:
        raise RuntimeError("No loadable policy checkpoints. Run python run.py q or python run.py ppo first.")
    if include_world:
        for (shaping, gamma), model_folder in model_folders.items():
            matching = [r[mode] for r in runs for mode in ("greedy", "sampled")
                        if r[mode] and r["report"].get("shaping", True) == shaping and r["gamma"] == gamma]
            try:
                add_world_predictions(model_folder, matching, device)
            except (OSError, RuntimeError, ValueError) as exc:
                warnings.append(f"World model prediction: {exc}")
    from textbook.content import build_lessons
    from embodied.catalog import load_embodied
    return dict(schema_version=2, map=list(MAP), seed=seed, inference_device=str(device), runs=runs,
                predictions=predictions, world_models=model_reports, warnings=warnings,
                bellman=bellman_update_demo(seed), lessons=build_lessons(project), embodied=load_embodied(project))
