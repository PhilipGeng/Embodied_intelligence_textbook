"""Report real installations and saved framework smoke tests without heavy auto-install."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

FRAMEWORKS = [
    ("Gymnasium", "gymnasium", "environment", "https://gymnasium.farama.org/introduction/create_custom_env/"),
    ("Stable-Baselines3", "stable_baselines3", "training", "https://stable-baselines3.readthedocs.io/en/master/"),
    ("SB3-Contrib", "sb3_contrib", "recurrent/training", "https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html"),
    ("MuJoCo", "mujoco", "physics", "https://mujoco.readthedocs.io/en/stable/python.html"),
    ("Isaac Lab", "isaaclab", "GPU robotics simulation", "https://docs.nvidia.com/isaaclab/concepts/reinforcement-learning"),
    ("ManiSkill", "mani_skill", "robot manipulation simulation", "https://maniskill.readthedocs.io/en/latest/"),
    ("RSL-RL", "rsl_rl", "robot locomotion training", "https://github.com/leggedrobotics/rsl_rl"),
    ("LeRobot", "lerobot", "demonstrations/IL/VLA", "https://huggingface.co/docs/lerobot/en/index"),
    ("DreamerV3", "dreamerv3", "latent world-model training", "https://github.com/danijar/dreamerv3"),
]


def diagnose(root=ROOT):
    rows = []
    for name, module_name, purpose, url in FRAMEWORKS:
        try:
            present = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            present = False
        row = dict(name=name, module=module_name, purpose=purpose, official=url,
                   installed=present, status="检测到但未导入验证" if present else "未安装")
        if present and module_name in ("gymnasium", "stable_baselines3", "sb3_contrib", "mujoco"):
            try:
                module = importlib.import_module(module_name)
                row.update(version=getattr(module, "__version__", "unknown"), status="导入验证通过")
            except Exception as exc:
                row.update(status="导入失败", error=str(exc))
        rows.append(row)
    smoke = []
    for path in (Path(root) / "output" / "frameworks").glob("*/report.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        smoke.append(dict(id=path.parent.name, **report))
    import torch
    return dict(frameworks=rows, smoke_experiments=smoke,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                note="Installed/imported is not trained-to-convergence. Saved short runs validate API flow only. GPU learning does not imply GPU physics.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "framework_status.json")
    args = parser.parse_args()
    result = diagnose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
