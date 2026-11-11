"""Honest capability status plus actual local experiment artifacts."""
import csv
import json
from pathlib import Path
import numpy as np

CAPABILITIES = [
    ("连续控制", "教学实现可运行", "Gaussian SAC / TD3，连续力与惯性", "python run_embodied.py sac --device cuda"),
    ("稀疏奖励与目标条件学习", "教学实现可运行", "Future HER；相同稀疏奖励的 SAC 对照", "python run_embodied.py her --device cuda"),
    ("模仿学习与数据覆盖", "教学实现可运行", "BC，程序演示与目标覆盖消融", "python run_embodied.py bc"),
    ("离线强化学习", "教学实现可运行", "IQL；BC/IQL 共享固定数据集", "python run_embodied.py iql --dataset output/embodied/bc_seed7_cuda/dataset.npz"),
    ("模型学习与规划", "教学实现可运行", "Delta 动力学与 CEM-MPC；非 Dreamer", "python run_embodied.py mpc"),
    ("随机化与域迁移", "教学实现可运行", "质量、阻尼、延迟；简化 sim-to-sim 对照", "python run_embodied.py sac --randomize"),
    ("环境标准接口", "第三方接口已验证", "Gymnasium 普通、连续、GoalEnv", "python framework_doctor.py"),
    ("循环策略 / POMDP", "第三方短跑已验证", "SB3-Contrib Recurrent PPO，隐藏速度", "python run_framework.py recurrent-ppo --env reach --partial --steps 512 --device cpu"),
    ("关节物理仿真", "第三方短跑已验证", "MuJoCo Reacher-v5 + SB3 SAC", "python run_framework.py sac --env Reacher-v5 --steps 512"),
    ("大规模 GPU 机器人仿真", "接入路线，未安装", "Isaac Lab / ManiSkill；查官方安装约束", "见 EMBODIED_TECHNIQUES.md"),
    ("真机、动作分块与 VLA", "接入路线，未安装", "LeRobot / ACT / Diffusion Policy；需相应数据", "见 EMBODIED_TECHNIQUES.md"),
    ("完整潜在世界模型", "桥梁已备，完整实现待补", "RSSM / Dreamer；现有状态模型不冒充 Dreamer", "见 WORLD_MODELS.md"),
    ("教师学生、AMP、腿式控制", "后续专章", "需关节任务、参考运动与物理仿真", "见 EMBODIED_TECHNIQUES.md"),
    ("安全约束、多机器人", "后续专章", "动作限幅不等于 CBF/CPO；单体不等于 MARL", "见 EMBODIED_TECHNIQUES.md"),
]


def load_embodied(project):
    project = Path(project)
    runs = []
    for folder in sorted((project / "output" / "embodied").glob("*")):
        if not (folder / "report.json").exists():
            continue
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        run = dict(id=folder.name, report=report, trajectories={}, learning=[])
        for domain in ("nominal", "shifted"):
            path = folder / f"trajectory_{domain}.json"
            if path.exists():
                run["trajectories"][domain] = json.loads(path.read_text(encoding="utf-8"))
        path = folder / "learning.csv"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                run["learning"] = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
        if (folder / "dynamics_weights.npy").exists():
            run["dynamics_weights"] = np.load(folder / "dynamics_weights.npy").tolist()
        runs.append(run)
    doctor = project / "output" / "framework_status.json"
    status = json.loads(doctor.read_text(encoding="utf-8")) if doctor.exists() else {}
    return dict(capabilities=[dict(title=t, status=s, detail=d, command=c) for t,s,d,c in CAPABILITIES],
                runs=runs, frameworks=status)
