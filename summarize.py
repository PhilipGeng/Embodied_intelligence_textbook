"""Summarize actual saved experiments, without ranking prediction against control."""
import csv
import json
from pathlib import Path
import sys
root = Path(__file__).parent
if (root / ".deps").exists():
    sys.path.insert(0, str(root / ".deps"))


def main():
    output = root / "output"
    reports = []
    for path in sorted(output.glob("*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        learning_path = path.parent / "learning.csv"
        if report.get("environment_steps") is None and learning_path.exists():
            with learning_path.open(encoding="utf-8") as f:
                entries = list(csv.DictReader(f))
            if entries:
                report["environment_steps"] = int(entries[-1]["steps"])
        reports.append(report)
    rows = []
    for report in reports:
        score = report.get("evaluation_result")
        if score:
            rows.append(dict(algorithm=report["algorithm"], seed=report["seed"],
                             device=report.get("device", "cpu (legacy)"),
                             shaping=report["shaping"], environment_steps=report.get("environment_steps"),
                             **score))
    if rows:
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines = ["# 实际运行结果", "", "这是教学验证，单随机种子不能用于算法优劣结论。",
             "预算单位分 episodes 与 environment steps；REINFORCE、表格方法与神经方法预算不等。",
             "MC/TD 是固定策略评价实验，不是策略改进。世界模型结果是预测检验，不是控制成功率。", "",
             "|算法|种子|设备|塑形|环境交互步数|送达成功率|危险区失败率|平均步数|原始折扣回报|",
             "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["algorithm"] in ("mc", "td"):
            continue
        lines.append(f"|{row['algorithm']}|{row['seed']}|{row['device']}|{row['shaping']}|{row['environment_steps']}|"
                     f"{row['success_rate']:.1%}|{row['hazard_rate']:.1%}|{row['mean_steps']:.2f}|"
                     f"{row['base_discounted_return']:.3f}|")
    lines += ["", "## 固定策略评价（不做策略改进）", "",
              "|方法|环境步数|起点价值估计|精确 Bellman 期望|绝对误差|",
              "|---|---:|---:|---:|---:|"]
    for report in reports:
        if "prediction_error" in report:
            error = report["prediction_error"]
            lines.append(f"|{report['algorithm']}|{error['steps']}|{error['start_estimate']:.4f}|"
                         f"{error['start_exact']:.4f}|{abs(error['start_estimate']-error['start_exact']):.4f}|")
    lines += ["", "## 世界模型预测验证（不是控制成功率）", ""]
    for report in reports:
        if "heldout_next_state_nll" in report:
            lines += [f"- seed={report['seed']}；按完整轨迹分开训练/验证。",
                      f"- 下一状态 NLL：{report['heldout_next_state_nll']:.4f}；"
                      f"top-1 准确率：{report['heldout_top1_state_accuracy']:.1%}。",
                      f"- 奖励 RMSE：{report['heldout_reward_rmse']:.4f}；"
                      f"continuation 准确率：{report['heldout_continuation_accuracy']:.1%}。"]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for path in sorted(output.glob("*/learning.csv")):
            with path.open(encoding="utf-8") as f:
                learning = list(csv.DictReader(f))
            if not learning:
                continue
            x = [float(r["steps"]) for r in learning]
            if "success_rate" in learning[0]:
                axes[0].plot(x, [float(r["success_rate"]) for r in learning], label=path.parent.name)
            else:
                axes[1].plot(x, [abs(float(r["start_estimate"]) - float(r["start_exact"]))
                                 for r in learning], label=path.parent.name)
        axes[0].set(title="Control: independent evaluation", xlabel="Environment steps", ylabel="Delivery success rate")
        axes[1].set(title="Prediction: same fixed policy", xlabel="Environment steps", ylabel="Absolute start-value error")
        for ax in axes:
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "learning_curves.png", dpi=160)
        plt.close(fig)
        from matplotlib.patches import Rectangle
        from warehouse.env import MAP
        candidates = [output / "ppo_seed7_cuda" / "example_route.json",
                      output / "ppo_seed7" / "example_route.json"]
        route_path = next((p for p in candidates if p.exists()), candidates[-1])
        route = json.loads(route_path.read_text(encoding="utf-8")) if route_path.exists() else []
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        palette = {"#": "#374151", "X": "#fca5a5", "P": "#fde68a", "D": "#86efac", "S": "#bfdbfe", ".": "#f8fafc"}
        for phase, ax in enumerate(axes):
            for r, row in enumerate(MAP):
                for c, cell in enumerate(row):
                    ax.add_patch(Rectangle((c - .5, r - .5), 1, 1,
                                           facecolor=palette[cell], edgecolor="#cbd5e1"))
                    if cell in "SPDX":
                        ax.text(c, r, cell, ha="center", va="center", fontsize=13, weight="bold")
            for tr in route:
                r, c, carrying = tr["state"]
                nr, nc, _ = tr["next_state"]
                if carrying == phase:
                    ax.annotate("", xy=(nc, nr), xytext=(c, r),
                                arrowprops=dict(arrowstyle="->", color="#2563eb" if phase == 0 else "#7c3aed", lw=2))
            ax.set(xlim=(-.5, 6.5), ylim=(4.5, -.5), xticks=range(7), yticks=range(5),
                   title="Stage 1: collect at P" if phase == 0 else "Stage 2: deliver to D")
            ax.set_aspect("equal")
        fig.suptitle("Same warehouse for every algorithm | X: terminal hazard | # walls | 10% slip")
        fig.tight_layout()
        fig.savefig(output / "warehouse_route.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable: saved CSV/Markdown; skipped plot.")
    print("Saved", output / "RESULTS.md")


if __name__ == "__main__":
    main()
