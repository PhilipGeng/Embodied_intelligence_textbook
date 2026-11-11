# 具身强化学习实验室

一个独立的强化学习实战项目，从仓库机器人取货送达任务出发，贯穿 Bellman、MC/TD、SARSA、Q-learning、DQN、策略梯度、Actor–Critic、GAE、PPO、SAC，并逐步扩展到世界模型。所有方法使用同一个随机 MDP，便于观察算法之间的联系。它不是物理仿真或真机控制器，而是用来隔离算法原理的具身任务抽象。

## 先运行，不必先读完所有代码

项目位于桌面的 `具身强化学习实验室` 文件夹。进入目录后运行：

```powershell
cd "$env:USERPROFILE\Desktop\具身强化学习实验室"
python checks.py
python run.py bellman
python run.py q --episodes 3000
python run.py ppo --steps 65536
python play.py ppo
python summarize.py
```

## 交互式过程可视化

```powershell
python visualize.py --open
```

这会读取已有检查点与训练 CSV，生成 `output/dashboard.html` 并在默认浏览器打开。页面自带数据、图表和脚本，可以直接双击离线播放，不需要启动网页服务或联网。训练权重不会改变。若刚完成新的训练，再执行一次即可更新页面。

页面提供四个学习环节：

1. **任务回放**：切换已训练算法、播放/暂停、逐步前后移动、拖动时间轴；同时查看取货阶段、动作概率或 Q 值、原始与塑形奖励、V、Bellman 目标、TD/GAE。可以按左右方向键切换步骤，或点击下方记录表。
2. **Bellman 更新**：用一条明确标注的成功教学轨迹，从终点向前回放 Q-learning 更新，显示旧 Q、目标、误差、新 Q 和阶段价值热图。这不是原有训练日志，也不是一次在线轨迹的时间顺序。
3. **训练曲线**：切换送达率、风险、原始回报与步数，选择要比较的算法；MC/TD 的固定策略价值误差单独显示，避免混成控制算法排名。
4. **世界模型**：对同一实际状态与动作，显示预测的下一位置分布、奖励和 continuation，并对照真实结果；只匹配同一奖励配置与 gamma 的模型。页面还列出已有 held-out 验证指标。

任务回放里的 TD/GAE 是用**冻结检查点在新回放轨迹上重算的诊断值**，不是历史更新时的实际优势；PPO 区域是可调优势/概率比率的公式实验，也不伪装成历史训练损失。页面的训练曲线则来自实际保存的 `learning.csv`。这三个来源在界面中分别标注。

```powershell
# 更换回放中的打滑随机序列；表格与神经模型使用同一任务
python visualize.py --seed 456 --open

# 默认在 CPU 做轻量模型检查；也支持 GPU
python visualize.py --device cuda --open

# 不加载世界模型，或导出到其他文件
python visualize.py --skip-world-model --output output/replay.html

# 检查显示出来的目标值、GAE、SAC soft value 和世界模型概率
python visual_checks.py
```

普通任务回放是冻结策略执行过程，不是边训练边动画。若要重现某次历史参数更新，需要训练时另外保存模型快照与优化日志；当前页面会如实显示已有数据和可独立验证的教学更新。

## CPU / GPU 设备选择

神经算法、世界模型和 `play.py` 支持 `--device auto/cpu/cuda`，默认 `auto`：当前 Python 的 PyTorch 可以使用 CUDA 时选 GPU，否则选 CPU。显式选择 `cuda` 但不可用时会报清楚的错误，不会悄悄退回 CPU。Bellman、MC、TD、Q-learning、SARSA 是 NumPy 表格计算，始终运行在 CPU。

```powershell
python run.py ppo --steps 65536 --device cuda
python run.py sac --steps 65536 --device cuda
python run.py world-model --episodes 300 --steps 2000 --device cuda
python run_suite.py --world-model --device cuda
python play.py ppo --device cuda
python device_checks.py
```

训练开始会显示实际设备名称，`report.json` 保存请求设备、实际设备、显卡名称与 PyTorch 版本。当前已检测到 RTX 3090（24GB），并通过所有神经 learner 的 CPU/CUDA 前向与反向更新、GAE 和跨设备模型加载检查。环境逻辑和回放数据仍在 CPU；网络、训练批次、SAC 温度、GAE 临时张量与世界模型计算在选定设备。

默认 GPU 结果目录追加 `_cuda`，例如 `output/ppo_seed7_cuda`，避免覆盖先前的 CPU 教学结果；显式 `--output` 则使用指定目录。`play.py` 在 CUDA 上优先查找对应 `_cuda` 模型，没有时可加载原 CPU 模型。也可指定检查点实现跨设备回放：

```powershell
python play.py ppo --checkpoint output/ppo_seed7_cuda --device cpu
```

当前网络与网格环境都很小，GPU 不一定比 CPU 更快；设备选项提供正确切换，不代表已证明加速。后续图像/RSSM 世界模型和更大训练批次更能利用 GPU。

需要 Python 3.10+、NumPy、PyTorch；绘图另需 Matplotlib。推荐自己的虚拟环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

当前机器的全局 `dill` 安装不完整，已仅在项目 `.deps/` 补齐，入口会优先使用这个目录，不修改全局安装。若复制项目到新环境，正常安装 requirements 即可；不必依赖这个本地目录。

当前机器绘图库缺少的 `pyparsing` 和 `kiwisolver` 也已仅补齐到同一个本地目录。可以直接打开 `warehouse_course.ipynb` 逐格学习；Notebook 包含场景、手动轨迹、奖励计算、各方法和世界模型数据检查。

已有保存好的模型，可以先运行 `python play.py ppo` 看一条实际取货送达过程：每步会打印状态、动作概率、原始/训练奖励，以及 Critic 的 V、Bellman 目标与 delta。`--sample` 改成按神经 Actor 的分布采样，`--seed 456` 改变打滑序列；一次路线演示不能替代独立统计评估。

## 1. 场景是怎样定义的

```text
S..#..D
.#...#.
...X...
.#...#.
P......
```

- S：机器人起点；P：货物；D：交付点；X：危险区；#：墙壁。
- 动作：上、右、下、左。90% 按命令执行，另外 10% 平分给两个侧向动作，模拟打滑。
- 状态：`(行, 列, 是否载货)`。同一位置取货前后不算同一状态，否则到 D 时应否成功无法由状态判断，不满足这里需要的马尔可夫性。
- 每步原始奖励 -0.2；撞墙额外 -0.6；第一次取货额外 +3；载货到 D 额外 +20；进入 X 额外 -8 并真正终止。
- 取货后的每步状态都保持载货，不会反复获得取货奖励。没取货就到 D 不算送达。
- 120 步只截断采样与评估，不是 MDP 中的真正死亡；TD、DQN、GAE 保留该转移的价值自举，递推 GAE 时又会阻止跨重置回合传播。

为便于初学者先把算法跑通，默认所有方法使用相同的**势函数奖励塑形**：

```text
training_reward = base_reward + γ Φ(next_state) - Φ(state)
Φ = -0.5 × 剩余路线的曼哈顿距离估计
真正终止状态的 Φ 为 0；时间截断时不清零
```

取货前的距离是“当前位置到 P + P 到 D”，取货后是“当前位置到 D”。这是人为给定的任务知识，不是模型自行发现的路径；距离不考虑墙，仍需探索正确路线。对完整折扣任务，塑形使回报相差初始状态常数，因此保留最优策略。`checks.py` 通过 Bellman 参考解验证该性质。有限评估截断可能留下末端势函数项，所以评估始终另外记录原始任务回报。`--no-shaping` 关闭塑形，对比稀疏奖励困难：

```powershell
python run.py ppo --steps 65536 --no-shaping --output output/ppo_sparse_seed7
```

**评价结果看什么：**独立环境中运行 200 个回合，不更新策略，记录送达率、危险区失败率、平均步数和原始折扣回报。神经 Actor 默认用最大概率动作评估，不是从策略采样；这两种评估不能混作同一个指标。训练途中曲线使用另一固定环境种子 50 回合；最终 200 回合使用新种子。一次示例轨迹也会保存。

## 2. 推荐的学习次序

你提到的知识点不是十种同级算法：Bellman 是关系式，MC/TD 是评价方式，GAE 是优势估计方法。建议按以下顺序，理解每次究竟替换哪个模块。

|课次|运行命令|学习问题|对应代码|
|---|---|---|---|
|1 Bellman|`python run.py bellman`|已知转移规律，如何反复求解最优价值？|`tabular.value_iteration`|
|2 Monte Carlo|`python run.py mc`|不知道转移模型，能否用完整轨迹估计固定策略？|`tabular.prediction`|
|3 TD(0)|`python run.py td`|不等轨迹结束，怎样一步自举评价同一策略？|`tabular.prediction`|
|4 SARSA|`python run.py sarsa`|评价实际带探索的下一动作，怎样改进策略？|`tabular.control`|
|5 Q-learning|`python run.py q`|改成下一动作取最大 Q，变化在哪里？|`tabular.control`|
|6 DQN|`python run.py dqn --steps 65536`|用神经网络代替 Q 表，为什么加回放和目标网络？|`neural.dqn`|
|7 REINFORCE|`python run.py reinforce --episodes 2000`|不先学 Q，怎样直接调整动作概率？|`neural.reinforce`|
|8 Actor–Critic|`python run.py ac --steps 65536`|用一步 TD 信号辅助 Actor，Critic 怎样训练？|`neural.actor_critic`|
|9 GAE|`python run.py gae --steps 65536`|只把一步 δ 换成多步加权优势，会怎样？|同上，`categorical_gae`|
|10 PPO|`python run.py ppo --steps 65536`|同一批轨迹多轮训练时，怎样抑制策略变化过大？|同上，PPO 分支|
|11 离散 SAC|`python run.py sac --steps 65536`|如何结合双 Q、经验回放与最大熵探索？|`neural.sac`|
|12 世界模型桥梁|`python run.py world-model --episodes 300 --steps 2000`|学习动作后果，与直接学习价值有何不同？|`world_model.py`|

MC/TD 使用同一个固定的、带 10% 探索的参考策略。参考策略由已知模型求出，仅用于**策略评价实验**，不是训练 MC/TD 学会最优行为。二者同时对照精确 Bellman 期望解。其他 model-free learner 不调用环境转移分布或参考策略。世界模型示例的数据采集明确混合随机与参考行为，不声称无先验自主解决任务。

完整批量运行：

```powershell
python run_suite.py --world-model
```

每个神经算法在普通 CPU 上通常需几十秒至数分钟，取决于机器。预算较小时不保证学会任务。正式比较应更换多个随机种子，例如 `--seeds 7 21 42`，统一交互与计算预算；当前默认是教学验证而非论文基准。

## 3. 统一的算法流程

```text
环境：机器人在哪、是否载货、执行动作后发生了什么
    ↓
数据：独立转移样本 / 保留时间顺序的完整轨迹
    ↓
评价：完整回报 G / Bellman 一步目标 / GAE 优势
    ↓
学习：Q 表 / Q 网络 / Actor 与 Critic
    ↓
固定策略，独立评估，查看是否真正送达而非仅训练损失下降
```

详解见 [COURSE.md](COURSE.md)，世界模型扩展见 [WORLD_MODELS.md](WORLD_MODELS.md)。

## 4. 文件与结果

```text
run.py                   每一讲相同的入口
run_suite.py             批量运行，支持多种子
checks.py                环境、Bellman、GAE 和序列边界检验
summarize.py             汇总真实结果与学习曲线
warehouse/env.py         同一个 MDP；离散/特征/RGB 观测
warehouse/tabular.py     Bellman、MC、TD、SARSA、Q-learning
warehouse/neural.py      DQN、REINFORCE、AC、GAE、PPO、离散 SAC
warehouse/device.py      cpu/cuda/auto 选择与报告
warehouse/inspection.py  从检查点计算回放与诊断数据
warehouse/visualizer/    离线交互页面的模板、样式与脚本
visualize.py             生成并打开可视化页面
visual_checks.py         验证页面数据与公式是否一致
warehouse/replay.py      转移回放 + 不跨回合的序列采样
warehouse/world_model.py 概率下一状态、奖励、continuation 预测
output/<算法_种子>/      模型、学习曲线 CSV、报告、示例路线
output/RESULTS.md         已运行实验的结果
output/learning_curves.png 控制与价值预测分开画图
output/warehouse_route.png 同一个仓库任务，PPO 示例取货与送达路径
```

神经方法保存的 `episodes.json` 保留奖励、原始奖励、真终止和截断，用于后续世界模型数据集。`checks.py` 不只检查能否导入：它验证转移概率、取货状态、Bellman 残差、塑形最优性、GAE 手算值和轨迹不跨回合等实际风险。


## 交互式教科书：理论与实践一起学

打开 `output/dashboard.html`，选择「理论 → 代码」。17 课按 Bellman、MC、TD、SARSA/Q、DQN、Actor-Critic、GAE、PPO、连续 SAC、TD3、HER、BC、IQL、MPC、动力学随机化、记忆策略、框架衔接组织。

每课包含：实际问题、推导步骤、公式符号与代码变量对照、从当前源文件提取的代码及行号、可调数值实验、可复现命令、观察建议和自测。浏览器数值演算不等于重新训练；已保存策略的回放、教学演算和框架短跑分别注明。

连续控制实战在 `embodied/`，是带惯性和动作延迟的二维对接子技能。参见 [具身技术覆盖与实践](EMBODIED_TECHNIQUES.md)。

```powershell
python visualize.py
python run_embodied.py sac --steps 20000 --device auto
python run_embodied.py her --steps 20000 --device auto
python run_embodied.py sac --sparse --steps 20000 --device auto
python framework_doctor.py
python run_framework.py sac --env Reacher-v5 --steps 512 --device auto
```

先预测参数改变的结果，再操作滑块，最后运行训练命令核对。对照实验应固定奖励、预算、环境和种子；多种子才能支持稳定的算法比较。保存的单种子实验不能证明算法普遍优劣。稀疏 SAC 的失败、BC 数据覆盖不足的失败也保留供分析。

完整 Dreamer/RSSM、真实机器人部署、视觉策略、Isaac Lab/ManiSkill、ACT/Diffusion/VLA 等仍标为后续扩展，未冒充已经实现。当前世界模型包含离散一步预测和连续动力学模型加 MPC。
#   E m b o d i e d _ i n t e l l i g e n c e _ t e x t b o o k  
 