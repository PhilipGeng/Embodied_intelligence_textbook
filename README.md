# 具身强化学习实验室

**从 Bellman 方程到连续控制与世界模型，让每一个公式都能在代码和实验中找到对应。**

这是一本可以运行、观察和修改的强化学习交互式教材。你将从仓库机器人“取货 → 送达”的任务出发，学习价值估计与策略优化，再进入有惯性、阻尼和动作延迟的连续对接任务，理解现代具身学习中的探索、模仿、离线学习与模型规划。

每个主题都围绕同一条学习链展开：

> 遇到了什么问题 → 为什么需要这个方法 → 数学关系如何推导 → 公式对应哪些变量和代码 → 改一个条件会发生什么 → 用实验检验解释。

适合已经接触过基本 Python，希望真正理解强化学习算法如何工作的读者。建议具备概率、期望、导数和梯度的基础知识；阅读时可以先运行案例，再回到公式。

## 先打开教材

在项目文件夹中双击 **[打开交互教材.html](打开交互教材.html)**，或直接打开 [output/dashboard.html](output/dashboard.html)。已有导出页面可以离线使用，无需安装 Python、启动服务或联网。

页面有六个学习区域：

|区域|可以做什么|重点理解|
|---|---|---|
|任务回放|逐步观察位置、载货状态、动作概率或 Q 值、奖励和价值|策略如何把局部动作组成完整任务|
|Bellman 更新|沿教学轨迹观察旧 Q、目标、TD 误差和更新后的 Q|未来奖励如何逐步影响早期决策|
|训练曲线|比较送达率、风险、原始回报与价值预测误差|损失下降与任务完成之间有什么区别|
|世界模型|对照预测的下一状态分布、奖励与实际结果|预测环境与评价动作各解决什么问题|
|理论 → 代码|阅读推导、点击公式变量定位源码、操作参数实验并自测|数学设计怎样变成实际程序|
|具身进阶|回放连续控制轨迹，比较名义动力学与参数偏移|惯性、探索、数据覆盖与模型误差怎样影响控制|

**浏览器负责阅读、演算和回放；新的训练通过 Python 命令启动。** 页面不会在拖动滑块时重新训练神经网络。完成训练后，运行 `python visualize.py --open`，即可把新结果导出到页面。

请打开上述教材入口。`warehouse/visualizer/dashboard.html` 是导出模板，`tmp/` 下的文件是开发过程文件。

## 安装与第一次实验

仅阅读已有教材时可以跳过安装。运行或修改实验需要 Python 3.10+。

以下命令在项目根目录执行，示例使用 Windows PowerShell：

```powershell
cd "$env:USERPROFILE\Desktop\具身强化学习实验室"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 检查任务定义与关键数学关系
python checks.py

# 已知环境模型：求 Bellman 参考解
python run.py bellman

# 不调用转移模型：从交互样本学习 Q
python run.py q --episodes 3000

# 将已有结果导出到交互教材
python visualize.py --open
```

若 PowerShell 无法激活环境，可直接用 `.venv\Scripts\python.exe` 替代 `python`。

本机项目包含用于补齐依赖的 `.deps/`，入口会优先读取它。该目录与机器环境有关，迁移到其他电脑时应在自己的虚拟环境中安装依赖。

### GPU

神经网络训练支持 `--device auto`、`cpu`、`cuda`，默认 `auto`。选择 `cuda` 但当前 PyTorch 不支持 CUDA 时会报错；需要先安装适合本机驱动的 CUDA 版 PyTorch。

```powershell
python run.py ppo --steps 65536 --device cuda
python run_embodied.py sac --steps 20000 --device cuda
python device_checks.py
```

本机已验证 RTX 3090 上的训练流程。表格算法与环境逻辑运行在 CPU；网络训练可以使用 GPU。小型教学网络不一定因 GPU 而更快，GPU 训练也不表示物理仿真运行在 GPU。

## 两个相互衔接的任务

### 1. 离散仓库：先看清强化学习的共同结构

```text
S..#..D
.#...#.
...X...
.#...#.
P......
```

机器人从 S 出发，在 P 取货后到 D 送达；`#` 是墙，X 是危险区。动作是上、右、下、左，执行时有 10% 的侧向打滑概率。

状态包含 `(行, 列, 是否载货)`。同一个位置在取货前后需要做出不同决策，因此“是否载货”必须进入状态。

原始奖励：每步 −0.2，撞墙额外 −0.6，首次取货额外 +3，送达额外 +20，进入危险区额外 −8 并终止。默认还使用势函数奖励塑形，帮助初学者跑通探索；报告另记原始任务回报。使用 `--no-shaping` 可关闭塑形，它不会删除原始的取货和送达奖励。

120 步上限属于时间截断，不等于任务真正终止。代码在自举和轨迹递推中分别处理这两个边界，详见 [课程讲解](COURSE.md)。

### 2. 连续对接：让动作面对惯性与动力学

机器人控制二维连续力，使载具靠近目标并保持。观测包含位置、速度与目标；环境包含质量、阻尼、边界碰撞与可选动作延迟。

这个轻量二维动力学环境用于学习连续控制。它尚不包含机械臂关节、复杂接触或真机硬件。60 步回合末，位置距离目标小于 0.12 记为最终成功；同时记录“曾经到达”，帮助区分途中碰到目标与最终保持在目标附近。最终成功指标没有速度阈值。

两个任务共享价值估计、策略改进和独立评估的思想，但状态、动作、奖励和预算不同，不能把成功率放在一起给算法排名。

## 推荐学习路线

建议把下面各阶段与页面的“理论 → 代码”配合使用。页面当前有 22 个独立章节，表中还列出 REINFORCE、离散 SAC 和世界模型等配套实战。

|阶段|学习内容|要回答的问题|入口|
|---|---|---|---|
|1 价值从哪里来|Bellman、Monte Carlo、TD|完整回报、条件期望和一步自举是什么关系？|`warehouse/tabular.py`|
|2 从评价到控制|SARSA、Q-learning、DQN|下一动作由谁选？为什么需要回放与目标网络？|`warehouse/tabular.py`、`warehouse/neural.py`|
|3 直接学习策略|REINFORCE、Actor–Critic|动作概率如何根据回报或 TD 信号改变？|`warehouse/neural.py`|
|4 更可靠地更新|GAE、PPO|优势如何估计？旧数据复用时怎样约束策略变化？|`warehouse/neural.py`|
|5 连续动作与探索|离散/连续 SAC、TD3、HER|熵、双 Q、延迟更新与目标重标记分别改变什么？|`warehouse/neural.py`、`embodied/learners.py`|
|6 从已有数据学习|BC、IQL|模仿损失与任务成功有什么关系？如何利用较好的数据动作？|`embodied/offline.py`|
|7 学习环境并规划|一步世界模型、模型学习 + MPC|预测模型怎样帮助选择动作？误差如何影响规划？|`warehouse/world_model.py`、`embodied/planning.py`|
|8 走向具身框架|动力学随机化、POMDP、循环策略、Gymnasium/SB3/MuJoCo|环境变化、记忆与仿真接口分别解决什么问题？|`embodied/frameworks.py`、`run_framework.py`|

每学一章，建议完成四步：**先预测结果 → 手动算一次 → 运行代码 → 只改一个条件做对照**。例如，先判断增大 gamma 会怎样影响早期状态价值，再操作页面滑块，最后修改实验配置检查实际训练表现。

## 实战命令

### 价值、策略梯度与 PPO

```powershell
python run.py mc --episodes 3000
python run.py td --episodes 3000
python run.py sarsa --episodes 3000
python run.py dqn --steps 65536
python run.py reinforce --episodes 2000
python run.py ac --steps 65536
python run.py gae --steps 65536
python run.py ppo --steps 65536
python run.py sac --steps 65536

# 查看已保存策略的逐步决策；--sample 改为按策略分布采样
python play.py ppo
python play.py ppo --sample
```

MC/TD 在这里评价同一套固定策略，不负责改进 Actor；其参考策略来自已知模型，专用于评价实验。其他 model-free 控制算法不调用真实转移分布。`run.py sac` 是离散动作 SAC。

### 连续控制、HER 与数据覆盖

```powershell
python run_embodied.py sac --steps 20000
python run_embodied.py td3 --steps 20000

# 相同稀疏奖励与交互预算：比较是否使用 HER
python run_embodied.py sac --sparse --steps 20000
python run_embodied.py her --steps 20000

# 演示覆盖与泛化；BC 数据由程序控制器生成
python run_embodied.py bc --updates 2000 --episodes 100
python run_embodied.py bc --random-goals-only --updates 2000 --output output/embodied/bc_coverage_gap

# 训练动力学随机化策略，再在名义域和偏移域评估
python run_embodied.py sac --randomize --steps 20000
```

BC 与 IQL 的公平比较应复用同一份数据。下面明确固定 CPU，使输出路径不随自动设备选择改变：

```powershell
python run_embodied.py bc --device cpu --output output/embodied/bc_shared
python run_embodied.py iql --device cpu --dataset output/embodied/bc_shared/dataset.npz
```

### 世界模型与规划

```powershell
# 离散任务：学习下一状态、奖励和 continuation
python run.py world-model --episodes 300 --steps 2000

# 连续任务：学习动力学，再用 MPC 重规划
python run_embodied.py mpc --episodes 100

# 无学习的手工 PD 控制器参考
python run_embodied.py pd
```

连续 MPC 使用学到的线性状态增量模型与 CEM 候选搜索。这是理解模型学习和规划的桥梁；完整 Dreamer 的潜在状态、RSSM 和想象训练尚未实现。

### 接入第三方实现与物理仿真

可选依赖单独安装，不影响基础教材：

```powershell
python -m pip install -r requirements-frameworks.txt
python framework_doctor.py
python run_framework.py ppo --env warehouse --steps 1024
python run_framework.py her-sac --env reach-sparse --steps 512
python run_framework.py recurrent-ppo --env reach --partial --steps 512
python run_framework.py sac --env Reacher-v5 --steps 512
```

这些短预算命令用于验证环境接口、优化步骤和模型保存，不保证策略收敛。Gymnasium、SB3、Recurrent PPO 与 MuJoCo 接口已有短跑验证；Isaac Lab、ManiSkill、RSL-RL、LeRobot 等提供后续接入路线，尚未集成运行。完整状态见 [具身技术地图](EMBODIED_TECHNIQUES.md)。

## 怎样解释结果

教材明确区分以下数据来源：

|来源|能说明什么|
|---|---|
|真实训练 CSV 与独立评估报告|该配置、种子和预算下的学习过程与任务表现|
|冻结检查点产生的新轨迹|当前策略怎样行动，以及重算后的价值/TD/GAE 诊断|
|手选数值与参数滑块|某个公式怎样计算、参数怎样改变局部数值|
|第三方框架短跑|接口和训练流程能否工作|

冻结策略上重算的优势不是历史训练时的优势；浏览器公式演算也不是训练日志。学习损失下降不能单独证明策略变好，应同时检查成功率、原始回报、风险与轨迹。

项目保留稀疏 SAC、BC 目标覆盖不足等失败记录，用于分析探索和数据分布。现有保存结果主要是单种子教学实验。算法比较需要统一任务、奖励、数据、预算与评估方式，并使用多个种子；动力学偏移实验属于简化仿真对照，尚不能证明真机迁移能力。

## 项目结构与输出

```text
打开交互教材.html          双击进入完整离线教材
visualize.py               将模型、数据与课程导出为 HTML
textbook/content.py        课程内容、公式与真实源码映射
warehouse/
  env.py                   离散仓库任务
  tabular.py               Bellman、MC/TD、SARSA、Q-learning
  neural.py                DQN、REINFORCE、AC、GAE、PPO、离散 SAC
  replay.py                转移与不跨回合的序列采样
  world_model.py           一步概率世界模型
  inspection.py            冻结模型回放与诊断数据
  visualizer/              页面模板、样式与交互脚本
embodied/
  env.py                   连续对接动力学
  networks.py              连续 Actor、双 Q 与概率计算
  learners.py              SAC、TD3、HER
  replay.py                目标条件回放与 HER 重标记
  offline.py               数据生成、BC 与 IQL
  planning.py              动力学学习与 MPC
  frameworks.py            Gymnasium 环境适配
run.py                     离散算法统一入口
run_embodied.py            连续控制、离线学习与 MPC 入口
run_framework.py           第三方算法与环境入口
framework_doctor.py        本机依赖和框架验证状态
warehouse_course.ipynb     可逐格运行的学习笔记
output/
  dashboard.html           已导出的完整交互教材
  <算法_种子>/             离散实验报告、曲线和检查点
  embodied/<实验>/         连续实验、数据、模型与逐步轨迹
  frameworks/<实验>/       第三方短跑报告与模型
  framework_status.json    本机框架检查结果
```

训练入口默认写入由算法、种子及设备等组成的目录；同配置重复运行可能覆盖旧结果。保留对照实验时请显式指定不同的 `--output`。CUDA 训练的默认目录通常带 `_cuda` 后缀。

## 验证与排查

```powershell
python checks.py
python embodied_checks.py
python visual_checks.py
python device_checks.py
```

检查覆盖 Bellman 残差、奖励塑形、GAE 边界、终止与截断、连续动力学、tanh 概率修正、HER 同回合重标记及页面数据一致性。

|遇到的问题|处理方式|
|---|---|
|页面没有样式，或下拉框为空|打开项目根目录的“打开交互教材.html”；必要时运行 `python visualize.py --open` 重新导出|
|训练后页面还是旧结果|重新导出页面，再刷新浏览器；页面是结果快照|
|CUDA 不可用|先使用 `--device cpu`，再检查当前 Python 的 PyTorch/CUDA 安装|
|第三方框架导入失败|安装 `requirements-frameworks.txt`，运行 `python framework_doctor.py`|
|预算跑完仍未成功|查看轨迹、奖励、探索和数据覆盖；短跑不保证收敛|

## 继续阅读

- [COURSE.md](COURSE.md)：基础算法的章节联系与详细讲解。
- [EMBODIED_TECHNIQUES.md](EMBODIED_TECHNIQUES.md)：连续控制、离线学习、动力学随机化与主流框架覆盖。
- [WORLD_MODELS.md](WORLD_MODELS.md)：从一步预测模型走向世界模型学习的路线。
- [Sutton & Barto：Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)：强化学习基础教材。

后续扩展包括完整 Dreamer/RSSM、视觉策略、复杂机器人仿真、ACT/Diffusion/VLA、安全约束与真机部署。当前教材会持续沿用“推导 → 实现 → 实验 → 解释”的组织方式，把新增技术接回共同的强化学习基础。
