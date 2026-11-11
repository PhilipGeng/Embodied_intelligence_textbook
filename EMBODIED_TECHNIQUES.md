# 具身技术地图：从交互教材到真实机器人框架

本项目的主线仍是“理论 → 公式 → 变量 → 实际函数 → 数据 → 改一个条件后的结果”。下列状态分别表示：原理教学实现、真实第三方接口已验证、或仅提供后续接入路线。它们不是同一个完成度。

## 1. 从原有仓库任务延伸出的可运行实战

原来的 5×7 仓库保留不变。新增**连续装卸对接子技能**：机器人要控制二维力，让有惯性和阻尼的载具到达目标并停稳。观测为位置、速度、目标，动作是两个连续力。它是轻量动力学教学环境，不是机械臂/人形机器人仿真器。

|主题|新增内容|理论如何落到实践|状态|
|---|---|---|---|
|连续 SAC|Gaussian Actor + 双 Q + tanh 修正 + 可学习熵温度|soft Bellman 目标；梯度经过采样动作和 Q|可训练与回放|
|TD3|确定性 Actor、双 Q、目标动作噪声、延迟更新|对 critic 误差与 Actor 追逐噪声的三项修正|可训练与回放|
|HER|同回合 future goal，80% 重标记，重算稀疏奖励|改变目标和奖励，不改变真实转移|可训练，提供无 HER 对照命令|
|BC 模仿学习|程序 PD 演示 + 噪声 + 随机低质量数据|把演示中的动作作为监督目标|可训练，提供目标覆盖不足的消融|
|IQL 离线 RL|expectile V、双 Q、自举与优势加权模仿|V 偏向较好数据动作；Q 目标不查询 Actor 新动作|可训练，同数据可与 BC 对比|
|模型学习 + MPC|学习线性 delta 动力学，CEM 搜索恒定力候选，逐步重规划|预测模型提供未来状态，规划累加成本并只执行首个动作|可运行；模型按完整回合留出验证|
|动力学随机化|质量、阻尼、指令延迟随机；同名义域和偏移域评估|优化一组环境上的表现，检验参数变化影响|可训练，是 sim-to-sim 教学，不是已验证真机迁移|
|POMDP 与循环策略|隐藏速度的观测 + SB3-Contrib Recurrent PPO|当前观测不足时，用 LSTM 状态整合历史|第三方训练流程可验证|
|物理仿真基准|MuJoCo `Reacher-v5` + SB3 SAC|相同策略/价值优化用于真正的关节动力学环境|真实第三方训练接口短跑验证|

`DockingEnv` 的动力学与代码一一对应：

```text
v_next = clip(v + dt × (applied_force / mass - drag × v))
p_next = clip(p + dt × v_next)
obs = [p_x, p_y, v_x, v_y, goal_x, goal_y]
```

动作按需要经过延迟队列；边界碰撞时相应速度清零。名义质量 1、阻尼 .35、零延迟；偏移域质量 1.6、阻尼 .7、延迟 2 步。随机训练质量 [.65,1.45]、阻尼 [.15,.65]、延迟 0 或 1。偏移域有意超出训练范围，不能保证随机化策略一定更好。

环境每回合采 60 步，成功不提前终止，鼓励到达后保持；最终位置距离目标小于 .12 定义 final success。另记 ever reached，区分“一度碰到目标”和“最后停稳”。这些指标不能与离散仓库送达率直接排名。

## 2. 命令与一个条件的对照实验

```powershell
# 连续控制：不要与 run.py sac 的离散 SAC 混淆
python run_embodied.py sac --steps 20000 --device cuda
python run_embodied.py td3 --steps 20000 --device cuda

# 稀疏奖励：同一奖励设置下比较是否重标记
python run_embodied.py sac --sparse --steps 20000 --device cuda
python run_embodied.py her --steps 20000 --device cuda

# 演示与离线 RL，IQL 复用同一份数据
python run_embodied.py bc --updates 2000 --episodes 100 --device cuda
python run_embodied.py iql --dataset output/embodied/bc_seed7_cuda/dataset.npz --updates 2000 --device cuda

# 明确的数据覆盖消融：只给随机目标，不强制覆盖部署目标
python run_embodied.py bc --random-goals-only --updates 2000 --device cuda --output output/embodied/bc_goal_coverage_gap

# 先学模型再规划；以及不学习的 PD 控制参考
python run_embodied.py mpc --episodes 100
python run_embodied.py pd

# 环境随机化与名义训练对照，最终都测试名义/偏移两个域
python run_embodied.py sac --randomize --steps 20000 --device cuda
```

报告、每步轨迹与模型保存在 `output/embodied/`。离线训练阶段没有新的环境交互，但生成教学数据本身仍有交互成本，报告单独记 dataset transitions。演示由程序控制器生成，不是人类遥操作；25% 回合固定部署目标、其他随机目标，20% 回合使用低质量随机行为。数据覆盖的消融会保留，不用调参掩盖失败。

## 3. 主流框架分层，避免把算法与仿真器混为一谈

|层次|框架|在教材里的角色|本机处理|
|---|---|---|---|
|环境接口|[Gymnasium](https://gymnasium.farama.org/introduction/create_custom_env/)|统一 reset/step、空间定义、真终止与截断|提供仓库、连续对接、GoalEnv 三个适配器并通过检查|
|通用训练|[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)|与手写算法交叉验证 PPO/SAC/TD3/HER|实际调用第三方模型训练，不是同名包装|
|记忆策略|[SB3-Contrib](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)|Recurrent PPO；部分可观测对照|提供 LSTM 训练入口|
|CPU 关节物理|[MuJoCo](https://mujoco.readthedocs.io/en/stable/python.html)|从点质量教学任务进入关节、力矩和接触|已接入 Gymnasium MuJoCo 任务；学习可用 GPU，物理并非 GPU 并行|
|GPU 仿真/腿式/操作|[Isaac Lab](https://docs.nvidia.com/isaaclab/concepts/reinforcement-learning)|大规模并行、奖励管理、域随机化、教师学生等|提供正式官方入口和安装状态，不自动安装整套 Isaac Sim|
|GPU 操作仿真|[ManiSkill](https://maniskill.readthedocs.io/en/latest/)|机械臂操作、演示、视觉及 GPU 仿真|官方接口路线；未宣称在本机完成安装训练|
|机器人 RL 训练库|[RSL-RL](https://github.com/leggedrobotics/rsl_rl)|常见机器人策略学习组件，与仿真器协同|官方接入路线，不与 SB3 当作同一个 API|
|数据/真机/模仿/VLA|[LeRobot](https://huggingface.co/docs/lerobot/en/index)|演示数据、策略训练与部署流程|官方接入路线；未把本项目 npz 冒充 LeRobotDataset|
|潜在世界模型|[DreamerV3](https://github.com/danijar/dreamerv3)|RSSM、预测学习、想象 Actor–Critic|已有序列与模型桥梁，尚未把线性模型称为 Dreamer|

查看实际导入与保存的短跑结果：

```powershell
python framework_doctor.py
python run_framework.py ppo --env warehouse --steps 1024 --device cpu
python run_framework.py her-sac --env reach-sparse --steps 512 --device cuda
python run_framework.py recurrent-ppo --env reach --partial --steps 512 --device cpu
python run_framework.py sac --env Reacher-v5 --steps 512 --device cuda
```

这些几百/一千步的短跑只验证 API、回放、优化和保存流程，不代表完成任务。MuJoCo Reacher 的回报也不等于仓库成功率。默认可视化会把这些框架验证记录单独列出。

## 4. 仍需后续专章的能力（明确不伪装成已完成）

- **真机 sim-to-real、系统辨识与在线适应**：当前随机化仅在简化仿真里验证；需真实机器人传感器/控制频率/执行器接口与校准数据。
- **视觉与部分可观测的端到端学习**：已有 RGB 和循环策略入口；还需 CNN/RSSM 图像编码、视觉任务评估与遮挡场景。
- **教师学生、特权信息、AMP/运动先验、腿式与人形控制**：需关节物理仿真、参考运动和任务奖励，不能由网格动作演示替代。
- **动作分块、Diffusion Policy、ACT、VLA 与人类干预 RL**：BC/IQL 先说明数据与价值的关系，复杂动作生成和真机干预走 LeRobot 等对应路线。
- **安全约束 RL/CBF、约束策略优化、多机器人 MARL**：当前动作限幅和边界统计不等于安全保证，单机器人环境也不能直接宣称支持多智能体信用分配。
- **完整 CQL、Dreamer/RSSM、TD-MPC 等高级实现**：IQL 与模型规划先搭基础，后续按原始论文逐章补齐并做独立测试。

理论与实践的边界应始终可见：成功的教学样例说明机制可工作，不证明它已解决真实机器人部署问题；失败的例子也应作为可复现的学习材料。
