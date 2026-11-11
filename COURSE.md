# 从一条取货送达轨迹，贯穿全部算法

本教程奖励一律记作 `r_t`：执行 `a_t` 后立即收到。状态 `s=(row,col,carrying)`，动作四选一，折扣 γ=0.97。先看 README 的地图和规则，再逐课运行。下面公式里的 `1-d` 只屏蔽**真实终止**，不屏蔽时间截断。

## 第 1 课：Bellman 方程先把未来拆成一步

机器人在 `(0,0,0)`，目标不是直接去 D，而是先去 P。一次向下的动作即时奖励可能很小，但其价值来自之后取货和送达。

```text
G_t = r_t + γ G_(t+1)
Vπ(s) = Σ_a π(a|s) Σ_s' P(s'|s,a) [r + γ(1-d)Vπ(s')]
Q*(s,a) = Σ_s' P(s'|s,a) [r + γ(1-d)max_a' Q*(s',a')]
```

`outcomes(s,a)` 返回本案例的真实分布：意图方向概率 0.9，两个侧向各 0.05。`value_iteration` 遍历状态和动作，按这个分布计算期望，直到价值变化足够小。它得到已知模型下的参考解；这不是让机器人通过探索学习，而是帮助后续算法验证结果。

**运行：**`python run.py bellman`。打开 `output/bellman_seed7/q.npy`。观察相同位置载货前后动作偏好是否不同。即使最优策略也可能因打滑失败，因此一次评估成功率不必为 100%。不同策略有限次实测略高于参考策略，也不证明期望回报更高。

## 第 2 课：Monte Carlo 用完整经历估计固定策略

先固定一个策略，不改变 Actor，问“它在起点的价值是多少”。我们使用带 10% 随机探索的参考策略，既保证可控的任务难度，也能用 Bellman 期望方程算出精确 V 作为对照。

沿一条完整轨迹，倒推 `G = r + γG`，给每个状态的首次访问记一个回报样本：

```text
count[s] += 1
V[s] += (G - V[s]) / count[s]
```

这就是样本平均。无需知道转移概率，也不用下一状态的 V 作为目标。但要等真正终止，且最后的打滑、补救和成功会影响早期回报。

**运行：**`python run.py mc --episodes 3000`。报告中的 `start_exact` 是固定策略的精确期望，`start_estimate` 是真实采样回报的平均。两者会有有限样本误差。实现设置很高的完整回合收集上限；若仍超时会报错，不会把未完成轨迹伪装成终止回报为零，也不通过丢掉长失败轨迹人为抬高估值。

## 第 3 课：TD 用下一状态的预测提前学习

策略与第 2 课完全相同，只改价值估计：

```text
target = r + γ(1-d)V[next_state]
delta = target - V[state]
V[state] += α delta
```

执行动作后，“即时奖励 + 折扣后的下一状态价值”相对原预期提升，delta 为正；降低则为负。这个学习不必等送达。但 Critic 自己预测不准时，目标也受影响。

**运行：**`python run.py td --episodes 3000`。和 MC 比较起点估值误差随交互数如何变化。这里 TD 使用固定学习率 0.05，MC 用样本均值，两者不会因回合数相同就具有同样收敛性质。`visited_state_rmse` 只对本次累计访问至少 20 次的状态计算，其状态集合随时间变化，不宜当成固定测试集误差；更直接的共同对照是起点估值误差。

MC 与 TD 在本课都只**评价**固定策略，不能把它们相同的控制成功率理解为训练效果。控制成功率属于那套固定策略，不属于学到的 V。

## 第 4 课：SARSA 把评价与动作选择接起来

开始训练 Q 表，并用 ε-greedy 选动作。SARSA 的下一动作也是由实际行为策略抽取：

```text
a_next ~ epsilon_greedy(Q[next_state])
target = r + γ(1-d)Q[next_state, a_next]
```

名字来自 `(State, Action, Reward, State, Action)`。如果下一状态靠近危险区，随机探索可能把机器人带进去，这种风险就能进入 SARSA 的评价。

**运行：**`python run.py sarsa --episodes 3000`。程序最终评价时关闭探索，使用贪心策略；训练时的探索风险和最终贪心策略风险要区分。地图和训练预算不同，SARSA 不保证总比 Q-learning 更安全。

## 第 5 课：Q-learning 只替换下一动作评价

同一个训练循环中，改成：

```text
target = r + γ(1-d)max(Q[next_state])
```

不管行为下一步随机做什么，目标都假设采取当前估计最好的下一动作。这就是本案例里 on-policy SARSA 与 off-policy Q-learning 最直观的区别。

**运行：**`python run.py q --episodes 3000`。比较与 SARSA 的线路和危险率。二者共用 `control`，仔细看 `next_value` 那一行即可找到核心差异。有限数据的 Q 表不必各项都精确，行为也可能已经很好。

## 第 6 课：DQN 将 Q 表换成网络

输入为 70 维 one-hot 状态（包含载货阶段），网络结构为 `Linear(70,64) → Tanh → Linear(64,4)`。输出四个动作的 Q 值。这里故意先不加像素编码器，避免视觉表征与 RL 稳定性混在一起。

```text
存储真实转移到 Replay
随机抽取一个 batch
target = r + γ(1-d)max(Q_target(next_state))
prediction = Q_online(state)[actual_action]
用 Huber loss 训练 Q_online
每 500 步将 online 参数复制到 target
```

Replay 使旧经验可以复用，打散连续状态的相关性；目标网络让 Bellman 目标变化较慢。本例是基础 DQN，不暗中使用 Double DQN、优先回放或其他改进。网络每 4 个环境步训练一次，warmup 为 512 步。

**运行：**`python run.py dqn --steps 65536`。查看它是否把局部错误 Q 估值变成危险路线。DQN 有神经网络不代表一定比这个小场景的 Q 表更好。

## 第 7 课：REINFORCE 直接训练策略

输入相同状态，Actor 输出四个动作的 logits，再按分类分布采样。整条轨迹结束后，使用每步之后的回报更新：

```text
actor_loss = -Σ_t γ^t log πθ(a_t|s_t) × (G_t - b)
```

`b` 是前几个 batch 回报形成的移动平均常数，不是神经 Critic；在当前 batch 更新期间固定。代码每 8 个完整回合更新一次，按回合数取平均；为数值尺度再除以固定的 10，不改变梯度方向。没有额外熵奖励，因此这一课能单独看到纯轨迹回报的波动。

**运行：**`python run.py reinforce --episodes 2000`。原始从初始状态折扣的目标中有外部 `γ^t`；不要把它和 reward-to-go 内部的折扣混淆。大幅增减训练回合，对比梯度估计噪声带来的差异。

## 第 8 课：Actor–Critic 用一步 TD 评价动作

Actor 输出动作分布，Critic 单独输出 V(s)。使用 16 个环境各采 32 步，固定采样时的旧 V：

```text
delta = r + γ(1-terminated)V_old(next_state) - V_old(state)
advantage = delta
return_target = advantage + V_old(state)
```

训练 Actor 时 `advantage` 固定；训练 Critic 时预测端有梯度，目标端固定。不要把已经 `detach` 的 delta 平方当作唯一 Critic loss，那会让 Critic 收不到梯度。

**运行：**`python run.py ac --steps 65536`。这是一种同步批量的一步 Actor–Critic，非逐步在线版本。每批数据只训练一轮；小批量中的样本平均不是对状态访问概率做精确积分。实现采用常用的均匀 rollout 状态采样、优势标准化与 0.01 熵奖励，不是第 7 课有限回合精确折扣梯度的逐字实现。这些共同设置在 AC/GAE/PPO 三课保持一致。

## 第 9 课：GAE 替换优势估计，而不新增网络

Actor、Critic、采样和训练循环都沿用第 8 课，只将 lambda 从 0 换成 0.95：

```text
delta[t] = r[t] + γ(1-terminated[t])V_old(next_state[t]) - V_old(state[t])
A[t] = delta[t] + γλ(1-boundary[t])A[t+1]
```

`terminated` 屏蔽价值自举；`boundary=terminated or truncated` 阻止倒推跨到重置后的下一回合。两者不能简单写成同一个 mask。

例如三步奖励 `[0,0,10]`、旧 V `[6,7,8]`，γ=.9、λ=.95，结果应为 `[1.93305,1.91,2]`。`checks.py` 会验证这个手算值以及超时边界。

**运行：**`python run.py gae --steps 65536`。与 AC 对比，评价信号从只看一步变化为吸收未来多步的变化。GAE 没有解决策略更新幅度或整个信用分配难题。

## 第 10 课：PPO 允许同批轨迹多轮训练

继续使用同一套 Actor、Critic 和 GAE。保存采样时的动作对数概率，训练时计算：

```text
ratio = exp(new_log_prob - old_log_prob)
actor_loss = -mean(min(ratio × A, clip(ratio,.8,1.2) × A))
```

每批数据训练 4 轮，每个小批量 128 条样本。Critic 仍预测固定 return target。裁剪让过大的有利概率变化不再从这条样本获得额外收益，不是动作概率硬上限，也不是安全保证。本例没有 KL early-stop，适合学习最小 PPO-Clip；若参数更新不稳，后续实验可以加入 KL 监控。

**运行：**`python run.py ppo --steps 65536`。`actor_critic` 函数中 AC、GAE、PPO 共用大量实现，便于看清真正变化：优势估计以及 Actor 的目标和更新轮数。

## 第 11 课：离散 SAC，把探索写进 Actor 目标

本任务是四个离散动作，所以实现的是 **discrete SAC**。Actor 输出四个动作的概率，两个 Critic 各输出四个 Q，使用 Replay 和缓慢更新的目标 Critic。

```text
V_soft(s') = Σ_a' π(a'|s') [min(Q1_target,Q2_target)(s',a') - α log π(a'|s')]
target = r + γ(1-d)V_soft(s')
critic_loss = mean((Q1(s,a)-target)² + (Q2(s,a)-target)²)

actor_loss = mean_s Σ_a π(a|s) [α log π(a|s) - min(Q1,Q2)(s,a)]
```

下一状态的未来价值不是取最大，而是按当前策略对四个动作求精确期望，另计策略熵。更新 Actor 时固定 Q 的参数与数值，但 π 和 logπ 保留梯度。离散动作少，直接求和就可以；这里没有使用连续版 SAC 的 tanh-Gaussian 或动作重参数化。

温度使用 `α=exp(log_alpha)` 保持正数。目标熵取 `0.5×log(4)`：实际熵高于目标时降低 α，低于目标时提高 α。代码使用 `alpha_loss=log_alpha×(entropy.detach()-target_entropy)` 做梯度下降。这个目标熵只是本例的教学设置，不是通用最优值。

**运行：**`python run.py sac --steps 65536`。对比 PPO：SAC 重用旧转移并持续更新，不需要旧策略概率比率或 GAE；其 Actor 根据 Q 与熵优化，Critic 用 soft Bellman 目标学习。每 4 环境步抽样更新一次，前 512 步随机探索。训练时采样动作，最终统一用最大概率动作评估。`learning.csv` 同时记录温度，观察策略探索强度怎样变化。

连续机械臂控制可以沿 SAC 思想继续扩展，但必须改 Actor 分布、动作概率密度计算及重参数化，不能仅把输出层从 4 改成关节数。

## 每一课都要做的四项检查

1. 对一条实际转移，算出即时奖励、Bellman 目标和误差。对塑形奖励还要同时看 base reward，避免把辅助信号当成任务成功。
2. 冻结策略后独立跑任务，看送达率、风险与原始折扣回报；训练 loss 降低不是充分证据。
3. 改种子或预算，观察结论是否稳定；一次 200 回合评估仍有抽样误差。
4. 修改一项机制再对比，别同时改奖励、网络、动作和优化器，否则难以解释差异。

下一步参见 `WORLD_MODELS.md`：同一个任务和轨迹数据将支持环境模型、潜在状态、想象轨迹与 Dreamer 式训练。


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
