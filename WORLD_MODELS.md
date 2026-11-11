# 在同一任务上扩展世界模型与 Dreamer

## 已经可运行的桥梁

```powershell
python run.py world-model --episodes 300 --steps 2000
```

设备默认自动选择，也可加 `--device cuda` 使用 GPU，或 `--device cpu` 固定 CPU。模型、训练/验证张量及想象轨迹使用同一设备；完整回合数据和序列采样保留在 CPU。GPU 训练保存的模型可通过 `torch.load(..., map_location="cpu", weights_only=True)` 加载到 CPU；RGB/RSSM 扩展应遵循同样的设备传递约定。

这里 `--episodes` 是数据采集回合数，`--steps` 在这个实验中表示模型梯度更新次数，而非环境交互步数；其他神经控制实验的 `--steps` 表示环境步数。报告分别记录模型 updates 和训练转移数。世界模型数据采集一半回合随机行动，一半使用带 30% 探索的已知模型参考策略；这是可控的数据收集实验，不是宣称模型自主学会任务。

`StateWorldModel` 从状态与动作预测：

```text
pψ(next_state | state, action)  —— 70 类概率分布
rψ(state, action)              —— 奖励预测
cψ(state, action)              —— 下一步是否非真实终止的概率
```

因为本环境有打滑，下一状态本来就不是确定的。使用交叉熵学习概率，而不是强迫模型永远预测同一个状态。奖励头采用 MSE，continuation 头使用二分类交叉熵；时间截断不等于物理终止，因此 continuation 标签不因截断变为零。

完整回合按 80%/20% 划分训练与验证，再展开转移，避免相邻转移泄漏到两个集合。输出包括下一状态 NLL、reward RMSE、continuation 准确率，以及模型采样的一小段想象轨迹。预测质量和控制成功率是两件事：此处没有训练想象中的 Actor。

**这不是 Dreamer。**它使用 fully observed one-hot state，没有图像编码器、潜在序列状态、RSSM、KL 正则或想象中的 Actor–Critic。它的作用是先让你运行并检查“环境后果如何学习”这一环。

## 接口为什么能继续扩展

`env.observation(state, mode=...)` 支持：

- `onehot`：70 维，供当前控制算法。
- `features`：7 维坐标、载货状态与目标相对位置。
- `rgb`：40×56×3 图像，机器人载货与否用不同颜色显示，供未来视觉编码器。

当前地图静态且 RGB 可由 state ID 重建。若日后加入随机外观或部分可观测信息，必须在交互时存真实 RGB，不能再仅靠 state ID 事后重建。未来图像模型不能把 state ID 暗中作为输入，否则不是像素控制。

`EpisodeStore.sample_sequences(B,L)` 返回：

```text
states          [B,L+1]
actions         [B,L]
rewards         [B,L]
terminated      [B,L]
truncated       [B,L]
mask            [B,L]     padding 位置为 0
is_first        [B,L+1]   是否真实回合开始
```

采样不跨回合；尾部不足 L 时 padding 并给 mask。`is_first=False` 的片段不能假装从真实初始潜在状态开始：未来 RSSM 实现应采集前置上下文作 burn-in，或明确处理初始潜在状态误差。当前 padding 与回合标志只提供基础数据，不宣称已经解决潜在状态初始化。

## 建议分四次扩展

### A. 从状态模型开始做短期规划

在 `imagine()` 中给定候选动作序列，采样多个未来，用预测奖励、continuation 与可选末端 V 评价。先用短规划窗口，比较模型预测和真实轨迹。环境模型错误可能被规划器利用，单步准确不保证多步可靠。

### B. 加入视觉编码和潜在序列模型

把 RGB 编成 e_t。构造类似 RSSM 的确定性记忆 h_t 与随机状态 z_t：

```text
h_t = f(h_(t-1), z_(t-1), a_(t-1))
prior:     p(z_t | h_t)
posterior: q(z_t | h_t, e_t)
```

真实数据到来时 posterior 利用观察修正潜在状态；想象时没有未来观察，使用 prior。训练包含观测重建、奖励预测、continuation 预测，以及 posterior/prior 的 KL 项。不同 Dreamer 版本对随机状态、KL、梯度和归一化有不同处理，不能简单把上述符号称作完整 Dreamer 实现。

### C. 在模型中训练 Actor–Critic

从真实序列的 posterior 状态出发，让 Actor 产生动作，world model 推出潜在未来，reward head 与 continuation head 给出预测。用 lambda-return 训练想象 Critic，再优化 Actor。对于离散动作，需要明确选择 score-function 或合适的直通梯度估计器；不能假设 categorical sampling 自然可微。

这里会回到教程的核心链条：Bellman → TD/多步回报 → Actor–Critic → lambda-return。PPO 不是实现 Dreamer 的必经环节。

### D. 回到真实环境收集和检验

用新策略与真实环境交互，更新数据集，再训练世界模型。独立评估时只让策略根据允许的观测行动，不询问真实 `outcomes()`。对比原始任务回报与相同真实交互预算下的 PPO/DQN；模型生成的想象步不能伪装成真实环境采样。

## 可以沿原论文学习的路线

- Dreamer: https://arxiv.org/abs/1912.01603
- DreamerV2: https://arxiv.org/abs/2010.02193
- DreamerV3: https://arxiv.org/abs/2301.04104

当前项目先稳定环境、数据与评价接口。将来更换世界模型时可以保持任务不变，从而区分“预测表征的改进”与“换了一个更容易的任务”。
