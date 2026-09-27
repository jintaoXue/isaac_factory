# R0 的论文贡献：执行一致的分层决策与半马尔可夫经验回放

本文对应当前实现 `source/algo/hierarchical/hc_factory/decision_consistent.py`，说明 G0 与 R0 的动作语义、决策时钟和价值更新。实验入口见 [R 系列实验说明](experiment_r_series.md)，问题依据见 [诊断归档](rl_diagnosis_2026-09-27.md)。公式采用 Markdown 中的 LaTeX 数学语法，可移入论文。

## 1. 框架是否仍是 A/B/C/D？

是，四层职责保持不变：

| 层 | 业务职责 | R0 的动作 |
|---|---|---|
| A | 产品释放 / product sequencing | 选择 next product 的产品类型，由环境解析为具体待释放批次并写入 staging |
| B | 产品服务优先级 / product prioritization | 从当前可启动的产品 slot 中选择下一个要服务的产品，重复选择形成顺序 |
| C | 任务规划 / task planning | 为 B 选中的产品选择下一项可启动任务 |
| D | 资源分配 / resource allocation | 为 C 选中的任务分配工人及适用的机器人 |

实现上 D 分成工人头与机器人头，所以四个业务层对应五个 Q 头：A、B、C、D-human、D-robot。当前两个 D 头都以任务为父上下文，并非新增“工人动作再条件化机器人动作”的串行策略。

**需要纠正：“区别就是 B 步内有决策时钟”不够准确。**

1. B 的业务职责仍是优先级决策，但计算形式从“每个 tick 生成一份排序”改为“每次提交后重新选择下一个产品”。
2. A、B、C、D-human、D-robot 都有独立的学习决策时钟。
3. 步内资源状态、动作可启动性、执行反馈、下一父上下文和 bootstrap mask 也同时修正。

R0 一个仿真 tick 的构造过程为：

```text
A：若允许释放，选择 next product 并更新影子 staging
    ↓
B：选择当前可服务的产品
    ↓
C：为该产品选择可启动任务
    ↓
D：选择当前可用工人/机器人
    ↓
提交到影子资源池，更新占用状态
    ↓
若仍可分配且未达到 K 次：返回 B，重新选择
    ↓
将 dispatch list 交给环境，推进一个仿真 tick
    ↓
读取实际启动结果与团队奖励，更新各头的 pending / replay
```

这里的“提交到影子资源池”只是构造本 tick 的并发分配列表，不让物理时间提前流逝。实际是否启动以环境反馈为准。

## 2. B 仍然决定优先级，但不再预先固定整份排序

令 $s_t$ 为 tick $t$ 的工厂状态，$a_t^A$ 为该 tick 的 A 动作。没有有效释放时，$a_t^A$ 表示零动作。

G0 的 B 可以示意为：先在同一状态上计算产品分数，再排序，随后按该顺序执行 C/D：

$$
\pi_t^{\mathrm{G0}}
=\operatorname{Rank}\left(\{Q_B(s_t,a_t^A,b)\}_{b\in\mathcal B_t}\right).
$$

上式省略探索分支和实现中的固定规则。特别地，G0 将 staging slot 放在排序末尾。

R0 的 B 则是条件于已提交前缀的逐次选择。令第 $k$ 次分配为：

$$
u_{t,k}=(b_{t,k},c_{t,k},h_{t,k},v_{t,k}),
$$

其中 $v_{t,k}$ 表示机器人动作，也可为空。定义此前已提交的分配前缀：

$$
p_{t,k}=(u_{t,1},\ldots,u_{t,k-1}).
$$

先将 A 的释放反映在影子状态，再逐次提交：

$$
\tilde s_{t,1}=\mathcal U_A(s_t,a_t^A),
\qquad
\tilde s_{t,k+1}=\mathcal U(\tilde s_{t,k},u_{t,k}).
$$

利用分支中的 B 动作为：

$$
b_{t,k}
=\underset{b\in\mathcal B(\tilde s_{t,k})}{\arg\max}
Q_B\left(\tilde s_{t,k},a_t^A,b\right).
$$

训练时在这个合法集合上执行 epsilon-greedy。C/D 完成一次影子提交后，资源状态改变，B 的候选集合和评分输入也随之改变。最终形成：

$$
\pi_t^{\mathrm{R0}}=(b_{t,1},\ldots,b_{t,m_t}),
\qquad 0\le m_t\le K.
$$

该序列是本 tick 实际构造出的服务优先顺序，不要求为所有产品形成一份完整排列。

**一个必须在论文中披露的实现差异：**R0 允许合法的 staging slot 与 WIP slot 一起被 B 选择，不再强制 staging 最后。这意味着 R0 不只是回放格式修复，也包含动作生成方式的调整。若要单独量化“决策时钟”的贡献，应进一步保持相同 B 规则做控制实验；当前 R0/G0 对比只能评估整个一致性改造组合。

## 3. 执行一致的动作状态与可行性

对于头 $\ell\in\mathcal H=\{A,B,C,D_h,D_r\}$，定义决策信息：

$$
x_{t,k}^{\ell}
=\left(\tilde s_{t,k}^{\ell},q_{t,k}^{\ell},\mathcal A_{t,k}^{\ell}\right).
$$

这里 $q^{\ell}$ 表示父动作上下文，避免与 C 层任务动作 $c$ 混淆：

$$
q^A=\varnothing,
\qquad q^B=a_t^A,
\qquad q^C=b_{t,k},
\qquad q^{D_h}=q^{D_r}=c_{t,k}.
$$

A 使用释放前状态；B/C/D 使用对应分配前缀更新后的影子状态。网络通过现有预处理与编码器接收这些信息，并不直接输入整条动作历史。$x$ 是决策信息的记号，不代表已经证明当前特征构成严格充分的马尔可夫状态。

令 $\mathcal F(s,u)$ 为任务实际启动的可行性判定，包含所需工人、机器人、工作站与指定区域龙门资源。目标是使构造出的分配满足：

$$
u_{t,k}\in\mathcal U_{\mathrm{feas}}(\tilde s_{t,k}),
\qquad
\mathcal U_{\mathrm{feas}}(s)=\{u:\mathcal F(s,u)=1\}.
$$

当前实现通过与环境共享的 task record 准备逻辑筛选 C 候选并核验完整分配，沿用当前环境的资源可行性假设，并非枚举所有产品×任务×工人×机器人组合。

影子提交同样使用环境资源占用逻辑。例如，同区物流不需要的机器人不会被继续保留，龙门占用对应具体区域，而不只是“任意一个空闲龙门”。

定义环境实际接受指示量：

$$
z_{t,k}=\mathbf 1\{u_{t,k}\text{ 被环境接受并启动}\}.
$$

B/C 的该次执行事件只有在 $z_{t,k}=1$ 时保留；D 还核对实际执行资源与提出的动作相同。机器人未实际使用时，不记录对应机器人执行事件。A 的有效产品释放单独记录，不要求同 tick 必须有任务启动。

被拒绝的提议可能是另一种“尝试调度 MDP”中的合法自环，不能一般性宣称其不应学习。R0 选择的是执行决策流：拒绝提议不进入该流，期间环境时间与奖励继续计入已有 pending；reject 指标用于检查可行性模型与执行时序差异。

## 4. 独立决策时钟：不仅是 B，也不仅是步内时钟

每个头都有一个按真实执行事件排序的序列。用 $n$ 表示该头的决策序号，$\tau_n^{\ell}$ 表示该事件所在的仿真 tick。

同一个 tick 内，事件通过实际提交顺序排序。因此即使：

$$
\tau_n^{\ell}=\tau_{n+1}^{\ell},
$$

两个事件仍然不同：它们可能具有不同资源状态、父上下文与合法动作集合。

定义时间跨度：

$$
\Delta_n^{\ell}=\tau_{n+1}^{\ell}-\tau_n^{\ell}\ge 0.
$$

- A 的 interval 从本次释放连接到下一次释放。
- B 的 interval 从本次产品选择连接到下一次实际服务选择。
- C 的 interval 从本次任务规划连接到下一次实际任务规划。
- D-human / D-robot 分别连接各自下一次实际资源分配。
- 其他头出现新决策不会强制结束当前头的 interval。
- 同一 tick 中连续两个 B 或 C/D 事件的时间跨度可以是0。

一个示例：

| 仿真 tick | 真实决策事件 |
|---|---|
| 100 | A 释放；B/C/D 完成分配1；B/C/D 完成分配2 |
| 101 | 无新决策，只推进环境 |
| 102 | B/C/D 完成分配3 |
| 103 | A 再次释放 |

由此得到：

- A 在100到103的跨度为3。
- B 在100的分配1到分配2的跨度为0，下一观测包含分配1的占用。
- B 在100的分配2到102的分配3的跨度为2。
- 若某个分配没有使用机器人，D-robot 的下一事件可能更晚，不能照搬 B 的跨度。

因此，这里的决策时钟是**每个头的事件边界与奖励结算规则**，不是新增一个仿真计时器，也不是为每个 tick 强行制造一个训练样本。

## 5. 半马尔可夫经验定义

设 $r_t$ 是环境执行 tick $t$ 的联合动作后得到的团队奖励。对于非终止 interval，累计奖励为：

$$
G_n^{\ell}
=\sum_{j=0}^{\Delta_n^{\ell}-1}\gamma^j r_{\tau_n^{\ell}+j}.
$$

令 $\alpha$ 为原有 decision reward scale，当前配置 $\alpha=0.01$，则存储奖励为：

$$
R_n^{\ell}=\alpha G_n^{\ell}.
$$

各头的经验元组为：

$$
\boxed{
 e_n^{\ell}=
 \left(
 x_n^{\ell},a_n^{\ell},R_n^{\ell},
 \gamma^{\Delta_n^{\ell}},x_{n+1}^{\ell},d_n^{\ell}
 \right)
}
$$

其中两端状态分别包含各自的父上下文和合法集：

$$
x_n^{\ell}=(\tilde s_n^{\ell},q_n^{\ell},\mathcal A_n^{\ell}),
\qquad
x_{n+1}^{\ell}=(\tilde s_{n+1}^{\ell},q_{n+1}^{\ell},\mathcal A_{n+1}^{\ell}).
$$

不能把 $q_n^{\ell}$ 无条件复制成 $q_{n+1}^{\ell}$。例如，D 下一次面对任务Y，bootstrap 就应使用任务Y的上下文，即使当前经验的动作服务任务X。

零时间 interval 使用空奖励和：

$$
\Delta_n^{\ell}=0
\quad\Rightarrow\quad
R_n^{\ell}=0,\qquad\gamma^{\Delta_n^{\ell}}=1.
$$

当前 tick 的真实环境奖励计入该 tick 最后一个同头决策开启的 interval；此前微决策通过零时间价值连接向后传递。步内链长度受 K 限制，没有无限次零时间分配。

当 episode 在边界 $T$ 结束时，对每个仍 pending 的头累计至 $T$，设置 $d_n^{\ell}=1$ 并关闭 interval；下一合法集置零。自动 reset 后的新订单状态不作为旧订单的有效 bootstrap 状态。checkpoint restore 跳转则直接丢弃 pending。

## 6. R0 的 Double-DQN 更新与实现对应

区分共享编码器参数 $\phi$、头参数 $\theta_{\ell}$ 与目标头参数 $\bar\theta_{\ell}$。定义：

$$
Q_{\phi,\theta_{\ell}}^{\ell}(x,a)
= f_{\theta_{\ell}}^{\ell}\bigl(g_{\phi}^{\ell}(x)\bigr)_a.
$$

在下一次真实决策的合法集合上选动作：

$$
a_{n+1}^{\ell,*}
=\underset{a\in\mathcal A_{n+1}^{\ell}}{\arg\max}
Q_{\phi,\theta_{\ell}}^{\ell}(x_{n+1}^{\ell},a).
$$

不考虑数值裁剪时，R0 的目标为：

$$
\boxed{
 y_n^{\ell}
 = R_n^{\ell}
 +(1-d_n^{\ell})\gamma^{\Delta_n^{\ell}}
 Q_{\phi,\bar\theta_{\ell}}^{\ell}
 \left(x_{n+1}^{\ell},a_{n+1}^{\ell,*}\right)
}
$$

对于终止样本直接取 $y_n^{\ell}=R_n^{\ell}$，无需对空合法集定义 argmax。对一般空合法集，代码也将 continuation value 置零。

实现保留原有数值保护，实际目标为：

$$
y_{n,\mathrm{impl}}^{\ell}
=\operatorname{clip}_{[-C_Q,C_Q]}
\left[
\operatorname{clip}_{[-C_R,C_R]}(R_n^{\ell})
+(1-d_n^{\ell})\gamma^{\Delta_n^{\ell}}
Q_{\phi,\bar\theta_{\ell}}^{\ell}
(x_{n+1}^{\ell},a_{n+1}^{\ell,*})
\right],
$$

其中当前默认 $C_R=100$、$C_Q=500$。关闭相应裁剪时可视作阈值无穷大。

用优先经验回放与加权 Huber 损失学习：

$$
\mathcal L_{\ell}
=\frac{1}{N_{\ell}}\sum_{n=1}^{N_{\ell}}
 w_n^{\ell}\,operatorname{Huber}_{\delta}
 \left(y_{n,\mathrm{impl}}^{\ell}
 -Q_{\phi,\theta_{\ell}}^{\ell}(x_n^{\ell},a_n^{\ell})\right).
$$

各个有可训练 batch 的头损失求和，联合更新共享编码器；每个头更新自己的 Q 网络。每个头每次学习事件只贡献一次 batch loss，不再按同 tick 的 dispatch 数重复进行多份随机 batch 更新。

**R0 与 R1 必须区分：**R0 上式的两端都使用在线编码器 $\phi$，只有 Q 头使用目标副本。R1 才将目标值分支的编码器替换为 $\bar\phi$，并通过 EMA 更新。R2 的候选匹配结构与各 reward 消融也不能计入 R0 的单独贡献。

## 7. 论文中的贡献点表述

### 中文：可用于 Introduction 的贡献列表

> 我们提出一种执行一致的分层决策与半马尔可夫经验回放机制，用于解决并发制造调度中动作生成、资源实际占用与价值更新之间的语义不一致问题。在保持产品释放、产品优先级、任务规划和资源分配的分层结构下，该机制通过前缀条件化的逐次产品选择构造步内服务顺序，依据当前资源状态与环境启动约束生成可执行分配，并利用实际执行反馈记录经验。进一步地，我们为各决策头建立独立事件时钟，使经验连接至该头下一次真实决策，保留相应的父动作上下文、合法动作集合与时间跨度，从而支持与执行过程一致的价值更新。该机制保持基础团队奖励不变，使动作与回放设计的影响能够与奖励塑形分开评估。

### English: contribution paragraph

> We propose an execution-consistent hierarchical decision and replay mechanism for concurrent manufacturing scheduling. While retaining the decomposition into product release, product prioritization, task planning, and resource allocation, the mechanism constructs within-tick service orders through prefix-conditioned sequential product selection. Feasibility-aware dispatch generation and execution feedback align recorded actions with actual resource commitments. We further introduce separate event clocks for the decision heads, connecting each executed decision to the next actual decision of the same head with its corresponding parent-action context, feasible action set, and elapsed simulation time. This supports semi-Markov value updates without introducing additional reward terms, allowing the effects of decision and replay design to be evaluated separately from reward shaping.

### 可用于方法小节的标题

**Execution-Consistent Hierarchical Dispatch and Decision-Time Replay**

中文：**执行一致的分层调度与决策时刻经验回放**。

“decision-time replay”在这里是对本实现的描述性命名，不意味着宣称独立决策时钟、SMDP或Double DQN是首次提出的新概念。

## 8. G0/R0 对照应如何解释

| 对照项 | G0 | R0 |
|---|---|---|
| A | 选择 next product | 同一业务职责，补记 A-only |
| B | 同一状态评分/排序，staging 固定最后 | 当前前缀下逐次选择，形成服务顺序 |
| C/D | 有步内合法性过滤，但网络观测/回放主要沿用 tick 起点状态 | 观测与合法集使用当前影子资源状态 |
| 记录依据 | 提议 dispatch 触发全局 interval | 实际执行反馈触发对应头事件 |
| 下一父上下文 | 当前上下文复用 | 保存下一真实决策上下文 |
| 时钟 | 全局 meaningful-decision interval | 五个头分别结算 interval |
| 时间折扣 | 已有按 tick 累积的 gamma^dt | 保留折扣形式，但改变 interval 边界 |
| C-none | 执行与目标合法集可能不一致 | 两端均使用实际合法集 |
| 奖励 | 基础团队奖励 | 相同基础团队奖励 |

注意：不能声称“G0 没有 SMDP 折扣，R0 首次加入 gamma^dt”。G0 已有时间跨度折扣；R0 的变化在于**用哪两个决策事件定义这个跨度，以及两端携带什么上下文和合法集合**。

主实验研究问题可表述为：

> 在不改变基础奖励的情况下，将动作构造、环境执行和逐层经验回放对齐，是否能够改善制造调度的有效样本利用与最终 makespan？

建议同时报告成功率、makespan、达到目标性能所需仿真 tick、墙钟时间、实际启动率、多候选决策数和各头更新次数。不同动作路径会改变有效决策数量，不能只比较训练 loss 或 buffer size。

## 9. 理论与实验边界

- 上述式子是当前分层独立 Q 学习的目标构造，不是联合最优 Bellman 方程的证明。实际下一父上下文来自运行中的上层策略，该策略会随训练变化。
- 保留同一团队奖励并不自动解决全部跨层信用分配问题；机制修正的是经验的执行语义、时间边界和条件信息。
- 当前 R0 不引入主动等待动作，采用有可行任务就分配的策略空间，不能保证包含所有全局最优调度。
- 神经网络观测压缩、策略非平稳性和函数逼近仍存在，不能由这些公式推出收敛或全局最优保证。
- R0/G0 是组合改造对照；只改变决策时钟、只改变可行性、只改变 B 的实验目前不是既有独立系列，若论文要求逐项归因需要新增消融。
- “显著改善样本效率/调度性能”须由多训练seed、独立开发集选模型和固定测试集评估支持；本文件不预填尚未得到的性能结论。
- 此文是与代码对齐的方法草稿，不是已完成文献检索的创新性证明；投稿前应补充相关工作与准确引用。

## 10. 代码对应

- `build_decision_action`：A与循环B/C/D、逐次产品选择、执行trace。
- `DecisionPool.feasible_masks / record / commit / observation`：环境同源启动检查、影子提交与前缀观测。
- `DecisionReplay.observe / _close`：每环境每头pending、执行过滤、奖励结算、next context与terminal。
- `DecisionReplay.encode / learn`：下一父上下文编码、Double-DQN分支、联合更新。
- `TaskManager.prepare_new_task_record / update_new_task_record`：启动准备与实际执行；`dispatch_outcomes`反馈实际接受与资源编号。

以上新逻辑默认关闭，原G/E路径保留；详细运行方式见 [R系列实验说明](experiment_r_series.md)。
