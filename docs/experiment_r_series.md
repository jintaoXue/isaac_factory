# R 系列：动作与回放一致性对照

诊断归档：[2026-09-27 诊断](rl_diagnosis_2026-09-27.md)。新代码只在 `decision_consistent=true` 时启用；默认 false，原 G/E 动作及学习路径保留。R 与 G0 使用 gap 动力学、N10、K10、T35000、scratch、无教师。E 默认是 legacy 动力学，不能把 R/E 的原始数值差直接归因于算法。

论文贡献、B 的业务语义与独立决策时钟公式见 [R0 方法文档](r0_execution_consistent_hierarchical_replay.md)。

## 设计判断

保留 A→B→C→D 的因子化动作空间。优先修正动作执行语义，不改成难以探索的巨大联合离散动作。

- A：产品释放可单独改变 staging，即使没有 dispatch 也记录。
- B：从当前“至少有一个可启动任务”的产品集合中选择一个 slot；每次提交后重新选择，取代旧的整表排序后对多个 slot 重复分配同一奖励。
- C：选择该 slot 当前可启动的真实任务；none 不进入执行和 bootstrap 合法集。
- D：选择当前可用的人/机器人；完整前缀资源状态输入网络。沿用人/机器人分头，未改成联合 critic。
- 一次 tick 最多 K 次提交。影子资源提交与环境共用 task record 准备/资源占用逻辑，包括指定区域龙门和同区运输不占 AGV。
- 环境仍是最终执行者；回放根据实际 `dispatch_outcomes` 接受、拒绝、有效人员/机器人编号过滤，不能用提议冒充启动。

这仍是 work-conserving 调度：没有新增主动等待动作，不保证覆盖所有全局最优调度；也仍是多头独立 Q 学习，上游非平稳性没有完全消失。R0 是一致性修复组合，不应将其整体增益宣称为某个单独模块的贡献。

## 系列与奖励消融

| 入口 | 相对前一主组的变化 | 训练奖励 |
|---|---|---|
| G0 | 旧路径基线 | 原 makespan 奖励 |
| R0 | 可启动动作 + 逐层决策回放 + 正确前缀/next context/mask | 原奖励 |
| R1 | R0 + EMA target encoder | 原奖励 |
| R2 | R1 + task-human match 候选结构 | 原奖励 |
| R2-mismatch | R2 + 仅人员错配项 | mismatch=.05，overwork=0 |
| R2-fatigue | R2 + 仅疲劳项 | mismatch=0，overwork=.01 |
| R2-both | R2 + 两项 | mismatch=.05，overwork=.01 |
| R2-greedy | 相同新动作/target 路径，人 D 固定贪心 | 原奖励；人头不训 |

R2 复用候选 match 特征和评分结构，但 `prior_weight` 从0初始化，不沿用旧 match 的固定正向速度初始化。其余匹配参数从零训练的随机初始化开始；没有教师、BC 或额外奖励。R2-greedy 的人头不执行、不训练，因此不启用无用的 match 网络。

reward 应作为独立消融：错配/疲劳分别影响目标，必须先证明不加 shaping 的 R0/R1/R2 有什么效果，再判断 shaping 是否额外有益。当前保留既有系数与 cap=.04，不把自动调权混进本轮实验。所有 `eval-R*` 都关闭 shaping，但开启人因监测。

## 决策时间与 TD

每个 head、每个 env 有独立 pending interval。遇到该 head 下一次实际执行决策才结算：

`(pre, context, action, actual_mask, R, gamma^dt, next_pre, next_context, next_actual_mask, done)`

- `R = sum_k gamma^k * r_k`，沿用 gamma=.9999、reward scale=.01。
- 同一 tick 内同 head 连续决策的 dt=0、R=0、discount=1。
- 其他 head 决策不会强制截断本 head 的 interval。
- episode 成功或截断时，flush 全部 pending，done=true、next_mask=0，禁止从自动 reset 后的状态 bootstrap。
- checkpoint restore 跳转丢弃 pending，不制造虚假环境转移。
- 拒绝动作不进入人员加工决策流，期间时间成本继续累积到已有 pending；若 reject 不为0，应检查 feasibility 与环境推进时序，而不是解释成已发生加工。

R1/R2 的 Double DQN：在线 encoder/Q 选择 next action，target encoder/target Q 计算值；encoder 在联合更新后按相同 tau 做 EMA。后期 tau 衰减沿用现有日程。

采样仍为 PER；每次有新 transition 且达到 learn_interval 才联合更新一次。有效样本数与旧路径不同，所以除了仿真 tick 预算，还要报告实际 decisions/updates 和墙钟时间。

## 运行

在项目根目录、已有 isaac-lab 环境执行：

```bash
bash run_2026_journal_experiments.sh R0 cuda:0 --dry-run
bash run_2026_journal_experiments.sh R0 cuda:0
bash run_2026_journal_experiments.sh R1 cuda:0
bash run_2026_journal_experiments.sh R2 cuda:0
```

这些命令是分别启动各实验，不建议把同一 GPU 上的多组长训练同时启动。默认100局、seed42，W&B 项目仍为 HcFactory_TPA；目录分别为 `logs/rl_games/HcFactory/hier_R0-S42` 等，已有目录自动加 `-v1` 后缀。

```bash
HC_R_SEED=53 bash run_2026_journal_experiments.sh R2 cuda:0
HC_MAX_HARD_EPISODES=10 HC_R_WANDB=0 bash run_2026_journal_experiments.sh R0 cpu
bash run_2026_journal_experiments.sh R2-mismatch cuda:0
bash run_2026_journal_experiments.sh R2-fatigue cuda:0
bash run_2026_journal_experiments.sh R2-both cuda:0
bash run_2026_journal_experiments.sh R2-greedy cuda:0
```

`HC_R_SEED` 是训练seed；不要用43–52作为训练seed后又把它们当独立测试集。默认epsilon日程与G0一致，1→.05 / 150万tick；这是控制变量选择，不表示100局一定覆盖充分的低探索训练。

默认只读 `.wandb_local.env` 中 W&B 身份配置，不改全局登录。`HC_R_WANDB=0` 禁用上传。底层入口为 `tools/run_r_series.py`，可直接调用。R recipe 固定 gap/N10/K10/T35000，避免继承旧 E/G 的教师/ORU/shaping 开关。

## checkpoint 与评估

R 权重使用原六文件形式，额外写入 `decision_schema_step_<step>.json`，最后写此文件作为完整 bundle 标志。加载时检查六文件、R/旧语义、match结构、固定人策略与target编码器配置。R/G/E 不能静默交叉加载。

与旧系统一样，这些是**权重 checkpoint**，不是精确恢复训练进度的快照：不含optimizer/replay/pending/随机状态；加载后 target encoder 从在线encoder重新同步。主实验从零开始，不将热启/续训混入scratch结果。

评估显式指定训练目录和step：

```bash
HC_LOAD_DIR=logs/rl_games/HcFactory/hier_R2-S42 \
HC_LOAD_STEP=300000 \
HC_TEST_SEEDS=9001,9002,9003,9004,9005,9006,9007,9008,9009,9010 \
bash run_2026_journal_experiments.sh eval-R2 cuda:0
```

上面300000仅为命令示例，应换成真实存在的checkpoint。先在9001–9010开发集选step，再固定模型，用默认43–52测试。`eval-R2-mismatch` 等必须对应其训练组名，所有评估均 epsilon=0、shaping关闭。重复评估自动分配独立输出目录，避免覆盖。

主指标仍为 `MetricFullorderCore/05_makespan`（训练每局）和 `MetricFullorderCore/09_mean_makespan`（协议评估），同时报告成功率，不能用训练最低点代替测试性能。

## 新增诊断指标

`MetricDecision/` 下是训练全过程累计计数：

- requested / accepted / rejected / accept_rate：提议与真实启动的一致性。
- execution_mismatches：D 提议与实际执行工人/机器人不一致。
- A_only：成功进入新决策流的单独产品释放。
- decisions_<head> / choices_<head>：执行决策数、有多个合法候选的决策数。
- transitions_<head> / context_changes_<head>：样本数与下一父上下文改变次数。
- updates_<head>：实际优化次数；与 buffer size 和 loss 联合分析。
- restores：恢复导致 pending 丢弃次数。

理想情况下 reject 与 execution_mismatches 为0；单一候选比例高时，不能把大量决策误认为大量可学习选择。核心问题修正后仍需多seed实验，不能承诺R一定胜过greedy。

## 验证

```bash
HC_SIM_BACKEND=logic OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s tests -v
```

新增 `test_decision_consistent.py` 验证 A-only、拒绝分配、next_context、准确折扣、零时间决策、终止、restore、真实并发前缀、优化梯度和target EMA。

`test_r_entry.py` 在临时目录执行 R0/R2 的160 tick截断短训练、保存、加载与评估。这个极短测试故意不能完成订单，只验证入口/梯度/checkpoint/终止链路；不构成性能结果。

本次开发还执行了 gap/seed9002 的真实环境多次dispatch诊断：2147 tick 内18次提议全部启动，最后一次包含两次分配，后一次观测和mask正确排除了前一次已用工人。该数值只描述这个诊断样本。
