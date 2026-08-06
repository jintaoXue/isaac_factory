# BSTAN Baseline 数据质量提升执行设计

## 1. 文档状态

- 状态：Phase E0-E2 已通过；v2.3 已在 10 个 Pilot episode 上重建验收，dataset v3 smoke 数据集已通过张量、target mask 和 split 门禁
- 基线：现有 `BSTAN-style GAT-GRU baseline`
- 前置版本：Phase A-D 已跑通，`dev_tyx` 的 18 单、扰动与龙门架修复已合入
- 本轮目标：提高训练数据的正确性、覆盖度和评估可信度，再重新训练 baseline
- 本轮不包含：其它 baseline、主模型、在线预测反馈、调度策略优化

本轮所说的“提升 baseline 效果”首先指提升结果的可信度和稳定性，而不是只提高单次 test 指标。模型结构先保持不变，只有数据门禁通过后才调整采样、损失和分类阈值。

### 1.1 Phase E0-E1 实现记录

本地已完成：

```text
collector_version = v0.6
disturbance event phase = CONFIG / START / END / SYSTEM
planned/actual event time and actual target logging
deterministic per-episode event schedule and target sampling
machine target limited to process-relevant resources
human/machine/gantry disturbance state -> DOWN
material permanent removal -> recoverable production-batch supply hold
ABORTED episode audit rejection
raw data quality JSON/CSV report
```

material hold 只选择当前 `producing` 且刚好没有 ongoing task 的批次；若计划时刻没有候选，则等待到某个在制批次进入工序间隙后再实际激活。hold 只屏蔽该批次的下一工序 action mask，并保留实际物理库存；恢复后自动继续生产。未启动批次不再作为 fallback，避免事件完成后生产轨迹完全不受影响。

本地验收：

```text
Phase E0-E2 pure-Python tests = 22 passed
Python syntax compilation = passed
Isaac Sim five-scenario smoke Pilot = passed
```

### 1.2 服务器 Pilot 验收记录

15 单负载下，none、machine、human、logistics、material 均完成 `15/15`，raw audit 均为 `trainable=1`。machine、human、gantry 均观察到 `DOWN -> IDLE` 恢复；material 在 step 515–660 为 `shortage_flag=1`，step 661 恢复为 0。

v1 标签校准统计显示：

```text
will_bottleneck_rate = 37.2%–52.9%
buffer argmax rate = 80.0%–89.2%
threshold=0.70, margin=0.10 candidate rate = 19.7%–24.6%
```

因此 v1 的强制 argmax 和 buffer 支配问题成立。当时的敏感性复算中，`0.65` 的总体正样本率为 `6.85%`，28 个事件，hot window 中 buffer 占约 `52.6%`；`0.70` 的总体正样本率仅 `3.9%`。该结果是 v2.3 修订前的历史记录，不是当前冻结参数。

material 有效目标修正后重跑：计划 step 514，等待在制批次工序间隙后于 step 773 激活，step 919 结束，库存短缺标记在 step 920 恢复为 0，最终完成 `15/15`。其 v2 结果为 5 个事件、`6.92%` 正样本率、28 个 hot windows，不再与 none 场景完全相同。替换该 run 后，五场景总体正样本率约 `6.4%`，Phase E2 标签验收通过。

第二轮使用 seed 43，每个场景各补 1 个 episode。5 个 episode 均完成 `15/15` 且 raw audit 为 `trainable=1`；machine 和 material 的实际目标相对 seed 42 发生变化，所有扰动的开始时间和持续时间均有变化。v2 各场景正样本率为 `4.57%–8.73%`，按窗口加权总体约 `7.1%`；hot window 中 buffer 占约 `54.2%`，与 seed 42 的约 `53.7%` 接近。两轮 Pilot 共 10 个有效 episode，标签分布与节点覆盖稳定。

## 2. 当前基线与问题判断

### 2.1 已跑通结果

当前 smoke dataset 来自 6 个无扰动 episode，每个 episode 只有 3 个订单：

```text
total samples = 395
positive samples = 28（7.09%）
train / validation / test positives = 20 / 4 / 4
test PR-AUC = 0.15714
test no-event PR-AUC = 0.06061
test F1@0.5 = 0.23529
```

该结果证明了数据到模型的链路可运行，但不能作为正式 baseline 结论：

1. validation 和 test 各只有 4 个正样本，指标方差极大；
2. 所有 episode 属于同一无扰动 scenario；
3. 正样本只覆盖一个 target type；
4. node/type 满分主要来自样本单一，不能说明泛化能力；
5. 当前数据仍是旧的 3 单配置，不能代表合入后的 18 单环境。

### 2.2 当前数据链路的主要风险

| 优先级 | 问题 | 对 baseline 的影响 |
|---|---|---|
| P0 | 旧训练数据是 3 单、单场景、6 episode | 正样本和瓶颈迁移不足 |
| P0 | L2 扰动开始和结束分两行记录，离线阶段未按 `disturbance_id` 配对 | 开始行可能被解释为持续到 episode 结束 |
| P0 | 配置级扰动和运行时事件共用 `disturbance_flag` | 非 `none` 场景可能整局恒为 1，丢失事件时间信息 |
| P0 | `working_disturbance_absent` 会落入一般 `working_* -> PROCESSING` | 离岗人员可能被错误统计为加工中 |
| P0 | material shortage 通过 reset 时永久少放物料实现 | 订单可能永远无法完成，只产生 `ABORTED`，不能进入训练集 |
| P0 | watchdog 样本虽已标记 `ABORTED`，但尚无全数据集统计门禁 | 某些配置可能大面积失效而未被及时发现 |
| P1 | L2 事件开始时刻和目标固定 | 模型可能学习时间位置或固定节点，而不是系统传播规律 |
| P1 | 记录的是配置目标，不一定是实际激活的 fallback 资源 | root-cause 审计不精确 |
| P1 | 当前 weak label 每窗口只取一个 argmax 节点 | 同时瓶颈和瓶颈迁移会被压缩，事件易碎片化 |
| P1 | 当前 type 等于 resource type | type head 与 node head 高度重复，不代表扰动原因预测 |
| P1 | 当前按 episode 分组，但同 run 的 episode 可进入不同 split | 相同配置和随机过程可能造成弱相关泄漏 |
| P1 | 只报告固定阈值 `0.5`，没有 episode 级置信区间 | 小样本下难以判断提升是否稳定 |

## 3. 本轮设计原则

### 3.1 先保证真值，再增加样本

正式重采集前必须先修复扰动区间、状态映射和可恢复缺料。否则增加 episode 只会放大错误标签。

### 3.2 扰动不是瓶颈标签本身

扰动用于描述可能的 root cause，瓶颈仍由未来窗口中的队列、占用、阻塞、饥饿、产出和系统吞吐影响确定。

```text
disturbance event != bottleneck event
```

同一个扰动可能没有形成瓶颈，也可能通过上下游传播后在另一个节点形成瓶颈。不能直接把被扰动节点写成 `future_bottleneck_node`。

### 3.3 不使用未来扰动作为输入

在 anchor 之前已经发生的扰动状态可作为历史输入；anchor 之后才发生的扰动只能用于标签解释或评估，不能进入输入特征。

### 3.4 正常和异常 episode 分流

```text
END      -> 可进入 Phase B/C 和模型训练
ABORTED  -> 保留 raw log 与审计摘要，禁止进入训练
```

死锁样本属于运行故障样本，可用于系统稳定性和故障诊断研究，但本轮不把截断序列当作普通负样本或瓶颈训练样本。

## 4. 数据语义修正

### 4.1 扰动事件采用可配对语义

保留 append-only CSV，但明确每行事件阶段：

```text
disturbance_id
event_phase = CONFIG | START | END
planned_start_time_step
actual_start_time_step
actual_end_time_step
planned_duration_steps
actual_target_resource_id
disturbance_type
intensity
```

离线阶段按 `(run_id, env_id, episode_id, disturbance_id)` 配对：

1. `CONFIG` 表示整局生效的 L0/L1 场景参数，不与 L2 区间混用；
2. `START + END` 形成实际 L2 区间；
3. 只有 `START` 且 episode 为 `END` 时视为未恢复数据错误；
4. 只有 `START` 且 episode 为 `ABORTED` 时允许右截断，但仍不进入训练；
5. 重复 START、孤立 END、负持续时间均触发 validator 失败。

### 4.2 拆分场景特征与运行时事件特征

建议替换当前含义混合的单一 `disturbance_flag`：

```text
global:
  scenario_disturbance_flag
  scenario_disturbance_intensity

node/window:
  runtime_disturbance_active
  runtime_disturbance_elapsed_ratio
```

其中：

- `scenario_disturbance_flag` 表示该 episode 是否使用非标称 L0/L1 配置；
- `runtime_disturbance_active` 只在实际 L2 时间区间和实际目标节点上为 1；
- 对传播分析可增加邻居节点的 `upstream_disturbance_active`，但第一轮不是必须项；
- 原 `disturbance_flag` 不再作为模糊兼容字段进入新版 dataset。

### 4.3 修正资源状态映射

第一轮至少固定以下语义：

```text
machine invalid                  -> DOWN
human working_disturbance_absent -> DOWN
gantry invalid                  -> DOWN
waiting for material/partner    -> STARVED
unable to release downstream    -> BLOCKED
normal operation                -> PROCESSING
```

状态映射需要单元测试覆盖，特别验证离岗人员不会贡献 `active_pct_s`。

### 4.4 物料扰动改为可恢复事件

当前永久跳过原材料可能使 18 单订单无法完成。正式训练数据改用可恢复方案：

1. episode 初始物料仍保证理论上可完成全部订单；
2. 在运行中延迟某批物料可用时间，或临时冻结供应 buffer；
3. 到恢复时间后补充物料或解除冻结；
4. START/END 和实际受影响 material/buffer 必须写入 disturbance log；
5. 永久缺料模式仅保留为故障压力测试，不进入 baseline 正式训练矩阵。

### 4.5 扰动时刻和目标随机化

固定 `600/700/800` step 和固定 `human_0/gantry_0/某机器` 容易产生捷径。每个 episode 应基于 run seed 和 episode id 确定性采样：

```text
event_start_ratio in [0.20, 0.60] of nominal makespan
duration around intensity-specific range
target sampled from active and process-relevant resources
```

实际激活可能因目标忙碌而延后，日志必须记录 `planned_start` 与 `actual_start`。采样结果可复现，但不同 episode 不应完全相同。

## 5. 标签质量提升

标签规则、版本演进、字段语义和 Pilot 实测结果详见 `doc/implement/BSTAN瓶颈标签v2.3实现说明.md`。

### 5.1 标签版本约束

当前代码只生成并接受小幅修订后的正式标签版本：

```text
label_version = bstan_weak_v2_3
```

旧版本实验通过对应 Git commit 复现，不在当前运行路径保留兼容分支。Phase C 收到旧标签时必须直接报版本错误。

### 5.2 v2.3 标签定义

v2.3 保持 baseline 的单节点预测接口。buffer/storage 继续作为图节点和模型输入，提供 occupancy、queue 和物料上下文，但当前 raw log 没有可靠的 buffer 局部上下游阻塞映射，因此不进入第一版预测目标集合。生产资源候选必须同时满足：

1. 节点局部压力分数达到 `0.50`：队列、waiting time、active duration、blocked/starved propagation 等；
2. 连续至少 2 个窗口，即持续至少 60 logic seconds；
3. 相对同类资源具有明显优势，避免所有节点低负载时强行选 argmax；
4. 对应时间段出现系统影响，即 operation throughput 下降或 WIP 积累中的至少一项。

若没有节点满足绝对条件，则该窗口为无瓶颈，不因存在 argmax 就强制产生正样本。

v2.3 排除前 120 logic seconds 的 warm-up，并将模型输入 `total_WIP` 和标签系统影响统一为窗口结束时点 WIP；窗口重叠计数仅保留在分析字段 `wip_overlap_count`。系统影响满足以下任一条件：

```text
end-of-window WIP 相对最近 3 个窗口增长 >= 1.0
最近 120 秒 processing operation 吞吐较此前 120 秒下降 >= 25%，且此前至少 2 个 operation 完成样本
```

operation completion 使用 v0.6 `job_trace.csv` 中已有的 processing `departure`，不需要重新采集。吞吐 support 不足时对应证据标记为 unavailable，不等价于系统未受影响。

系统影响原始触发后保持 `min_event_windows` 个窗口，使瞬时系统状态变化可以确认持续的局部压力；最终事件仍需同一生产节点连续至少 `min_event_windows` 个窗口。`material_shortage` 只作为 cause/输入特征，不再写成 bottleneck symptom。

场景配置写入分析字段，配对后的 runtime disturbance 只用于解释字段，不进入 `MODEL_FEATURE_FIELDS`。现有张量中的 `disturbance_flag` 输入槽固定为 0，真实运行区间只写入非模型字段 `runtime_disturbance_active`。

Phase C 仅接受 `bstan_weak_v2_3`，dataset schema 固定为 `bstan_dataset_v3`。输入旧标签时直接报版本错误。

v2.1 在两轮十个 Pilot 上结构验证通过，但实际得到 `3268` 个 observed anchors、`0` 个正样本。诊断显示每个 episode 有 `2-12` 个通过全部门禁的单窗口候选，但同节点最长连续长度全部为 `1`；整单 `stage_complete` 在 120 秒周期内也无法满足 throughput/cycle support。该版本不进入训练，保留在 Git 历史，不在当前代码路径保留分支。

v2.2 使用 operation throughput 和 impact hold 后得到 124 个事件、14.6% 正样本，但 `storage_BlackStorage_02` 占 121 个事件。该 buffer 在事件窗口始终满载，局部 queue growth 和 upstream blockage 均为 0，只因全局 starvation proxy 被错误当作局部传播而通过。v2.2 不进入训练，保留在 Git 历史。process-only 阈值扫描中，`0.50` 得到 53 个事件、6.36% 正样本、6 个生产节点并覆盖五类场景，因此冻结为 v2.3 初始阈值。这 53 个事件中没有事件在 runtime disturbance 开始后 120 秒内出现，因此当前 Pilot 只支持标签分布与流程验证，不支持扰动已引发可观测瓶颈的结论。

### 5.3 阈值校准

阈值不能根据 test 指标反向调节。建议流程：

1. 使用独立 pilot 数据中的正常场景和已知扰动场景检查特征分布；
2. 固定 v2.3 process score 权重与阈值 `0.50`，相对优势为 `0.10`；
3. 将完整 `score_config` 写入 `label_metadata.json`；
4. 冻结配置后再采正式 train/validation/test 数据；
5. 对阈值上下浮动 `0.05` 做敏感性分析，确认事件数量不会剧烈坍缩。

### 5.4 标签解释字段

在不改变 baseline 输出 head 的前提下，增加分析字段：

```text
bottleneck_symptom_type
candidate_cause_type
cause_target_resource_id
cause_label_confidence
disturbance_to_bottleneck_delay_s
```

`candidate_cause_type` 不作为本轮 baseline 的 type target。当前 `future_bottleneck_type` 仍为 resource type，以保持模型接口稳定；文档和结果中需继续声明这一限制。

## 6. 正式数据采集矩阵

### 6.1 订单负载校准

保持 `CfgRegistrationInfos["ProductWaterPipe"] = 18`，通过运行参数改变实际订单数：

```text
--product_order_count = 10, 15, 18
dimension = none
seed = 42
episodes per load = 1
```

三个档位均使用 `tmux` 完整运行，并记录完成时间、`completed_jobs`、episode 结束原因和资源利用率。选择能够稳定完成且形成明显排队竞争的档位作为正式数据负载；当前首选为 15 单，10 单作为低负载对照，18 单作为压力测试。未显式传参时保持现有 18 单行为。

降低订单数只减少实际进入生产的批次，不减少已注册的 18 批实体。产品排序 mask 必须在 `not_started` 数量为 0 后停止放行该产品，避免低负载配置继续启动额外批次。

服务器 Pilot 发现 human 扰动曾将内部对象名 `num_02_NormalHuman` 写入 `actual_target_resource_id`，而图节点目录使用 `human_2`。`v0.6` 起所有资源型扰动目标统一使用图节点 ID，raw audit 同时校验 machine、human、logistics 的 START 目标存在于资源目录且 END 目标不变。

### 6.2 Pilot 校验

先用最小矩阵验证数据正确性，不立即大规模采集：

```text
dimensions = none, machine, human, logistics, material
intensity = 1.0
seed = 42
episodes per run = 2
orders per episode = calibrated load (candidate: 15)
```

共 10 个 attempted episodes。Pilot 只用于检查事件、状态、标签与完成率，不进入正式 test 指标。

### 6.3 正式 v1 矩阵

Pilot 通过后采集：

```text
none: intensity 0.0
machine/human/logistics/material: intensity 0.5, 1.0, 1.5
seeds: 42, 43, 44
episodes per configuration: 3
orders per episode: calibrated load (candidate: 15)
policy: rule_based
```

配置单元数：

```text
1 + 4 * 3 = 13 scenario cells
13 * 3 seeds * 3 episodes = 117 attempted episodes
```

正式数据按 seed 固定拆分：

```text
train      = seed 42
validation = seed 43
test       = seed 44
```

这样每个 split 都包含全部 scenario cell，同时同一 run 的 episode 不会跨 split。该划分用于第一版同分布比较；未见强度和未见 policy 的 OOD 测试留到后续扩展。

### 6.4 不通过重复负样本制造规模

若某场景持续没有正样本，应先检查扰动是否生效和标签是否合理，不直接增加大量相同 episode。正式采集允许根据 pilot 删除无效强度或调整扰动范围，但调整后必须重新冻结配置。

## 7. 数据质量门禁

### 7.1 Raw episode 门禁

每个训练 episode 必须满足：

```text
START = 1
END = 1
ABORTED = 0
production_done = 1
completed_jobs = configured product order count
all expected resources have t=0 state
all L2 START events have exactly one END
actual target exists in node catalog or mapped buffer/material catalog
timestamps are monotonic and within episode boundary
```

### 7.2 Run/scenario 门禁

```text
valid completion rate per scenario cell >= 90%
ABORTED rate per scenario cell <= 10%
no single target receives > 50% of randomized events when alternatives exist
actual event start/duration have non-zero variation
none scenario runtime disturbance event count = 0
```

若某个 scenario cell 超过 ABORTED 上限，整格数据暂停进入 dataset，先修仿真或扰动语义。

### 7.3 Phase B/C 门禁

```text
feature/label primary keys unique
all input features finite
no future disturbance fields in input
positive target node/type exists and its target_node_mask/target_type_mask = 1
run/seed groups do not overlap across splits
every split contains positive and negative samples
overall observed positive rate target range = 5% to 30%
each split has at least 30 positive samples
```

`5%-30%` 是数据诊断范围，不是通过调阈值强行达到的指标。若超出范围，需要检查场景负载和标签，不得直接删负样本修改原始分布。

位置/type 指标的发布门槛：

```text
validation/test 中某 target type 正样本 >= 10，才单独报告该 type 指标；
不足时标记 insufficient_support，不用满分或 0 分作结论。
```

## 8. Dataset 与采样策略

### 8.1 Split 升级

Phase C 增加显式 split manifest，支持按 seed/run 指定 train、validation、test。默认随机 episode split 仅保留给 smoke test。

必须保证：

```text
group key = run_id
同一 run 的所有 env/episode 只能属于一个 split
normalization only fits train
label threshold/config frozen before test
```

### 8.2 类别不平衡处理顺序

处理顺序固定为：

1. 先通过场景覆盖获得真实正样本；
2. 保留现有 train-only `pos_weight`；
3. 若 train positive rate 仍低于 5%，再比较 episode-aware balanced sampler；
4. focal loss 只作为消融项，不直接替换默认 BCE；
5. validation/test 永远保持自然分布，不 oversample。

不对连续相邻窗口进行简单复制式 oversampling，避免模型反复看到高度相关的同一事件。

## 9. Baseline 训练与评估调整

### 9.1 模型结构先冻结

第一轮继续使用当前配置：

```text
GAT hidden = 64
GAT heads = 4
GRU hidden = 128
GRU layers = 1
dropout = 0.2
window = 30s
history = 4 windows
horizon = 120s
```

数据 v2 首次结果出来前不扩大网络，避免无法判断提升来自数据还是模型容量。

### 9.2 分类阈值

同时报告：

```text
F1@0.5
F1@validation-selected-threshold
```

第二个阈值只使用 validation split 选择，然后冻结到 test。PR-AUC 不依赖单一阈值，仍作为 occurrence 主指标。

### 9.3 评估输出

除现有指标外增加：

```text
per-scenario PR-AUC / F1 / positive support
per-type support and metrics
event-level precision / recall / F1
episode-level bootstrap 95% confidence interval
Brier score or calibration error
ABORTED/completion statistics
```

主要结论顺序：

1. test PR-AUC 及 episode-level 95% CI；
2. 相对 no-event/random prevalence baseline 的提升；
3. event-level recall 与误报；
4. node Top-1/Top-3/MRR；
5. time-to-start/duration MAE；
6. type 指标仅在 support 达标时解释。

### 9.4 稳定性复跑

同一冻结 dataset 至少使用 3 个 model seeds 训练：

```text
model seeds = 42, 43, 44
```

最终报告均值、标准差和最佳 checkpoint 对应结果，不只报告单次最高值。

## 10. 预计代码影响范围

确认设计后，预计按阶段修改：

```text
env_asset_cfg/cfg_disturbance.py
src/disturbance.py
src/bottleneck_data.py
tools/build_bottleneck_features.py
tools/bstan_baseline/schema.py
tools/bstan_baseline/dataset.py
tools/bstan_baseline/metrics.py
tools/bstan_baseline/trainer.py
对应 test 文件
```

训练模型主体 `model.py` 第一轮不改。

## 11. 实施阶段

### Phase E0：数据审计工具

1. 汇总 END/ABORTED、completed jobs、事件配对和 scenario 分布；
2. 输出 `data_quality_report.json` 和可读 CSV；
3. 仅接受 collector v0.6 的 `episode_*/env_*` 目录与完整生命周期、扰动真值字段；旧采集版本直接拒绝。

退出条件：能明确指出每个 episode 是否可训练以及拒绝原因。

### Phase E1：采集与扰动真值修正

1. 增加扰动阶段和实际时间/目标字段；
2. 修正 human/gantry DOWN 状态；
3. 将 material 改为可恢复扰动；
4. 按 episode 可复现地随机化事件；
5. 升级 collector version。

退出条件：5 场景 pilot 的事件区间、目标和生命周期全部通过门禁。

### Phase E2：Phase B 标签 v2.3

1. 正确配对扰动区间；
2. 拆分 scenario 与 runtime 特征；
3. 实现 weak label v2.3、process target、warm-up、operation-level 系统影响、impact hold 和 metadata；
4. 增加阈值敏感性与正样本分布报告。

退出条件：pilot 标签可解释，且没有未来泄漏或强制 argmax 假正样本。

### Phase E3：正式采集与图数据集

1. 运行 117 episode 正式矩阵；
2. Phase A/B 批量 validator；
3. 按 seed/run 构建固定 split；
4. 输出数据集 manifest 和质量报告。

退出条件：raw、split、正负样本和 support 门禁全部通过。

### Phase E4：Baseline 重训与结果对比

1. 冻结模型结构训练 3 个 model seeds；
2. 比较 BCE+pos_weight 与可选 balanced sampler；
3. validation 选择 occurrence threshold；
4. 输出总体、分场景、事件级指标和置信区间；
5. 与当前 smoke 结果分开记录，不直接横向宣称性能提升。

退出条件：结果可复现、置信区间可计算、每个结论都有足够 support。

## 12. 服务器执行流程

仍沿用当前协作方式：

1. 本地完成代码、纯 Python 测试和静态检查；
2. 提交并推送 `dev_xwt`；
3. 服务器使用现有 10 episode v0.6 raw 重建 v2.3 Phase B；
4. 本地根据服务器质量报告确认是否进入正式采集；
5. 正式数据通过门禁后构建 dataset 并训练；
6. 服务器结果确认后再回填本文档的实际统计与验收记录。

原始数据、派生数据集和模型产物继续放在 ignored output 目录，不提交到 Git：

```text
output/bottleneck_dataset/<run_id>/
  episode_*/env_*/
  derived_phase_b_v2_3/

output/bottleneck_dataset/experiments/<experiment_id>/
  data_quality_report.json
  dataset/
  models/seed_*/
  aggregate_metrics.json
```

## 13. 本轮需要确认的决策

1. 正式 baseline 数据不再沿用旧 3 单 smoke 数据；先比较 10/15/18 单，当前以 15 单为正式负载候选，18 单保留为压力测试。
2. `ABORTED` 只用于诊断，不进入本轮 BSTAN 训练。
3. material shortage 改为可恢复事件，永久缺料只做压力测试。
4. 正式矩阵采用 13 scenario cells、3 seeds、每格 3 episode，共 117 attempted episodes。
5. train/validation/test 按 seed 和 run 固定拆分，不再随机拆同一 run 的 episode。
6. 正式新标签使用 `bstan_weak_v2_3`，buffer 保留为图输入但不进入第一版预测目标；扰动只提供 cause 信息，不直接充当瓶颈标签。
7. 第一轮冻结 GAT-GRU 结构，先验证数据提升，再做 sampler/focal loss 消融。
8. occurrence 以 PR-AUC 为主，同时报告验证集选阈值后的 F1 和 episode-level 置信区间。
