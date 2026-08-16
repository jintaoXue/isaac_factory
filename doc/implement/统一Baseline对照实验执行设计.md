# 统一 Baseline 对照实验执行设计

## 1. 文档状态

- 状态：已切换到最新 tyx 共享派生口径，等待服务器重建验收
- raw 来源：`collector_version=v0.3` 数据
- 派生来源：`dev_tyx@c101eff` 的 `tools/bn_agg`
- 第一实现目标：BSTAN-style GAT-GRU
- 最终目标：所有 baseline 使用同一 raw、共享 features、标签、split 和评估协议
- 本文不修改或伪造服务器已有 raw 数据

本轮实验不再要求 raw 必须由 `collector_version=v0.6` 生成。v0.6 及
`bstan_weak_v2_3` 的既有实现仅由 Git 历史记录，不在新的 benchmark 主路径中
保留兼容分支、过时设计文档或无效兜底。

> 2026-08-16 修订：第 4-14 节记录了第一轮 canonical 方案的设计过程，
> 现行可执行口径以第 16 节为准。第 15 节结果仅作为历史 smoke，不进入
> PDFormer/BSTAN 正式横向比较。

### 1.1 首轮服务器数据 cohort

首轮 BSTAN 验证只读取服务器上以下两个已采集 run：

```text
output/bottleneck_dataset/new_machine1.0
output/bottleneck_dataset/new_human1.0
```

根据 `dev_tyx@a8b4f38` 的 `batch_bn_collect.sh` 和
`01.瓶颈数据采集规范.md`：

- `new_machine1.0` 为 `disturbance_dim=machine`、`intensity=1.0`；
- `new_human1.0` 为 `disturbance_dim=human`、`intensity=1.0`；
- 每个 run 计划采集 20 个 episode，实际纳入数以 raw quality gate 为准；
- L2 事件每个 episode 重新采样，同时保留该维度的 L0/L1 持续扰动；
- 目录改名不改写 `episode_config.csv` 中的 raw provenance。

这两个 run 用于验证 canonical 链路和 BSTAN 在 machine/human I1
上的 baseline 结果，不代表完整场景矩阵。正式实验还需补入
`none`、`logistics`、`material` 以及预定的多 seed 数据。数据目录名只出现
在执行配置中，不硬编码进 canonical builder。

## 2. 实验原则

baseline 对照实验只允许模型结构变化，以下内容必须固定：

```text
raw episodes
canonical node registry
window features
graph topology
prediction targets
episode split
feature normalization
evaluation metrics
```

统一链路为：

```text
tyx raw v0.3
  -> raw quality gate
  -> canonical aggregation
  -> shared labels
  -> shared episode split
  -> model-specific tensor exporters
  -> training and unified evaluation
```

不得采用以下方式进行主结果对比：

- BSTAN 使用一套标签而 PDFormer 使用另一套标签；
- 随机打散窗口，使同一 episode 同时进入 train 和 test；
- 将扰动发生本身直接作为瓶颈发生标签；
- 某个模型读取其他模型不可见的未来信息；
- 通过修改 `collector_version` 声称旧 raw 满足新采集协议。

## 3. Raw 输入契约

本轮 canonical builder 只接受 tyx `v0.3` raw：

```text
<run_id>/
  run_manifest.json
  episode_<id>/
    env_<id>/
      episode_config.csv
      disturbance_log.csv
      resource_event_log.jsonl
      job_trace.csv
      buffer_event_log.csv
      route_transport_task.csv
      material_inventory_log.csv
```

主要可用证据：

| raw 文件 | canonical 用途 |
| --- | --- |
| `episode_config.csv` | 订单、工艺、资源配置、扰动配置和 provenance |
| `resource_event_log.jsonl` | 资源状态时间线、任务和工件关联 |
| `job_trace.csv` | 排队、加工、离开、完工和吞吐量 |
| `buffer_event_log.csv` | buffer 占用、容量和 supporting materials |
| `route_transport_task.csv` | 运输等待、路线和载体状态 |
| `material_inventory_log.csv` | 库存位置、短缺和物料进度 |
| `disturbance_log.csv` | 扰动上下文、目标和实际时间区间 |

## 4. Phase A：Raw 质量门禁

### 4.1 完整 episode 判定

tyx raw 没有 `episode_lifecycle.csv`，因此不能使用最后一条日志直接认定正常结束。
一个 episode 只有满足以下条件才能进入 benchmark：

1. 必需 raw 文件存在且可解析；
2. `episode_config.csv` 恰好包含一条配置；
3. raw 中 run、env、episode 标识一致；
4. 时间戳单调且不超过推断的 episode end；
5. `stage_complete` 的唯一 job 数等于订单总数；
6. episode 尾部不存在无法闭合的任务或运输记录；
7. 扰动开始和结束记录可以配对；
8. 工艺机器、buffer 和日志节点可以解析到 canonical node registry。

不能证明正常完成的 episode 标记为 `rejected`，不能通过默认值强行纳入训练。

### 4.2 质量产物

```text
canonical_factory_bn_v1/
  raw_quality_report.json
  accepted_episodes.csv
  rejected_episodes.csv
```

报告至少包含：

- 完工数量和期望订单数；
- episode end 的推断依据；
- 缺失文件和悬空记录；
- 资源节点覆盖率；
- 扰动配对结果；
- 首次观测时间；
- 可用于训练的窗口范围。

## 5. Phase B：Canonical 聚合

以 tyx 的 `tools/bn_agg` 模块化结构为基础，生成与模型无关的数据层。

### 5.1 状态规范化

不直接信任 v0.3 中已写入的 `from_state/to_state`，而是根据
`raw_from_state/raw_to_state` 做一次确定性的 canonical 映射，修正：

- machine `invalid`；
- human `working_disturbance_absent`；
- machine `waiting_*`；
- machine `materialReadyFor_*`；
- human、robot 和 gantry 的 wait/processing 语义。

映射表必须集中定义并写入 dataset manifest，不在多个 exporter 中重复实现。

### 5.2 节点注册表

节点不能只由发生过状态变化的资源决定。canonical node registry 从以下信息合并：

- `process_time_config.machine`；
- `human_config`；
- `robot_config`；
- `gantry_config`；
- buffer 日志及 supporting materials；
- 实际资源事件和运输载体。

全数据集使用固定节点顺序。每个 episode 输出 `node_mask` 和
`observed_mask`，用于区分不存在、未观测和正常空闲。

### 5.3 动态图构建

图边从 episode 配置和 raw 元数据生成，不保留硬编码 `PROCESS_CHAIN`：

```text
process_flow
buffer_supply
human_capability
robot_capability
gantry_capability
self_loop
```

当前工厂布局一致时，各 episode 应得到相同 topology signature。未来布局变化时，
通过 node mask 和 episode adjacency 表达，不修改模型标签定义。

### 5.4 时间窗口与缺失处理

第一版固定：

```text
window_size = 30 logic seconds
stride = 30 logic seconds
input_windows = 4
input_history = 120 logic seconds
prediction_horizon = 120 logic seconds
warmup = 120 logic seconds
```

由于 v0.3 没有完整 t=0 快照：

- warmup 以前不生成训练 anchor；
- 不用未来第一条状态反向填充过去；
- 缺失观测由 mask 表达；
- buffer 和 material 使用最近一次已观测值前向保持；
- 无法确定的初始状态不伪装为真实观测。

### 5.5 Canonical 产物

```text
canonical_factory_bn_v1/<episode_key>/
  node_registry.csv
  edge_table.csv
  window_feature_table.csv
  bottleneck_event.csv
  bottleneck_label.csv
  episode_summary.json
```

## 6. Phase C：共享瓶颈标签

新标签版本统一命名为：

```text
dataset_contract = canonical_factory_bn_v1
label_version = factory_bn_weak_v1
target_node_category = process
```

不再使用带某个模型名称的共享标签版本号。

### 6.1 事件定义

候选 process 节点需要依次通过：

1. absolute score gate；
2. relative margin gate；
3. system impact gate；
4. consecutive-window persistence gate。

system impact 使用预测时刻可以观测到的运行证据，例如：

- total WIP growth；
- completed operation throughput drop；
- 在数据支持充分时增加 cycle-time growth。

### 6.2 扰动语义

扰动不直接创建 bottleneck event。扰动字段只作为：

- 当前可观测上下文；
- root-cause candidate；
- 分场景评估维度；
- 事件后分析依据。

只有扰动引起了满足局部压力、持续性和系统影响门禁的运行结果时，才形成正标签。

### 6.3 预测目标

所有学习模型共享：

| 任务 | 标签 |
| --- | --- |
| 瓶颈发生 | `will_bottleneck` |
| 瓶颈节点 | `future_bottleneck_node` |
| 节点类型 | `future_bottleneck_type` |
| 开始时间 | `time_to_start` |
| 持续时间 | `duration` |
| 监督有效性 | `label_observed` |

事件级 severity 在 tyx v0.3 中缺少可靠变化，暂不作为主 benchmark 任务。

## 7. Phase D：共享 Dataset 与 Split

全局产物：

```text
benchmark_v1/
  dataset_manifest.json
  split_manifest.json
  canonical_tensors.npz
```

split 规则：

- 按 episode 切分；
- 同一 episode 只能属于一个 split；
- 尽量按 disturbance scenario 和 seed 分层；
- 标准化参数只能在 train split 上拟合；
- 所有模型读取同一个 `split_manifest.json`；
- manifest 固定 episode key、source commit、配置摘要和文件校验值。

首轮 machine/human I1 计划共 40 个 episode，实际 split 只使用通过
raw quality gate 的 episode。该 cohort 可用于跑通 baseline 并比较两个已覆盖场景，
但不足以支撑全扰动矩阵或跨强度结论。

## 8. 最终 Baseline 矩阵

### 8.1 主结果模型

| 编号 | 模型 | 类型 | 实验作用 |
| --- | --- | --- | --- |
| B2 | XGBoost | 非深度、非图 | 衡量特征工程本身的效果 |
| B3 | LSTM | 深度时序、非图 | 判断图结构是否必要 |
| B4 | GCN-GRU | 传统时空图 | 普通图卷积与循环网络基线 |
| B5 | BSTAN-style GAT-GRU | 图注意力时序 | 判断 attention graph 的收益 |

最终 baseline 只包含 B2-B5，不实现 B0 No-event 或 B1 Persistence 模型。类别
prevalence、always-negative 和 train-majority 仍可作为指标解释中的统计参照，但不编号、
不训练，也不进入 baseline 模型主表。

PDFormer 和最终 proposed model 使用相同共享任务契约与 B2-B5 比较，但不再编号为
baseline。最终 baseline 主表固定为：

```text
B2  XGBoost
B3  LSTM
B4  GCN-GRU
B5  BSTAN-style GAT-GRU
```

### 8.2 消融实验

以下模型进入消融表，不与外部 baseline 混写：

| 消融 | 目的 |
| --- | --- |
| GRU-only | 与 GCN-GRU、GAT-GRU 配对，验证图结构 |
| GAT-only | 验证时间编码贡献 |
| GAT-GRU without global features | 验证系统级特征贡献 |
| GAT-GRU without disturbance context | 验证扰动上下文贡献 |
| Proposed without key module | 验证主模型新增模块贡献 |

## 9. 模型输入公平性

所有主结果模型使用相同历史范围、标签和 split。

- XGBoost：标量头读取各节点历史的 last/mean/max/delta、mask、global 和 jobs；
  A.1 cell 头读取目标节点历史摘要、全图聚合、future offset 和 jobs，不读取 adjacency；
- LSTM：每个时刻展平固定节点顺序，不读取 adjacency；
- GCN-GRU：每个时刻 GCN，再沿时间维度 GRU；
- GAT-GRU：每个时刻 GAT，再沿时间维度 GRU；
- PDFormer：读取相同 node features、graph、mask 和标签。

XGBoost 不直接为每个 future cell 复制完整 `T x N x F`，否则 512 窗全量目标会产生
不可接受的内存开销。A.1 主任务采用共享 cell classifier。每个训练 cell 表示
`(sample, future_offset, node)`，特征由节点与全图历史统计、future offset、
jobs remaining/total 和有效性 mask 组成。当前实现具体使用目标节点的
last/mean/max/delta、最后窗口的全图 mean/max、future offset 周期编码和 jobs 状态。
训练保留全部正 cell，并使用固定 seed 对负 cell 下采样；validation/test 必须恢复完整
有效 `[K,N]`，不得下采样后计算指标。

模型专属 auxiliary loss 必须单独报告。不能依靠它获得的额外监督与其他模型的主结果直接比较。

## 10. 统一评估协议

| 任务 | 主要指标 |
| --- | --- |
| occurrence | PR-AUC、ROC-AUC、F1、precision、recall |
| node | Top-1、Top-3、MRR |
| type | accuracy、macro-F1 |
| time-to-start | MAE |
| duration | MAE |

约束：

- F1 threshold 只允许通过 validation split 选择；
- test split 只能使用固定 threshold；
- node、type 和 regression 只在对应正样本上计算；
- 同时报告正负样本数和 episode 数；
- 主结论以多 seed 均值和标准差为准。

## 11. 代码修改范围

确认设计后，预计修改范围如下：

```text
tools/bn_agg/*
  -> canonical aggregation、state mapping、shared labels

tools/audit_bottleneck_data.py
  -> tyx v0.3 raw quality gate

tools/factory_baselines/dataset.py
  -> 读取 canonical dataset 和 shared split

tools/factory_baselines/graph_builder.py
  -> 读取 canonical edge table

PDFormer/factory_bn/export_dataset.py
PDFormer/factory_bn/dataset.py
  -> 读取 canonical dataset，删除随机窗口 split

source/isaaclab_tasks/test/
  -> raw audit、state mapping、label、split 和 exporter tests
```

新 benchmark 主路径稳定后，删除不再执行的 v0.6 专属兼容逻辑；不增加
`if v0.3 ... else v0.6 ...` 形式的长期兜底。

## 12. 实施顺序

### Step 1：BSTAN 优先

先完成：

```text
tyx raw v0.3
  -> audit
  -> canonical features and labels
  -> shared split
  -> BSTAN-style GAT-GRU training
  -> unified metrics
```

BSTAN 优先的原因：

- 当前已有可运行的 GAT-GRU 模型和多任务输出头；
- 可最早暴露 canonical 数据契约的问题；
- 跑通后 LSTM 和 GCN-GRU 只需替换 encoder；
- 不需要先处理 PDFormer 更复杂的 exporter 和训练接口。

### Step 2：B2 XGBoost

在同一 dataset 上实现 XGBoost。先完成 A.1 完整 occupancy 输出，再接入共享 A.3 和
近窗辅助目标。验证通过后冻结 B2 命令与产物格式。

### Step 3：B3 LSTM

复用 B5 的预测头和 loss，以展平固定节点顺序的每时刻特征作为 LSTM 输入，不读取
adjacency。验证通过后冻结 B3 命令与产物格式。

### Step 4：B4 GCN-GRU

复用 BSTAN 训练器和预测头实现 GCN-GRU，并完成 GCN 与 GAT 对比。

### Step 5：B5 BSTAN-style GAT-GRU

保留已实现的 GAT-GRU，并在七点共享设计决策落地后，用最终 shared split、mask、标签和
evaluator 重建正式结果。

### Step 6：主模型对齐

最后将 PDFormer 改为读取相同 dataset 和 shared split，再与 B2-B5 比较。

## 13. 验收标准

第一阶段 BSTAN 完成需满足：

1. 原始 tyx raw 文件没有被修改；
2. rejected episode 有明确理由；
3. canonical node registry 和 topology 可复现；
4. 标签不由扰动记录直接生成；
5. train、validation、test episode 集合无交集；
6. BSTAN 可以训练、保存 checkpoint 并完成 test evaluation；
7. manifest 能追溯 raw run、source commit、标签版本和 split；
8. 测试覆盖状态映射、缺失观测、右删失、split 隔离和数据张量形状。

完成以上验收后，再开始新增其他 baseline，避免每个模型各自形成一套数据处理逻辑。

## 14. 第一轮实现落点

当前第一轮实现采用以下唯一版本：

```text
raw_contract_version = tyx_raw_v0.3
canonical_contract_version = canonical_factory_bn_v1
label_version = factory_bn_weak_v1
dataset_version = bstan_canonical_dataset_v1
```

已完成的本地代码范围：

- strict v0.3 raw 完整性审计；
- raw resource state 的 canonical 重映射；
- 未产生状态事件的配置资源补齐及 observation mask；
- buffer supporting materials 的 canonical graph config；
- canonical features、events、labels 和 provenance checksum；
- episode-level shared split manifest；
- shared factory baseline dataset、B5 训练、checkpoint 和 test evaluation；
- validation 选择 occurrence F1 threshold，test 固定使用该 threshold；
- 一条命令执行 audit、canonical build 和 shared baseline dataset build。

首轮服务器 raw 与 BSTAN smoke 验收已通过，可以在同一 canonical dataset
和 split 上开始后续 baseline 适配。

## 15. 首轮服务器验收结果

### 15.1 Raw 与 dataset

2026-08-16 在服务器 `dev_xwt` 上对以下 raw 执行验收：

```text
new_machine1.0
new_human1.0
```

Raw quality gate 结果：

```text
attempted = 40
trainable = 40
rejected = 0
```

Canonical factory baseline dataset：

```text
dataset_contract = canonical_factory_bn_v1
dataset_version = bstan_canonical_dataset_v1
label_version = factory_bn_weak_v1
total_samples = 33210
positive_samples = 178
positive_rate = 0.005360
episode_split = 28 / 6 / 6
sample_split = 23092 / 5066 / 5052
```

Split 无 episode 交叉，validation/test 均含正负样本。正例分布为：

| 范围 | 正例窗口 | 总窗口 | 正例率 |
| --- | ---: | ---: | ---: |
| train | 146 | 23092 | 0.6323% |
| validation | 12 | 5066 | 0.2369% |
| test | 20 | 5052 | 0.3959% |
| human I1 | 154 | 14318 | 1.0756% |
| machine I1 | 24 | 18892 | 0.1270% |

40 个 episode 中 22 个产生至少一个正例窗口。machine 正例明显少于
human，这是当前数据与 weak label 门禁共同形成的实际分布，不通过降低
阈值人为追平。

### 15.2 BSTAN-style GAT-GRU smoke 结果

```text
best_epoch = 13
epochs_trained = 28
validation_pr_auc = 0.023736
test_pr_auc = 0.081533
validation_selected_threshold = 0.040635
test_f1 = 0.079051
```

Occurrence 结果：

| split | PR-AUC | prevalence baseline | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.023736 | 0.002369 | 0.0513 | 0.3333 | 0.0889 |
| test | 0.081533 | 0.003959 | 0.0429 | 0.5000 | 0.0791 |

Test PR-AUC 约为当前 test prevalence 的 20.6 倍，说明模型已学到非随机信号；
但 test 只有 20 个正窗口，且来自 4 个 episode，该倍数和 F1 不能作为
稳定泛化结论。

条件于真实正例的 test 辅助任务结果：

```text
node_top1 = 0.75
node_top3 = 0.90
node_mrr = 0.8367
time_to_start_mae_s = 28.16
duration_mae_s = 7.89
```

这些 node/type/regression 指标只在 ground-truth 正样本上计算，不是端到端指标。
type macro-F1 仅 0.1778，当前正标签的资源类型支持高度集中，不能用
0.80 accuracy 声称已具备多类型泛化能力。

### 15.3 本轮结论

本轮证明了以下工程链路：

```text
tyx raw v0.3
  -> strict raw quality gate
  -> canonical features and weak labels
  -> episode-level shared split
  -> BSTAN training/checkpoint/test evaluation
```

它不证明扰动因果、跨 seed/布局泛化或 machine 单场景性能。正式实验需补齐
场景矩阵和多 seed，并在同一 shared split 上增加其他 baseline。

## 16. 现行共享实现（dev_tyx c101eff）

### 16.1 单一派生入口

当前主路径不再运行 `build_bottleneck_features.py` 或
`build_canonical_benchmark.py`，统一执行：

```text
tyx raw v0.3
  -> audit_bottleneck_data.py
  -> dev_tyx tools/bn_agg
  -> shared_bn_agg_v1
  -> model-specific causal sequence exporter
```

共享版本固定为：

```text
derived_contract = tyx_bn_agg_v1
label_version = tyx_bn_agg_event_v1
shared baseline tensor dataset = bstan_tyxbn_dataset_v2
prediction target = factory_a1a3_remain_v1
window_size = 60 s
input_windows = 12
horizon = 180 s
score_threshold = 0.55
min_event_windows = 1
```

`build_shared_benchmark.py` 直接调用 `bn_agg.pipeline.process_env_dir`，并固定
`closed_windows_only=True`。只保留完整窗口，且 BSTAN 只接收
`label_horizon_ready=1` 的标签，从而排除 episode 尾部预测区间不完整的样本。

### 16.2 与 `04.期望输出.md` 对齐的预测目标

BSTAN 不重新定义评分或标签，只消费 tyx 的 21 个窗口特征、
`bottleneck_score_s`、事件表、`job_kpi` 和 `bottleneck_label`。输入严格使用
目标窗口之前的 12 个窗口 `[t-12, t)`，不把目标窗口特征放进历史序列。

按照 `dev_tyx@c101eff` 根目录的 `04.期望输出.md`，当前模型目标固定为：

| 期望输出 | BSTAN 监督与输出 | 与 PDFormer 关系 |
| --- | --- | --- |
| A.1 瓶颈 0/1、开始、持续、工位 | 主头预测从当前时刻到剩余工单清零的 `score/hot[K,N]` 及 `remain_len`；连通 hot 段还原事件 | 同一标签与事件还原规则 |
| A.2 setting×工位热力图 | 对多个 setting 的 A.1 预测按工位聚合 | 同属实验后处理，不增加网络头 |
| A.3 瓶颈原因 | `root_cause_reason` 十分类，空值为 `-1` 并 mask loss | 同一类别顺序 |
| 近 180 秒辅助任务 | `will_bottleneck`、mark node、time-to-start | 与 PDFormer 辅头一致 |
| B.1/B.2/B.3 | 本轮不训练 | PDFormer 当前也未实现产品/工序预测头 |

A.1 hot 定义与 tyx 保持一致：`score>=0.55`，或当前节点出现
`is_turning_point` / `disturbance_active_s`。未来长度上限为 512 个 60 秒窗口，
但 loss 只重点监督前 60 窗并按时间衰减。checkpoint 统一按 validation
`hot_f1` 选择，PR-AUC 仅作为 180 秒 will 辅助指标。

图边与最新 PDFormer 的工厂先验保持一致：工艺链、同机型 workstation、
buffer-machine affinity、agent-machine、同类 gantry/robot 和 self-loop。
BSTAN 仍采用 episode 级 70/15/15 split，并只用 train split 拟合标准化参数。

tyx 标签不提供 `severity_weak`，因此 severity head、loss、指标和数据字段已从
现行 BSTAN 路径删除，没有默认值或兼容兜底。

直接 type/duration 头也已删除。工位由 mark 或 A.1 事件节点给出，duration
由 hot 连通段给出，避免同一含义出现两套不一致标签。

### 16.3 历史 smoke 的处理

第 15 节的 33,210 样本和模型指标来自旧 `canonical_factory_bn_v1`，只能证明
训练代码可运行。共享口径重建后，样本数、正例率、节点数和模型指标都会变化，
不得把两次结果放在同一正式结果表中比较。

### 16.4 待与 tyx 讨论的问题

| 问题 | 当前 BSTAN 处理 | 建议统一动作 |
| --- | --- | --- |
| PDFormer 按窗口随机切分，同 episode 可跨 split | episode 级切分 | 两边统一读取同一 split manifest |
| episode 尾部 horizon 不完整 | closed windows + `label_horizon_ready` | PDFormer 离线训练也固定该门禁 |
| 时间线在首次事件前假定 IDLE | 保持 tyx 现状以确保口径一致 | 采集初始快照或引入明确 observed mask |
| L2 扰动区间直接并入瓶颈事件 | 保持 tyx 现状以确保标签一致 | 区分 disturbance cause 与 operational bottleneck label |
| 工艺链、buffer affinity 和 material consumer 硬编码 | 与 PDFormer 相同 | 后续从 episode 配置生成并版本化拓扑 |
| A.3 原因标签稀疏且多数类明显 | 输出 cause accuracy、macro-F1 和 majority baseline | 标签不足时只作分析字段，不宣称可用原因分类 |

主预测目标不一致的问题已在 BSTAN 中修复。前两项也已在 BSTAN exporter
修复，但正式公平对照仍要求 PDFormer 采用同一 split 与 horizon gate；其余项
不在本轮私自改变，避免再次形成两套标签口径。

## 17. B2 XGBoost 实现与服务器验收

### 17.1 实现状态

B2 已接入与 B5 相同的 `dataset.pt`、manifest 和 episode split，不重新读取 raw 或
派生另一套标签。实现入口：

```text
tools/train_b2_xgboost.py
tools/factory_baselines/b2_xgboost.py
```

B2-B5 的共享 dataset、schema、graph 和 metrics，以及 B3-B5 共用的 heads、loss、
trainer，均位于 `tools/factory_baselines/`。旧 `tools/bstan_baseline/` 已删除，不保留
alias 或兼容转发；历史路径只通过 Git 查询。

模型包含：

| Head | XGBoost 任务 | 共享目标 |
| --- | --- | --- |
| occurrence | binary classifier | 近 180 秒 `will_bottleneck` |
| mark node | multiclass classifier | 第一个未来瓶颈节点 |
| time-to-start | regressor | 第一个未来事件开始时间 |
| cause | multiclass classifier | A.3 十分类 |
| remain length | regressor | 到剩余工单清零的窗口数 |
| remain hot | shared binary cell classifier | A.1 `hot[K,N]` |
| remain score | shared cell regressor | A.1 `score[K,N]` |

标量头和 cell 头都不读取 adjacency，因此 B2 保持“非图模型”属性。cell 训练保留全部
正例，负例按固定 seed 下采样；validation/test 对全部有效未来 cell 进行推理和评价。
occurrence threshold 和 occupancy threshold 都只在 validation 上选择，后者通过流式
概率直方图计算，避免一次性持有全部未来概率。

本地已完成：

```text
Python compile passed
16 focused dataset/B2/B3/B4/B5 unit tests passed
XGBoost 3.2.0 synthetic end-to-end smoke passed
B3/B4/B5 one-epoch synthetic train/evaluate smoke passed
```

本地 smoke 只验证代码链路，不作为实验结果。

### 17.2 服务器命令

先确认服务器环境安装 XGBoost：

```bash
python -c "import xgboost; print(xgboost.__version__)" || \
  python -m pip install xgboost
```

使用已经生成的共享 dataset：

```bash
DATA_ROOT="$PWD/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
BENCHMARK_DIR="$DATA_ROOT/experiments/canonical_bstan_machine_human_v1"
DATASET_DIR="$BENCHMARK_DIR"
MODEL_DIR="$BENCHMARK_DIR/models/b2_xgboost_seed42"

python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/train_b2_xgboost.py \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$MODEL_DIR" \
  --seed 42 \
  --n_jobs 8
```

如果现行共享 dataset 实际位于 `bstan_dataset_v*` 子目录，应将 `DATASET_DIR` 指向包含
`dataset.pt` 和 `dataset_manifest.json` 的那个目录。运行前可验证：

```bash
test -f "$DATASET_DIR/dataset.pt"
test -f "$DATASET_DIR/dataset_manifest.json"
```

### 17.3 预期产物

```text
models/b2_xgboost_seed42/
  config.json
  run_summary.json
  metrics.json
  metrics_validation.json
  metrics_test.json
  predictions_validation.csv
  predictions_test.csv
  occupancy_events_validation.csv
  occupancy_events_test.csv
  confusion_matrix_validation.csv
  confusion_matrix_test.csv
  occurrence.json
  node.json
  time_to_start.json          # 非常量 head 时存在
  cause.json
  remain_len.json             # 非常量 head 时存在
  remain_hot.json
  remain_score.json
```

常量 head 不生成空模型文件，其常量值直接记录在 `config.json`。服务器验收至少检查：

```bash
python -m json.tool "$MODEL_DIR/run_summary.json"
python -m json.tool "$MODEL_DIR/metrics.json"
```

## 18. B3-B5 共享 PyTorch 实现

### 18.1 文件结构

```text
tools/factory_baselines/
  b3_lstm.py
  b4_gcn_gru.py
  b5_gat_gru.py
  torch_heads.py
  torch_losses.py
  torch_trainer.py

tools/train_b3_lstm.py
tools/train_b4_gcn_gru.py
tools/train_b5_gat_gru.py
tools/evaluate_torch_baseline.py
```

三种模型只替换 encoder，统一输出 `node_hidden[B,N,H]`，随后使用同一个
`FactoryPredictionHeads` 产生 occurrence、mark、time-to-start、remain score/hot、
remain length 和 cause。三者共用相同 loss、validation checkpoint 指标、threshold、
metrics 和产物格式。

### 18.2 Encoder 定义

| Baseline | Encoder | 是否读取 adjacency |
| --- | --- | --- |
| B3 LSTM | 每个时刻展平固定节点顺序，LSTM 编码历史；graph hidden 与可学习 node embedding 组合为 node hidden | 否 |
| B4 GCN-GRU | 每个窗口两层对称归一化 GCN，之后每节点共享 GRU | 是 |
| B5 GAT-GRU | 每个窗口两层 dense multi-head GAT，之后每节点共享 GRU | 是 |

B3 的 forward 显式丢弃 adjacency；测试要求改变 adjacency 后输出逐值不变。B4 测试要求
full adjacency 与 identity adjacency 产生不同 node hidden。B4 与 B5 的隐藏维度默认均为
64 的空间层和 128 的 GRU，B3 默认 128 LSTM hidden、128 node hidden。

### 18.3 Checkpoint 契约

新 PyTorch checkpoint 必须包含：

```text
model_kind = b3_lstm | b4_gcn_gru | b5_gat_gru
model_config
model_state_dict
loss_config
train_config
dataset_manifest_sha256
```

`evaluate_torch_baseline.py` 从 `model_kind` 恢复对应模型，不根据目录名猜测。重构前的 B5
checkpoint 不含 `model_kind`，不在当前主路径增加兼容兜底；正式服务器验证需要重新训练
B5。raw 无需重采，但聚合规则更新后必须从 raw 重建 `shared_bn_agg_v1` 和 `dataset.pt`，
不能继续使用第 15 节旧 canonical 聚合产物。

### 18.4 统一服务器训练命令

```bash
DATA_ROOT="$PWD/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"
DATASET_DIR="$DATA_ROOT/experiments/canonical_bstan_machine_human_v1"
MODELS_DIR="$DATASET_DIR/models"

python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/train_b3_lstm.py \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$MODELS_DIR/b3_lstm_seed42" \
  --device cuda:0 \
  --seed 42

python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/train_b4_gcn_gru.py \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$MODELS_DIR/b4_gcn_gru_seed42" \
  --device cuda:0 \
  --seed 42

python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/train_b5_gat_gru.py \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$MODELS_DIR/b5_gat_gru_seed42" \
  --device cuda:0 \
  --seed 42
```

B2 使用第 17 节命令。四个模型均输出 `run_summary.json`、`metrics.json`、validation/test
predictions、occupancy events 和 confusion matrix；B3-B5 另外输出 `best.pt`、`last.pt`
和 `history.csv`。

### 18.5 验收顺序

服务器统一测试时按 B2、B3、B4、B5 顺序执行。先分别使用 1 epoch smoke 输出目录验证，
再运行正式配置；不要让 smoke 与正式结果共用目录。完成后检查：

```text
baseline_id/model_kind 正确
dataset manifest hash 相同
episode split 相同
validation/test 样本数相同
所有模型输出 A.1/A.3 共享字段
```

## 19. 服务器批处理脚本

根目录提供两个可版本化的命令入口：

```text
batch_factory_baseline_build.sh  raw audit -> bn_agg -> derived -> shared dataset
batch_factory_baseline_train.sh  shared dataset -> B2/B3/B4/B5
```

聚合与 dataset 重建：

```bash
./batch_factory_baseline_build.sh MH
```

`MH` 展开为 `new_machine1.0` 与 `new_human1.0`。脚本默认删除这两个 raw run 下旧的
`shared_bn_agg_v1`，使用当前分支的 `dev_tyx@c101eff` 同源 `bn_agg` 重新派生，再写入：

```text
output/bottleneck_dataset/experiments/shared_bstan_machine_human_v1/
  raw_quality_report.json
  shared_build_summary.json
  dataset.pt
  dataset_manifest.json
  split_manifest.json
```

该目录名与旧 `canonical_bstan_machine_human_v1` 分开，防止旧 label、旧 checkpoint 和
新共享口径混用。需要自定义实验目录或 raw 组合时使用：

```bash
BENCHMARK_TAG=my_experiment ./batch_factory_baseline_build.sh \
  new_machine1.0 new_human1.0
```

先执行四模型 smoke：

```bash
RUN_MODE=smoke ./batch_factory_baseline_train.sh ALL
```

smoke 通过后执行正式训练：

```bash
./batch_factory_baseline_train.sh ALL
```

也可以只训练一个模型，例如 `./batch_factory_baseline_train.sh B5`。所有模型从同一
`dataset.pt` 和 `split_manifest.json` 读取监督与划分，脚本不在训练阶段重新聚合或改标签。
