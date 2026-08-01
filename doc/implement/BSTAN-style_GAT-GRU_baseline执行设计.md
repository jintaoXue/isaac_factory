# BSTAN-style GAT-GRU Baseline 执行设计

## 1. 文档状态

- 状态：实施中（Phase A、Phase B 已验收）
- 目标：先跑通一套可复现、可训练、可评估的 BSTAN-style GAT-GRU baseline
- 本文范围：数据采集修正、离线特征与标签、图数据集、模型、训练和验收
- 本文不包含：具体代码实现、完整动态图库、预测反馈调度、主模型实现

本文中的 baseline 是根据 BSTAN 思路实现的 `GAT + GRU` 制造瓶颈预测基线。由于论文没有官方开源代码，后续论文和代码中应称为：

```text
BSTAN-style GAT-GRU baseline
```

或：

```text
BSTAN-inspired GAT-GRU baseline
```

不能写成对原论文的严格复现。

## 2. 最小闭环目标

第一版固定以下时间尺度：

```text
window_size = 30 logic seconds
stride = 30 logic seconds
input_windows = 4
input_history = 120 logic seconds
prediction_horizon = 120 logic seconds
min_event_windows = 2
min_event_duration = 60 logic seconds
```

模型输入和输出：

```text
Input:
  X: [batch, 4, num_nodes, num_features]
  A: [batch, num_nodes, num_nodes]
  node_mask: [batch, num_nodes]

Output:
  will_bottleneck
  future_bottleneck_node
  future_bottleneck_type
  time_to_start
  duration
  severity_weak
```

第一版验收不是追求高指标，而是完成：

```text
Isaac Sim raw logs
  -> window features and labels
  -> static manufacturing graph
  -> sequence samples
  -> GAT-GRU training
  -> validation/test metrics
  -> reproducible checkpoint and prediction file
```

## 3. 当前代码审阅结论

### 3.1 已具备的链路

当前采集器在每个逻辑步结束时执行：

```text
manager.step
  -> disturbance_injector.step
  -> action masker refresh
  -> time_step += 1
  -> perception_manager.step
  -> bottleneck_collector.step
```

采集器只读 `env_state_action_dict`，不会改变调度逻辑，挂载位置适合作为训练数据采集入口。

当前按以下层级输出：

```text
output/bottleneck_dataset/<run_id>/
  run_manifest.json
  episode_<id>/
    env_<id>/
      episode_config.csv
      episode_lifecycle.csv
      disturbance_log.csv
      resource_event_log.jsonl
      job_trace.csv
      buffer_event_log.csv
      route_transport_task.csv
      material_inventory_log.csv
```

已有离线脚本 `build_bottleneck_features.py` 可以生成：

```text
window_feature_table.csv
bottleneck_label.csv
bottleneck_event.csv
job_kpi.csv
pipeline_summary.json
```

### 3.2 初始代码不能直接训练 GAT-GRU 的问题

| 优先级 | 问题 | 影响 |
|---|---|---|
| P0 | 默认不写资源初始状态 | 从未改变状态的节点不会出现在 timeline，节点集合不稳定 |
| P0 | buffer reset 时只缓存、不写 `t=0` 快照 | 第一个窗口可能使用第一次事件后的 occupancy 反推初值 |
| P0 | 没有明确 episode end 记录 | 离线脚本用最后一个事件推断结束时间，可能截短尾部状态 |
| P0 | 窗口特征和标签没有 `episode_id` | 多 episode 合并后主键冲突，无法安全切分 |
| P0 | 当前未来标签以 `window_start_s` 为锚点 | 输入包含当前窗口完整数据时，可能把窗口内部事件当成未来事件 |
| P0 | episode 尾部 horizon 不完整时仍生成负标签 | 未来不可观测的样本会被错误标成 `will_bottleneck=0` |
| P0 | 还没有节点目录、边表和序列样本索引 | 不能构造固定形状的图序列 |
| P0 | `waiting_*` 当前映射为 `BLOCKED` | 实际更接近等待物料/被预留，可能反转上下游传播含义 |
| P1 | `route_delay`、material shortage 被复制到所有节点 | 空间差异被抹平，GAT 难以学习局部关系 |
| P1 | 缺少 `blocked_ratio`、`starved_ratio`、`output_rate` 等字段 | 与 PRD P0 特征集合不完全一致 |
| P1 | disturbance 表头是 `severity`，写入字段却是 `intensity` | 扰动强度被 CSV writer 丢弃 |
| P1 | 当前 bottleneck type 等于 resource type | 类型任务与节点任务高度重复，暂时只能作为兼容输出 |
| P1 | 没有相关单元测试和可用样例数据 | 修改后容易产生静默 schema 回归 |

### 3.3 实施验收记录

Phase A 已于 2026-08-01 使用服务器真实仿真完成 smoke 验收：

```text
run_id = 2026-08-01_15-45-47_seed42
collector_version = v0.4
num_envs = 1
episodes = 6
completed_jobs = 3 / episode
resource_nodes = 19 / episode
episode lifecycle = 6/6 START=1, END=1, production_done=1
```

6 个 episode 均可被原有离线脚本解析，订单 makespan 为 2118-2247 logic seconds，未发现缺失 episode 或未完成 job。该次运行证明 Phase A schema 和真实 Isaac Sim 运行链路可用。

Phase B 随后使用同一 raw run 完成服务器验收：

```text
window_size = 30s
stride = 30s
prediction_horizon = 120s
feature nodes = 37 / window（19 个事件资源 + 18 个 buffer）
feature rows = 16169
label rows = 437
observed labels = 413
censored tail labels = 24（每 episode 4 个）
merged bottleneck events = 13
positive labels = 28 / 413（6.78%）
```

6 个 episode 均通过 Phase B 主键唯一性、输入有限值和删失标签门禁。该 smoke dataset 同时包含正负样本，但规模和场景多样性仍只适合管线验收，不作为正式训练数据。

## 4. Baseline 范围决策

### 4.1 第一版节点

使用稳定的资源级节点，不使用每个 job 的 material 实例作为节点：

```text
productive machine workstation
active gantry
human
transport robot
buffer/storage
```

选择理由：

- 与当前 `resource_event_log` 和 `buffer_event_log` 粒度一致；
- 节点数量在 episode 内固定；
- 可以预测 machine、human、transport、buffer 瓶颈；
- material 当前是 job-level 件实例，数量随订单变化，不适合第一版固定图；
- material shortage 第一版作为受影响 station/buffer 的节点特征，不作为独立节点。

以下机器不进入 baseline 图：

```text
corresponding_process_task == ["none"] 的未使用机器
未激活的 gantry
当前 scenario 中不存在的 human/robot
```

跨 scenario 训练时，全局节点表由 dataset 中所有 episode 的启动配置或配置上界生成。这里只读取静态资源声明，不读取标签和未来调度结果，因此不构成目标泄漏。每个 episode 通过 `node_mask` 屏蔽不存在的节点。

### 4.2 第一版边

采用静态先验图，保持 baseline 与后续动态主模型的差异：

| 边类型 | 方向 | 构建方式 |
|---|---|---|
| `process_flow` | 双向 | 按产品工艺顺序连接相邻 processing station 的所有可用 workstation |
| `buffer_supply` | 双向 | 按 buffer 支持物料和工序 required materials 连接 buffer 与 station |
| `human_capability` | 双向 | human 与可人工参与的生产 station 相连 |
| `robot_capability` | 双向 | transport robot 与生产 station/buffer 相连 |
| `gantry_capability` | 双向 | active gantry 与生产 station/buffer 相连 |
| `self_loop` | 自环 | 每个有效节点添加 |

第一版 GAT 使用这些边的并集，不把 edge type 输入 attention。边表仍保留 `edge_type`，方便后续做 relational ablation。

第一版不加入：

```text
未来 task assignment
未来 transport task
动态 agent edge
learned adaptive adjacency
route segment / intersection node
```

这样可以避免未来信息泄漏，也能把动态图能力保留为主模型的明确增量。

### 4.3 第一版节点特征

模型输入仅使用当前或历史窗口可获得的字段：

```text
queue_length_s
avg_waiting_time_s
occupancy_ratio_s
queue_growth_rate_s
active_pct_s
blocked_ratio_s
starved_ratio_s
current_active_duration_s
output_rate_s
transport_waiting_time_s
route_delay_s
material_shortage_flag_s
resource_type one-hot
```

不适用于某种资源的数值字段填 0，并保留 `resource_type`。

以下字段只用于标签或分析，不能进入输入：

```text
bottleneck_score_s
is_bottleneck_window
future_bottleneck_object_id
future_bottleneck_type
time_to_start
duration
severity_weak
future disturbance
future throughput
```

### 4.4 图级特征

第一版暂不单独增加 global encoder。以下全局量可以复制到每个节点，或在图池化后拼接：

```text
total_WIP
throughput_rolling
num_busy_resources
num_blocked_resources
num_starved_resources
disturbance_flag
```

首轮建议在 GAT 后做 masked mean pooling，再拼接图级特征，避免把同一个全局值重复写入每个节点。

## 5. 必须先做的数据采集修正

### 5.1 collector schema 升级

修改：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/cfg_bottleneck_data.py
source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/src/bottleneck_data.py
source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/hc_single_env_base.py
```

collector version 从 `v0.3` 升为 `v0.4`。旧数据仍可读，但 dataset manifest 必须记录版本。

### 5.2 写入初始状态

调整为：

```text
log_resource_init_events = True
```

episode reset 后写入：

```text
resource_event_log.jsonl:
  t=0, from_state=INIT, to_state=<actual state>

buffer_event_log.csv:
  t=0, event=init

material_inventory_log.csv:
  t=0, event=init
```

初始事件必须覆盖所有有效资源，而不只是之后发生状态变化的资源。

### 5.3 增加 episode lifecycle

新增：

```text
episode_lifecycle.csv
```

字段：

```text
run_id
env_id
episode_id
event
time_step
logic_time_s
production_done
completed_jobs
```

每个 episode 至少写：

```text
START at t=0
END at production_done 所在逻辑步
```

离线脚本优先使用 `END` 作为 `episode_end`，旧数据才回退到最大事件时间。

### 5.4 补充 episode 静态配置

`episode_config.csv` 中：

- `process_time_config` 同时保存 `machine` 和 `required_materials`；
- `buffer_capacity_config` 同时保存 `capacity` 和 `supporting_materials`；
- 明确增加 `policy_type`，值来自当前 `algo`；
- 增加可复现的 `scenario_id`，由 disturbance dimension/intensity、资源数量和订单配置生成；
- 保留实际生效的 human、robot 和 active gantry 配置。

这些字段只描述 episode 启动时已知的系统配置，不包含未来调度结果。

### 5.5 修正状态语义

第一版映射改为：

```text
working_*          -> PROCESSING
waiting_*          -> WAITING
materialReadyFor_* -> READY
invalid            -> STOP
free               -> IDLE
```

聚合时：

```text
STARVED_STATES = {STARVED, WAITING}
BLOCKED_STATES = {BLOCKED}
```

当前仿真没有“下游无法接收导致资源不能释放”的明确状态时，不人为生成 `BLOCKED`。`blocked_ratio_s` 可以为 0，直到真实 blockage 事件补齐。

### 5.6 修正 disturbance schema

统一使用：

```text
intensity
```

替换当前 writer header 中的 `severity`，并保留 start/end 两条事件或合并为有完整起止时间的一条事件。第一版建议保留事件记录，但离线聚合必须按 `disturbance_id` 合并区间。

## 6. 离线特征与标签改造

主要修改：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/build_bottleneck_features.py
```

### 6.1 主键和窗口

所有输出增加：

```text
run_id
env_id
episode_id
window_index
window_start_step
window_end_step
window_start_s
window_end_s
window_size_s
stride_s
```

支持独立的：

```text
--window_size
--stride
--horizon
```

最后一个不足完整 `window_size` 的窗口默认丢弃，避免不同窗口长度造成比例和 duration 偏差。

所有 `*_time_step` 必须先使用该 episode 的 `logic_dt` 转换为 logic seconds。即使当前默认 `logic_dt=1`，离线脚本也不能把 step 永久假设为秒。

### 6.2 特征计算修正

新增或修正：

```text
blocked_ratio_s = blocked_time_s / actual_window_length
starved_ratio_s = starved_time_s / actual_window_length
output_rate_s = departures_at_node / actual_window_length
transport_waiting_time_s = pickup_time - request_time
total_WIP = producing jobs + queued jobs
throughput_rolling = completed jobs in current/history window
disturbance_flag = disturbance interval overlaps current window
```

局部归属规则：

- buffer occupancy 使用事件区间的时间加权平均，不能对事件行做普通算术平均；
- transport waiting/delay 写到 carrier、from node 和 to node，不复制到无关节点；
- material shortage 写到对应 material 所需的 station/buffer；
- queue 和 waiting 使用 canonical node ID；
- raw storage ID `BlackStorage_00` 统一为 `storage_BlackStorage_00`。

### 6.3 弱标签分数

当前所有资源共用一个 score，会让不具备 active state 的 buffer 系统性吃亏。`bstan_weak_v1` 使用按资源类别定义、最终都位于 `[0,1]` 的分数：

```text
process resource (machine/gantry/human/robot):
  0.25 * norm(queue_length)
+ 0.20 * norm(avg_waiting_time)
+ 0.25 * active_pct
+ 0.10 * norm(current_active_duration)
+ 0.10 * upstream_blocked_ratio
+ 0.10 * downstream_starved_ratio

buffer:
  0.50 * occupancy_ratio
+ 0.30 * norm(queue_length)
+ 0.20 * norm(max(queue_growth_rate, 0))
```

归一化只在同一个 window、同一资源类别内进行。`score_threshold` 保持可配置，但必须在 dataset manifest 中保存；不能根据 test 指标反向选择阈值。

### 6.4 标签时间锚点

每个预测样本的时间锚点必须是最后一个输入窗口的结束时间：

```text
anchor_time = input_window[-1].window_end_s
```

未来事件满足：

```text
event.start_s > anchor_time
event.start_s <= anchor_time + prediction_horizon
```

不得继续使用当前实现的 `window_start_s` 作为锚点。

### 6.5 episode 尾部删失

只有满足以下条件时才生成监督样本：

```text
anchor_time + prediction_horizon <= episode_end_s
```

否则标记：

```text
label_observed = 0
```

并从训练、验证和测试中排除，不能当作负样本。

如果 future event 在 episode 结束时仍未结束，则：

```text
duration_observed = 0
```

该样本仍可训练 occurrence/node/type/time-to-start，但不能计算 duration loss。

### 6.6 事件与弱标签

事件仍按以下规则合并：

```text
same bottleneck node
consecutive hot windows
n_windows >= 2
```

标签输出增加：

```text
label_version = "bstan_weak_v1"
score_config
score_threshold
min_event_windows
prediction_horizon
label_observed
duration_observed
```

`severity_weak` 第一版固定为：

```text
severity_weak =
    0.7 * event.max_score
  + 0.3 * min(event.duration_s / prediction_horizon, 1.0)
```

该字段是可复现的弱标签，不解释为真实 throughput loss。

## 7. 图数据集构建

新增：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/
  build_bstan_dataset.py
  bstan_baseline/
    __init__.py
    schema.py
    dataset.py
    graph_builder.py
```

这些模块必须是纯 Python/PyTorch，不依赖启动 Isaac Sim。

### 7.1 中间产物

每个 dataset 目录输出：

```text
bstan_dataset/
  dataset.pt
  node_catalog.csv
  graph_edge_table.csv
  model_sample_index.csv
  split.json
  normalization.json
  dataset_manifest.json
```

`dataset_manifest.json` 至少记录：

```text
source run directories
collector versions
label version
window size
stride
input windows
prediction horizon
feature names and order
node ID order
edge types
sample counts
positive rate
train/validation/test episode counts
git commit
```

### 7.2 tensor schema

`dataset.pt`：

```text
x                 float32 [S, T, N, F]
adjacency         bool    [S, N, N] or [G, N, N]
node_mask         bool    [S, N]
global_features   float32 [S, T, U]
y_occurrence      float32 [S]
y_node            int64   [S]
y_type            int64   [S]
y_time_to_start   float32 [S]
y_duration        float32 [S]
y_severity        float32 [S]
positive_mask     bool    [S]
duration_mask     bool    [S]
sample_group_id   int64   [S]
```

如果同一 scenario 的静态图相同，允许只保存一份 adjacency，并通过 graph ID 引用。

### 7.3 样本序列

对每个 `(run_id, env_id, episode_id)` 独立构建：

```text
input windows = [k-3, k-2, k-1, k]
anchor time = window[k].end_s
label = anchor 后 120s 内第一个 bottleneck event
```

禁止：

```text
跨 episode 拼接
跨 env 拼接
使用未来窗口做 normalization
使用未来 task assignment 构图
```

### 7.4 数据切分

按 episode group 切分：

```text
group_id = run_id + env_id + episode_id
train = 70%
validation = 15%
test = 15%
```

如果数据覆盖多个 scenario，优先采用分层 group split，使各 split 的正样本率和 scenario 分布不过度失衡。

所有 normalization 均只在 train split 拟合：

```text
continuous feature: train mean/std
duration target: optionally scale by horizon
missing/not-applicable feature: fill 0 after normalization and use resource type
```

## 8. 模型设计

新增：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/bstan_baseline/
  model.py
  losses.py
  metrics.py
  trainer.py

source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/
  train_bstan_baseline.py
  evaluate_bstan_baseline.py
```

### 8.1 不引入 torch_geometric

当前环境有 PyTorch，但没有 `torch_geometric`。工厂图节点数量较小，第一版使用纯 PyTorch 实现 dense multi-head GAT：

```text
masked attention over adjacency
multi-head concat in hidden layer
head mean in output layer
self-loop included
node_mask applied before softmax and pooling
```

这样可以避免额外 CUDA wheel 和 PyTorch 版本兼容问题，也方便单元测试。

### 8.2 网络结构

```text
X [B,T,N,F]
  -> feature projection
  -> GAT layer 1, ELU, dropout
  -> GAT layer 2
  -> spatial embedding [B,T,N,D]
  -> reshape [B*N,T,D]
  -> shared GRU over time
  -> final node hidden [B,N,H]
```

预测头：

```text
node head:
  node hidden -> one logit per node

graph embedding:
  masked mean pooling(node hidden)
  + final global features

occurrence head:
  graph embedding -> binary logit

type head:
  graph embedding -> bottleneck type logits

regression heads:
  graph embedding -> time_to_start
  graph embedding -> duration
  graph embedding -> severity_weak
```

默认参数：

```text
gat_hidden = 64
gat_heads = 4
gat_layers = 2
gru_hidden = 128
gru_layers = 1
dropout = 0.2
```

参数全部进入训练配置和 checkpoint，后续允许调整。

### 8.3 损失

```text
L =
  lambda_occ  * BCEWithLogits(y_occurrence)
+ lambda_node * CE(y_node)
+ lambda_type * CE(y_type)
+ lambda_tts  * SmoothL1(time_to_start)
+ lambda_dur  * SmoothL1(duration)
+ lambda_sev  * SmoothL1(severity)
```

规则：

- occurrence 对所有可观测样本计算；
- node/type/time-to-start/severity 只对正样本计算；
- duration 只对正样本且 `duration_observed=1` 时计算；
- 不存在节点的 logits 在 softmax 前置为负无穷；
- occurrence 使用 train split 计算的 `pos_weight`；
- loss 权重记录在配置中；
- 第一版不自动使用 focal loss，只有类别极度不平衡时再作为配置项开启。

## 9. 训练、评估与输出

### 9.1 训练配置

```text
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 32
max_epochs = 100
early_stopping_patience = 15
gradient_clip_norm = 1.0
selection_metric = validation PR-AUC
```

固定 Python、NumPy 和 PyTorch seed，并保存实际 device 和依赖版本。

### 9.2 指标

必须输出：

```text
occurrence:
  PR-AUC, ROC-AUC, Precision, Recall, F1

node:
  Top-1 accuracy, Top-3 accuracy, MRR

type:
  Macro-F1, confusion matrix

regression:
  MAE(time_to_start)
  MAE(duration)
  MAE(severity_weak)
```

同时输出 majority/no-event baseline，避免只看神经网络自身数值。

### 9.3 训练产物

```text
output/bottleneck_models/bstan_gat_gru/<experiment_id>/
  config.json
  best.pt
  last.pt
  metrics.json
  history.csv
  predictions_test.csv
  confusion_matrix.csv
  run_summary.json
```

checkpoint 必须包含：

```text
model_state_dict
optimizer_state_dict
model config
feature order
node order
normalization stats
label mappings
dataset manifest hash
git commit
```

## 10. 测试计划

### 10.1 collector 测试

使用最小 fake env state 测试：

- reset 会写所有资源的 INIT；
- buffer/material 会写 `t=0`；
- episode START/END 各一条；
- disturbance intensity 不丢失；
- waiting state 不再计入 BLOCKED。

### 10.2 feature/label 测试

使用手工构造的小型 timeline 测试：

- 事件区间聚合准确；
- 30s 窗口 ratio 正确；
- storage ID canonicalization 正确；
- anchor 使用 input end；
- horizon 不完整样本被删失；
- 不跨 episode 合并 event；
- `episode_id` 出现在所有输出主键中。

### 10.3 graph/dataset 测试

- 节点顺序确定且可复现；
- adjacency 只包含允许边；
- node mask 正确；
- 序列 shape 为 `[S,4,N,F]`；
- 任何输入列都不属于 future label；
- split 中 episode group 不重叠；
- normalization 只拟合 train。

### 10.4 model 测试

- CPU forward shape；
- masked node 不获得预测概率；
- 无正样本 batch 不产生 NaN；
- 两个 epoch synthetic overfit smoke test；
- checkpoint save/load 后输出一致。

### 10.5 数据质量门禁

正式训练前 validator 必须检查：

```text
每个 episode 都有 START/END
每个有效节点都有 t=0 状态
feature/label 主键无重复
所有输入特征为有限值
所有正样本 target node 都在 node catalog 且 node_mask=1
train/validation/test episode 无交集
每个 split 同时包含正负样本
训练集 positive rate 不为 0 或 1
```

若门禁失败，脚本应退出并报告具体 episode/sample，不能带 warning 继续训练。

## 11. 执行顺序

### Phase A：采集层 P0 修正

状态：已完成并通过服务器 smoke 验收。

1. 升级 collector schema 到 v0.4。
2. 写 INIT、buffer/material 初值和 episode lifecycle。
3. 修正 state mapping 和 disturbance intensity。
4. 补充 episode 静态图构建所需配置。
5. 增加 collector 单元测试。

退出条件：

```text
一个 1 env / 1 episode smoke run 能完整输出；
每个有效节点从 t=0 到 episode end 都可重建状态。
```

### Phase B：特征与标签

状态：已完成本地测试并通过服务器真实数据验收。

1. 增加 episode 主键和 stride。
2. 补 ratio、局部 transport/material、global features。
3. 修正预测 anchor 和尾部删失。
4. 输出 label metadata。
5. 增加纯 Python 单元测试。

退出条件：

```text
一个 episode 能生成无主键重复、无未来泄漏的 30s 特征和 120s 标签。
```

### Phase C：图数据集

1. 构建 node catalog 和 static prior graph。
2. 构建四窗口序列。
3. episode-level split。
4. train-only normalization。
5. 保存 dataset manifest。

退出条件：

```text
dataset validator 全部通过；
DataLoader 能输出固定 shape batch。
```

### Phase D：模型和训练

1. 实现纯 PyTorch dense GAT。
2. 接 GRU 和多任务 heads。
3. 实现 masked loss、metrics、checkpoint。
4. 先 synthetic overfit，再跑真实 smoke dataset。

退出条件：

```text
CPU 和 CUDA 至少一个环境可完成训练；
best checkpoint 可独立评估；
test predictions 可追溯回原 sample。
```

## 12. 建议执行命令

具体路径和 run ID 在实现后由脚本打印。预期命令形式：

```bash
# 1. 采集 smoke 数据
python train.py \
  --task HRTPaHC-v1 \
  --algo rule_based \
  --num_envs 1 \
  --max_episodes 6 \
  --seed 42 \
  --device cuda:0 \
  --headless

# 2. 生成窗口特征和标签
python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/build_bottleneck_features.py \
  --run_dir <run_dir> \
  --window_size 30 \
  --stride 30 \
  --horizon 120

# 3. 构建 BSTAN 数据集
python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/build_bstan_dataset.py \
  --run_dirs <run_dir_1> <run_dir_2> \
  --window_size 30 \
  --input_windows 4 \
  --horizon 120 \
  --out_dir <dataset_dir>

# 4. 训练
python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/train_bstan_baseline.py \
  --dataset_dir <dataset_dir> \
  --device cuda:0

# 5. 独立评估
python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/evaluate_bstan_baseline.py \
  --dataset_dir <dataset_dir> \
  --checkpoint <best.pt> \
  --split test
```

## 13. 数据量建议

只为跑通：

```text
1 scenario
1 env
>= 6 episodes
```

用于第一轮可信对比：

```text
scenarios:
  none
  machine
  human
  logistics
  material

each scenario:
  >= 3 seeds
  >= 10 episodes per seed
```

如果不同 disturbance run 的资源数量不同，使用全局节点并集和 episode node mask，不为每个 scenario 训练独立模型。

## 14. 风险与取舍

### 14.1 弱标签自我预测

`bottleneck_score` 由输入特征规则生成，模型可能主要学习复制规则，而不是真实吞吐影响。

控制方式：

- 明确标记 `label_version=bstan_weak_v1`；
- 标签必须位于未来 horizon，不能预测同窗 score；
- 同时报告 order throughput/cycle time；
- 后续用 disturbance 和反事实 throughput loss 增强标签。

### 14.2 type head 冗余

第一版 type 来自 future node 的 resource type，因此 node 和 type 高度相关。保留该 head 是为了接口与主模型一致，但不能把 type 指标作为主要创新证据。

### 14.3 静态 capability 边较粗

human/robot/gantry 与 station/buffer 的 capability 边可能较密。GAT 可以学习 attention 权重，但这不等于动态调度关系。后续主模型使用实际 assignment edge 时，应通过 ablation 验证增益。

### 14.4 小数据下复杂模型不稳定

第一版保持两层 GAT 和一层 GRU，并提供 majority、GRU-only baseline。若有效样本不足，不通过扩大网络解决。

## 15. 本轮评审需要确认的设计点

实现前建议确认以下决策：

1. baseline 第一版不把 material job 实例作为图节点，只把 shortage 映射为 station/buffer 特征。
2. baseline 使用静态先验图；动态 assignment/transport graph 留给主模型。
3. 使用纯 PyTorch dense GAT，不增加 `torch_geometric`。
4. 第一版固定 `30s window + 4 windows history + 120s horizon`。
5. 在训练 baseline 前先升级 collector 到 v0.4 并重新采集数据。
6. weak label 必须修正时间锚点和 episode 尾部删失后才能用于训练。
