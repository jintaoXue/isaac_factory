# 统一 Baseline 对照实验执行设计

## 1. 文档状态

- 状态：BSTAN 优先链路已进入本地实现，等待服务器 raw 验收
- raw 来源：`dev_tyx@a8b4f38` 采集的 `collector_version=v0.3` 数据
- 第一实现目标：BSTAN-style GAT-GRU
- 最终目标：所有 baseline 使用同一 raw、canonical features、标签、split 和评估协议
- 本文不修改或伪造服务器已有 raw 数据

本轮实验不再要求 raw 必须由 `collector_version=v0.6` 生成。v0.6 及
`bstan_weak_v2_3` 的既有实现继续由 Git 历史和原实现文档记录，但不在新的
benchmark 主路径中保留兼容分支或无效兜底。

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

现有 10 个 Pilot episode 可先按 `6/2/2` 完成 smoke baseline，但不足以支撑单场景强结论。

## 8. 最终 Baseline 矩阵

### 8.1 主结果模型

| 编号 | 模型 | 类型 | 实验作用 |
| --- | --- | --- | --- |
| B0 | No-event / Prevalence | 类别先验 | 判断是否超过无事件基线 |
| B1 | Persistence Heuristic | 规则模型 | 判断是否只需外推当前状态 |
| B2 | XGBoost | 非深度、非图 | 衡量特征工程本身的效果 |
| B3 | LSTM | 深度时序、非图 | 判断图结构是否必要 |
| B4 | GCN-GRU | 传统时空图 | 普通图卷积与循环网络基线 |
| B5 | BSTAN-style GAT-GRU | 图注意力时序 | 判断 attention graph 的收益 |
| B6 | PDFormer | 图 Transformer | 较强现代时空模型 |
| Proposed | 最终主模型 | 主方法 | 与统一 baseline 比较 |

最小可交付主表为：

```text
No-event
XGBoost
LSTM
GCN-GRU
BSTAN-style GAT-GRU
Proposed Model
```

PDFormer 在 canonical 输入和 episode split 完成适配后加入正式主表。

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

- XGBoost：展平 `T x N x F`、mask 和 global features；
- LSTM：每个时刻展平固定节点顺序，不读取 adjacency；
- GCN-GRU：每个时刻 GCN，再沿时间维度 GRU；
- GAT-GRU：每个时刻 GAT，再沿时间维度 GRU；
- PDFormer：读取相同 node features、graph、mask 和标签。

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

tools/bstan_baseline/dataset.py
  -> 读取 canonical dataset 和 shared split

tools/bstan_baseline/graph_builder.py
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

### Step 2：低成本对照

在同一 dataset 上实现 No-event、Persistence、XGBoost 和 LSTM。

### Step 3：图模型对照

复用 BSTAN 训练器和预测头实现 GCN-GRU，并完成 GCN 与 GAT 对比。

### Step 4：强模型适配

最后将 PDFormer 改为读取 canonical dataset 和 shared split，再进入正式主表。

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
- BSTAN dataset、训练、checkpoint 和 test evaluation；
- validation 选择 occurrence F1 threshold，test 固定使用该 threshold；
- 一条命令执行 audit、canonical build 和 BSTAN dataset build。

服务器 raw 验收通过前，不开始 LSTM、GCN-GRU 或 PDFormer 适配。
