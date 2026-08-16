# BSTAN 瓶颈标签 v2.3 实现说明

> 历史实现：该标签路径已由 Git 历史保留，不再属于当前 benchmark 主路径。
> 现行实现统一使用 `dev_tyx@c101eff` 的 `tools/bn_agg`，详见
> `统一Baseline对照实验执行设计.md` 第 16 节。

## 1. 文档目的

本文档单独记录 BSTAN-style GAT-GRU baseline 的瓶颈标签定义、修订证据、当前实现契约和服务器验收结果，方便后续回顾和复现。

当前唯一执行版本：

```text
collector_version = v0.6
label_version = bstan_weak_v2_3
derived_dir = derived_phase_b_v2_3
dataset_version = bstan_dataset_v3
target_node_category = process
implementation_commit = 92aa1c0
```

旧标签版本只通过 Git 历史复现，当前代码路径不保留兼容分支或兜底解析。

## 2. 预测目标

对每个 anchor 时刻 `t`，使用过去 4 个 30 秒窗口，即 120 logic seconds 的图序列，预测未来 120 logic seconds 内是否将出现瓶颈事件。

当 `will_bottleneck=1` 时，同时预测：

| 任务 | 标签字段 | 定义 |
| --- | --- | --- |
| 发生 | `will_bottleneck` | 未来 horizon 内是否有事件 |
| 位置 | `future_bottleneck_object_id` | 第一个未来事件的生产资源节点 |
| 类型 | `future_bottleneck_type` | 目标节点的 resource type，不是扰动原因 |
| 开始时间 | `time_to_start` | 事件开始时间减 anchor 时间 |
| 持续时间 | `duration` | 事件连续 hot windows 的时长 |
| 严重度 | `severity_weak` | 局部分数与持续时间的弱监督组合 |

本版本是 weak label：标签由可解释规则从仿真日志派生，不是人工逐窗口标注的强标签。

## 3. 输入证据与时间语义

标签使用 collector v0.6 的原始日志：

```text
resource_event_log.jsonl
buffer_event_log.csv
job_trace.csv
material_inventory_log.csv
disturbance_log.csv
episode_lifecycle.csv
episode_config.csv
```

时间统一使用 logic seconds。当前 `logic_dt=1.0`，因此 logic time 与 `time_step` 数值一致，但实现不依赖 wall-clock 时间或 Isaac Sim 渲染帧率。

每个窗口使用完整的 `[window_start, window_end)` 区间。`total_WIP` 和标签系统影响均使用窗口结束时点 WIP，避免与跨窗口 overlap 计数混用。

## 4. v2.3 标签算法

### 4.1 节点局部压力分数

生产资源的局部分数为：

```text
score = 0.25 * normalized_queue_length
      + 0.20 * normalized_avg_waiting_time
      + 0.25 * active_pct
      + 0.10 * normalized_current_active_duration
      + 0.10 * upstream_blocked_ratio
      + 0.10 * downstream_starved_ratio
```

分数被限制在 `[0, 1]`。queue、waiting time 和 active duration 的归一化在同一窗口尺度下进行。

buffer 仍计算 occupancy、queue 和 growth 特征，并作为 GAT 图节点参与信息传播；但由于当前 raw log 没有可靠的 buffer 局部上下游阻塞映射，buffer 不进入第一版标签目标集。

### 4.2 单窗口候选门禁

每个窗口只在 process 节点中排序。得分最高节点必须同时满足：

```text
absolute gate: score >= 0.50
relative gate: best_score - second_best_score >= 0.10
warm-up gate: window_start >= 120 seconds
system-impact gate: WIP growth or operation throughput drop
```

relative gate 在所有 process 候选节点之间计算，用于防止低负载时仅因存在 argmax 就制造瓶颈。

### 4.3 系统影响门禁

系统影响满足任一条件：

```text
WIP growth:
  end-of-window WIP - previous 3-window mean WIP >= 1.0

operation throughput drop:
  previous 120-second completed operations >= 2
  and (baseline - recent) / baseline >= 0.25
```

operation completion 使用 `job_trace.csv` 中 processing task 的 `departure`。不再使用整单 `stage_complete` 作为短周期吞吐信号，也不使用缺少稳定 support 的 operation cycle growth。

原始影响信号保持 `min_event_windows=2` 个窗口，使瞬时的系统变化可以与持续的局部压力对齐。这是 impact hold，不是人为扩展最终瓶颈事件。

### 4.4 事件合并

只有同一 process 节点连续至少 2 个 hot windows 才生成事件。在 30 秒窗口和 30 秒 stride 下，最短事件持续 60 logic seconds。

事件严重度为：

```text
severity_weak = 0.7 * max_score
              + 0.3 * min(duration / prediction_horizon, 1.0)
```

若事件持续到 episode 结束，`duration_observed=0`，表示持续时间右删失，训练 duration head 时不应当作完整值。

### 4.5 未来标签与删失

对 anchor `t`，只搜索：

```text
t < event_start <= t + 120 seconds
```

若有多个事件，使用最早事件。当 `t + horizon` 超过 episode end 时，设置 `label_observed=0`，`will_bottleneck` 留空，不将未完整未来区间错当作负样本。

### 4.6 symptom 与 cause 分离

`bottleneck_symptom_type` 来自瓶颈节点的当前状态：

```text
blocked_downstream
starved_upstream
queue_growth
queue_buildup
high_utilization
system_pressure
```

`material_shortage` 是可能原因，不是 process 节点的 symptom。runtime disturbance 只用于生成事后解释字段：

```text
candidate_cause_type
cause_target_resource_id
cause_label_confidence
disturbance_to_bottleneck_delay_s
```

扰动本身不直接产生正标签，未来才开始的扰动也不进入 anchor 时刻的模型输入，避免未来泄漏。

## 5. Phase C 与模型契约

dataset v3 将“图节点”和“可预测目标”分开：

| 张量 | 作用 |
| --- | --- |
| `node_mask` | 标记 episode 中存在的图节点，buffer 也可为 1 |
| `target_node_mask` | 标记允许 node head 输出的 process 节点 |
| `target_type_mask` | 标记允许 type head 输出的 process resource type |

GAT 继续使用 `node_mask` 处理全图；node/type logits 分别使用 target mask，因此 buffer 可以传播上下文，但不会被预测为瓶颈目标。

严格门禁包括：

```text
label_version must equal bstan_weak_v2_3
dataset_version must equal bstan_dataset_v3
collector_version must equal v0.6
positive target must exist in node/type catalogs
positive target masks must equal 1
target masks cannot contain inactive graph nodes
every split must contain positive and negative samples
episode groups cannot cross splits
normalization is fitted on train only
```

## 6. 版本演进与决策证据

| 版本 | 主要问题 | 证据 | 处理 |
| --- | --- | --- | --- |
| v1 | 强制 argmax，buffer 长期支配 | 正样本率 37.2%-52.9%，buffer argmax 80.0%-89.2% | 放弃 |
| v2.1 | 系统影响只持续单窗口，整单吞吐/cycle support 不足 | 3268 observed anchors，0 个正样本 | 放弃 |
| v2.2 | 全局 starvation proxy 被当作 buffer 局部传播 | 124 个事件中 121 个为 `storage_BlackStorage_02` | 放弃 |
| v2.3 | process-only target，operation throughput，impact hold，严格 target mask | 53 个事件，208 个正样本，无 buffer target | 当前版本 |

关键 Git 记录：

```text
6135c2c feat: add bottleneck weak labels v2
b04f77b feat: refine bottleneck weak labels v2.1
25de8f4 refactor: remove legacy bottleneck label paths
d587732 fix: align bottleneck labels with operation flow
92aa1c0 fix: target productive bottleneck resources
```

## 7. Pilot 实测结果

10 个 15 单 Pilot episode 在同一批 raw v0.6 日志上重建 v2.3：

| Run | Scenario | Events | Positive anchors | Rate |
| --- | --- | ---: | ---: | ---: |
| `2026-08-01_22-36-03_seed42` | none | 5 | 20 | 6.78% |
| `2026-08-01_22-52-42_seed42` | machine | 6 | 24 | 8.86% |
| `2026-08-01_23-47-48_seed42` | human | 8 | 30 | 8.80% |
| `2026-08-02_00-01-12_seed42` | logistics | 4 | 16 | 3.70% |
| `2026-08-02_00-50-55_seed42` | material | 3 | 12 | 4.15% |
| `2026-08-02_01-09-18_seed43` | none | 4 | 16 | 5.82% |
| `2026-08-02_01-18-03_seed43` | machine | 4 | 14 | 4.58% |
| `2026-08-02_01-27-33_seed43` | human | 6 | 24 | 7.02% |
| `2026-08-02_01-37-44_seed43` | logistics | 4 | 16 | 3.65% |
| `2026-08-02_01-50-50_seed43` | material | 9 | 36 | 12.95% |

汇总：

```text
events = 53
observed anchors = 3268
positive anchors = 208
raw-label positive rate = 6.36%
invalid buffer targets = 0
```

process-only 阈值扫描中，`0.50` 保留 6 个事件目标节点并覆盖全部五类场景。阈值升到 `0.55` 后事件降至 17 个，因此冻结 `0.50`，不再根据 test 指标调参。

这些事件中，runtime disturbance 开始后 120 秒内的事件数为 0。这不是把标签改成扰动真值的理由；它表明当前 Pilot 只验证了数据和标签流程，尚不支持扰动已造成可观测瓶颈的因果结论。

## 8. Dataset v3 验收结果

10 个 Pilot episode 构建的 smoke dataset：

```text
total_samples = 3238
positive_samples = 208
positive_rate = 6.42%
split samples = train 1732 / validation 641 / test 865
split episodes = train 6 / validation 2 / test 2
split positives = train 132 / validation 44 / test 32
x shape = (3238, 4, 34, 17)
global shape = (3238, 4, 6)
buffer nodes = 18
validation = passed
```

`3238 = 3268 - 10 * 3`：每个 episode 前 3 个 anchor 无法组成 4 个历史窗口，因此不进入 dataset。正样本位于 warm-up 之后，208 个全部保留。

Phase B 有 37 个窗口节点，Phase C 根据 `process_time_config` 过滤 3 个未参与当前产品流程的 machine 节点，最终保留 34 个图节点：18 个 buffer 和 16 个生产相关资源。

## 9. 当前限制

1. Pilot 只有每场景 2 个 episode，适合 smoke training，不适合发布强的分场景结论。
2. 当前只使用 rule-based policy、一种产品、15 单负载和一种工厂布局。
3. weak label 依赖规则阈值，尚无人工强标签或外部 ground truth 校验。
4. 同一 episode 内相邻窗口高度相关，样本数不等于独立实验数。
5. `future_bottleneck_type` 是 resource type，与 node target 高度相关，不代表根因类型预测。
6. buffer 暂不是标签目标，不代表 buffer 在制造系统中不可能成为瓶颈。当未来有可靠的局部流量、阻塞传播和容量真值时，可以通过新标签版本重新纳入。

## 10. 流水线变更时如何处理

| 变更 | 是否重新采集 | 必须执行 |
| --- | --- | --- |
| 只修改离线标签阈值/规则，raw v0.6 已有所需字段 | 否 | 新 label version、重建 Phase B/C、重跑敏感性与门禁 |
| 修改产品工艺路由或 `process_time_config` | 是 | 新 Pilot、重建 node catalog/graph，重新冻结阈值 |
| 增删 machine/human/gantry/AGV/buffer | 是 | 验证 t=0 资源状态、节点映射、图边和 target mask |
| 修改订单量、产品类型或 policy | 是 | 新负载 Pilot、重做标签分布与 split 评估 |
| 修改扰动强度、目标或时间 | 是 | 重做扰动生效与事件延迟审计，不直接把扰动改成标签 |
| 只修改 GAT-GRU 结构或训练参数 | 否 | 保持 dataset manifest，用新 model run 记录差异 |

阈值只能在独立 Pilot 或 validation 规则冻结前调整，不能根据 test 指标反向调整。任何修改标签语义的变更都必须升级 label version，不覆盖 v2.3 的语义。

## 11. 代码与产物映射

| 位置 | 职责 |
| --- | --- |
| `tools/build_bottleneck_features.py` | 窗口特征、局部分数、系统影响、事件和未来标签 |
| `tools/bstan_baseline/schema.py` | label/dataset/collector 版本和张量 schema |
| `tools/bstan_baseline/dataset.py` | node catalog、target mask、split、normalization 和严格验证 |
| `tools/bstan_baseline/model.py` | node/type logits 的 target mask |
| `tools/bstan_baseline/trainer.py` | 训练前严格检查 dataset manifest |
| `test/test_build_bottleneck_features.py` | 标签门禁、吞吐下降、buffer 排除和 metadata 测试 |
| `test/test_bstan_dataset.py` | dataset v3、target mask 和 split 测试 |
| `test/test_bstan_model.py` | node/type 输出屏蔽测试 |

每个 run 的 Phase B 产物：

```text
derived_phase_b_v2_3/episode_XX/env_XX/
  window_feature_table.csv
  bottleneck_event.csv
  bottleneck_label.csv
  label_metadata.json
  job_kpi.csv
  pipeline_summary.json
```

Phase C 产物：

```text
bstan_dataset_v3/
  dataset.pt
  dataset_manifest.json
  split.json
  normalization.json
  node_catalog.csv
  graph_edge_table.csv
  model_sample_index.csv
```

## 12. 当前结论

v2.3 解决了强制 argmax、零事件、静态 buffer 支配、整单吞吐 support 不足、未来区间误标负样本以及 buffer 仍可被模型输出等问题。它已通过 10 个 Pilot episode 的 Phase B 重建和 dataset v3 验收，可用于下一步 smoke training。

它仍是 baseline 的弱监督标签，当前结果不足以支持扰动因果、跨布局泛化或单场景性能结论。正式实验仍需按数据质量设计扩充场景矩阵并固定独立 split。
