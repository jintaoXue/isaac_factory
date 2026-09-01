# PDFormer-BSTAN 共享实验七点设计评审

## 1. 文档目的

本文用于讨论 PDFormer 与 BSTAN-style GAT-GRU 的共享实验设计。评审基于：

```text
dev_tyx = c101eff
dev_xwt = 9c928b3
shared derived = dev_tyx tools/bn_agg
prediction target = factory_a1a3_remain_v1
```

当前两条模型链路都已经具备 A.1 未来占用预测和 A.3 原因分类接口，但“代码可运行”
不等于“可以直接做公平、可解释的正式对照实验”。下文将问题分为：

- 当前实现；
- 风险与影响；
- 推荐的统一设计；
- 是否需要重新采集 raw 数据。

基本原则是：raw、派生特征、标签、split 和评估协议由所有模型共享，模型之间只改变
模型结构及经过记录的训练超参数。

本轮 baseline 范围固定为 B2 XGBoost、B3 LSTM、B4 GCN-GRU 和 B5
BSTAN-style GAT-GRU。B0 No-event 与 B1 Persistence 不实现为 baseline；其对应的
prevalence、always-negative 等数值仅作为指标背景。PDFormer 作为待对齐模型，不编号为
baseline。

## 2. 结论摘要

| 编号 | 问题 | 当前判断 | 推荐动作 | 是否重采 raw |
| --- | --- | --- | --- | --- |
| 1 | split 不一致 | 正式对比阻塞项 | 两边共用 episode-level split manifest | 否 |
| 2 | A.1 目标节点范围不明确 | 任务语义未冻结 | 图保留全部节点；正式 A.1 推荐只预测生产执行资源 | 否 |
| 3 | `node_mask` 未完整应用 | 固定布局影响小，跨布局会错误监督 | loss、metrics、事件恢复统一应用有效节点 mask | 否 |
| 4 | 辅助头时间锚点偏移 | `will/mark/tts` 存在至多一窗口边界泄漏 | 未来辅助头统一以输入结束边界 `t` 为 anchor | 否 |
| 5 | 扰动和瓶颈事件合并 | 原因与结果混淆 | disturbance 作为 cause/context，运行影响确认 bottleneck | 否，只重建 derived |
| 6 | majority baseline 使用评估集分布 | 对照指标不规范 | 只从 train 确定多数类，固定评估 val/test | 否 |
| 7 | A.1 只有占用格指标 | 不能完整评价最终事件输出 | 增加共享事件匹配与事件级指标 | 否 |

优先级建议：先统一 1、3、4、6、7；第 2 和第 5 涉及标签语义，需要双方确认后一起
修改。任何一方都不应单独改变共享标签，否则 BSTAN 与 PDFormer 将再次变成不同任务。

## 3. 问题一：数据划分方式不一致

### 3.1 当前实现

BSTAN 按 episode 划分 train、validation、test，同一个 episode 不跨 split。
PDFormer 当前将所有窗口样本打散后按比例切分，因此一个 episode 的相邻窗口可能分别
进入 train 和 test。

制造业时序窗口高度重叠。例如两个相邻样本可能共享 11/12 的输入窗口，也可能共享
绝大多数未来占用目标。随机窗口切分会让测试集包含与训练样本近乎重复的序列。

### 3.2 风险

- 产生同 episode 泄漏，使测试指标偏乐观；
- BSTAN 和 PDFormer 的测试难度不同，指标不能公平横向比较；
- 无法评价模型对未见 episode、未见扰动实现的泛化能力。

### 3.3 推荐设计

已确定采用 BSTAN 的 episode-level 方案：

```text
1. 为所有通过 raw quality gate 的 episode 生成稳定 episode_key；
2. 按 scenario、intensity、seed 分层；
3. 在 episode 层面切分 train/validation/test；
4. 输出唯一 shared_split_manifest.json；
5. PDFormer、BSTAN、LSTM 等全部读取同一 manifest；
6. normalization、class weight 和 majority class 只从 train 拟合。
```

正式 test split 一旦冻结，不因某个模型效果变化而重新划分。

## 4. 问题二：A.1 可以预测哪些节点

### 4.1 当前实现

全局图同时包含生产资源和 buffer/storage 节点。PDFormer 当前 A.1
`remain_to_jobs_done_score/hot[K,N]` 对全部节点生成标签与预测，事件恢复也可能输出：

```text
node_id = storage_BlackStorage_02
```

BSTAN 为保持任务一致，目前也采用全节点 A.1。与此同时，`04.期望输出.md` 将 A.1
描述为“发生在哪个工位/工序”，因此代码目标与业务措辞尚未完全统一。

### 4.2 两种合理定义

**全资源定义**：machine、human、robot、gantry、buffer 都可成为瓶颈节点。输出应称为
“瓶颈资源/位置”，buffer 满载、供料约束也属于预测对象。

**生产资源定义**：buffer 仍是图输入并参与消息传播，但 A.1 只允许实际执行生产或运输
任务的节点成为目标。输出可继续称为“瓶颈工位/工序”。

### 4.3 推荐设计

结合当前 A.1 的业务表述，正式实验推荐：

```text
图输入节点 = 全部异构节点，包括 buffer
A.1 目标节点 = 生产执行资源
buffer 作用 = 队列、库存、阻塞和供料上下文
```

实现上必须在共享数据层定义 `target_node_mask`，并由所有模型共同使用。若讨论后决定
buffer 也是业务瓶颈目标，则两边继续全节点预测，但文档和输出字段应统一改称“瓶颈
资源/位置”，并补充 buffer 正常满载与异常拥堵的区分规则。

该选择只改变 derived 标签和 tensor target，不需要重采 raw。

## 5. 问题三：节点有效性 mask 未完整应用

### 5.1 `node_mask` 的含义

为了批量训练，不同 episode 需要使用同一个全局节点顺序。假设全局表有五个节点，某个
布局只有三个节点：

```text
global nodes = [machine_0, machine_1, human_0, gantry_0, storage_A]
node_mask    = [1,         0,         1,       0,        1]
```

`node_mask=0` 表示该节点在当前 episode 中不存在，它不是“存在但状态为零”。

`target_node_mask` 是另一层语义：节点可以真实存在，但不一定允许成为 A.1 目标。

### 5.2 当前风险

当前 A.1 loss 和 metrics 主要应用未来时间 `remain_mask`，没有完整组合节点有效性。
当未来更换布局、减少设备或合并不同资源配置时，不存在的节点会被当作大量负样本：

- 训练模型学习“该节点永远不是瓶颈”，而不是“该节点不适用”；
- 大量容易的真负例可能抬高指标；
- 不存在节点上的非零预测会被误算为 false positive；
- 事件恢复可能输出当前 episode 根本不存在的节点。

当前 machine/human cohort 的工厂布局基本相同，所以这一问题可能暂时不改变结果，但它会
阻止跨布局实验。

### 5.3 推荐设计

统一定义：

```text
valid_graph_node  = node_mask
valid_a1_target   = node_mask & target_node_mask
valid_future_cell = remain_mask & valid_a1_target
```

如果问题二决定全节点预测，则 `target_node_mask` 对所有可预测资源为 1；如果采用生产资源
定义，则 buffer 对应位置为 0。上述 mask 必须同时进入：

- score/hot loss；
- occupancy precision、recall、F1；
- threshold selection；
- occupancy-to-events；
- mark-node loss 和指标。

## 6. 问题四：辅助任务时间锚点不一致

### 6.1 PDFormer 当前实现

PDFormer 当前样本切片为：

```text
输入窗口                    = [t-input_windows, t)
A.1 remain occupancy       = [t, jobs_done)
will/mark/cause/tts index  = t-1
```

`bn_agg` 中 `will[i]` 的定义是寻找：

```text
event.start > window[i].start
event.start <= window[i].start + 180s
```

因此输入已经包含完整的 `t-1` 窗口，但 `will[t-1]` 的预测区间从该窗口起点开始。如果
事件在 `t-1` 窗口内部启动，模型已经观察到部分事件状态，却仍把它作为未来目标。泄漏
范围最多为一个窗口，当前设置下即最多 60 秒。

A.1 remain occupancy 从 `t` 开始，没有该问题。STGNPP 的 next event 选择也从 `t`
开始，但其 time-to-event 参考点仍使用 `t-1` 的窗口起点，会多计算一个窗口。

### 6.2 辅助头的任务含义

| 辅助头 | 当前用途 | 推荐时间语义 |
| --- | --- | --- |
| `will` | 未来 180 秒是否发生瓶颈 | 从输入结束边界 `t` 向后看 180 秒 |
| `mark` | 未来最早瓶颈节点 | 与 `will[t]` 对应同一事件 |
| `time_to_start` | 距离事件开始的时间 | `event.start - boundary(t)` |
| `cause` | 当前窗口原因分类 | 需决定是当前诊断还是未来事件原因 |

### 6.3 推荐设计

```text
history                 = [t-input_windows, t)
A.1 occupancy           = [t, jobs_done)
will/mark/time-to-start = label[t]
STGNPP time reference   = boundary(t)
```

对 A.3 推荐采用“预测到的未来 A.1 事件原因”，使 cause 与 A.1 指向同一事件。如果业务
还需要当前原因诊断，应另设 `current_cause` 任务，不能与 `future_event_cause` 共用一个
含义不清的标签。

## 7. 问题五：扰动事件与瓶颈事件混合

### 7.1 当前实现

`bn_agg` 当前将运行分数产生的 `score_events` 与扰动日志产生的
`disturbance_events` 做 union，合并结果同时用于 A.1 事件和近 180 秒标签。

这等价于允许以下推理：

```text
发生机器故障 -> 直接产生瓶颈正标签
```

但扰动是潜在原因，不必然形成系统瓶颈。如果故障发生时没有待加工任务、存在冗余设备，
或者没有造成队列增长和吞吐下降，合理标签应是“发生扰动，但未形成瓶颈”。

### 7.2 风险

- 模型可能学习扰动检测，而不是瓶颈预测；
- A.1 将原因当作结果，瓶颈定义不够严格；
- A.3 的扰动原因与 A.1 标签存在定义上的循环；
- 无法支持“某种扰动是否真正导致瓶颈”的分析。

当前 PDFormer 与 BSTAN 使用相同 union，因此模型间对比仍然公平，但只能称为共享 weak
label，不能据此宣称扰动与瓶颈之间存在强因果关系。

### 7.3 推荐设计

共享标签分为三层：

```text
L1 disturbance event
   machine failure / human unavailable / logistics delay / material shortage

L2 operational evidence
   queue growth / blocked / starved / throughput drop / sustained utilization

L3 bottleneck event
   节点局部压力持续存在，并有 L2 系统影响支持
```

A.1 只监督 L3；A.3 将 L1/L2 与 L3 做时间、路径和目标节点关联，生成候选原因及置信度。
扰动可以作为模型可观测上下文，但不能因为出现扰动就自动创建 A.1 正样本。

该修订可以完全基于现有 raw 的 disturbance、resource、job 和 buffer 日志重建，不需要
重新运行 Isaac Sim。

## 8. 问题六：多数类 baseline 的计算不规范

### 8.1 为什么需要多数类 baseline

A.3 原因类别不均衡。若 90% 标签都是 `queue_buildup`，一个不读取输入、永远预测
`queue_buildup` 的模型也有 90% accuracy。多数类 baseline 用于判断模型是否真正超过
这种最简单策略。

### 8.2 当前问题

当前评估会分别查看 validation/test 的真实原因标签，并在各自集合中选择最多的类别。
这相当于看完 test 答案以后再决定 baseline 永远猜什么，使用了测试集标签分布。

它不会影响 PDFormer 本身的 `cause_acc`，只会让用于比较的参考指标不规范或偏高。

### 8.3 推荐设计

```text
majority_class = argmax(count(train cause labels))
validation baseline prediction = fixed majority_class
test baseline prediction       = fixed majority_class
```

训练阶段同时只使用 train 计算 cause class weight。A.3 报告至少包含：

```text
accuracy
macro-F1
per-class precision/recall/F1/support
confusion matrix
train-derived majority accuracy
```

某一类别在独立 episode 中支持不足时，结果只作为探索分析，不作该原因具有可靠预测能力的
结论。

## 9. 问题七：A.1 缺少事件级评估

### 9.1 当前指标覆盖范围

PDFormer 当前主要使用未来 `hot[K,N]` 的 cell-level precision、recall、F1，以及
`remain_len` MAE。它们可以评价未来时间窗口乘节点的占用格，但不能完整评价
`04.期望输出.md` 要求的：

```text
是否发生
发生在哪个工位
开始时间
持续时间
```

连续多个 anchor 的未来目标高度重叠，仅报告所有 cell 的 micro F1 还会让长 episode 和
重复预测占据更大权重。

### 9.2 共享事件恢复

所有模型必须调用同一个 `occupancy_to_events`：

```text
输入：hot probability [K,N]、validation 选定阈值、有效节点 mask
处理：同节点连续 hot 窗口合并
输出：node_id、start_time、end_time、duration、mean/max confidence
```

不能让不同模型各自使用不同的 gap merge、最短持续时间或阈值规则。

### 9.3 推荐事件匹配协议

建议先按时间生成候选配对，再进行 episode 内一对一匹配：

```text
候选条件：temporal IoU >= 0.30
匹配目标：优先最大 temporal IoU
约束：一个预测事件和一个真实事件最多各匹配一次
```

同时输出两套结果：

1. 时间匹配：不预先要求节点相同，用于独立评价节点是否预测正确；
2. 严格匹配：时间匹配且节点相同，用于评价最终可用预警。

IoU、最短事件长度和 merge gap 只能在 validation 上选定，test 固定使用。

### 9.4 推荐指标

| 层次 | 指标 | 用途 |
| --- | --- | --- |
| Occupancy | PR-AUC、precision、recall、F1 | 保留当前稠密预测评价 |
| Event detection | event precision、recall、F1 | 是否正确发现完整事件 |
| Location | node accuracy、top-k accuracy | 事件发生位置是否正确 |
| Timing | onset MAE、onset tolerance recall | 预警开始时间是否可用 |
| Duration | duration MAE、duration IoU | 持续时间是否准确 |
| Operations | false alarms per episode | 每轮仿真的误报负担 |
| Completion | remain length MAE | 剩余工单完成长度预测 |

所有指标先按 episode 计算，再报告 episode macro mean；同时可保留全 cell micro 指标用于与
当前结果衔接。正式实验建议按 episode bootstrap 给出 95% confidence interval。

## 10. 推荐统一后的任务契约

```text
输入：过去 12 个完整 60 秒窗口
anchor：最后一个输入窗口的结束边界 t

A.1：
  从 t 到剩余工单清零的 score/hot[K,N]
  remain_len
  由共享 occupancy_to_events 恢复节点、开始和持续时间

近窗辅助：
  t 后 180 秒 will
  与该 will 对应的第一个事件节点 mark
  event.start - t

A.3：
  与预测 A.1 事件对应的未来原因分类
  原因标签与置信度来自 disturbance/context 到 operational event 的关联

数据：
  shared episode split
  train-only normalization/class statistics
  node_mask 和 target_node_mask 全链路生效

评估：
  occupancy 指标 + event 指标
  validation 选阈值，test 固定
  episode macro + bootstrap CI
```

## 11. 讨论时需要确认的决策

建议会议逐项形成明确结论：

| 决策 | 推荐结论 |
| --- | --- |
| split | 采用共享 episode-level manifest |
| A.1 节点范围 | 全节点输入，生产执行资源作为目标 |
| buffer 语义 | 输入上下文，不直接称为瓶颈工位 |
| mask | `node_mask & target_node_mask & remain_mask` 全链路应用 |
| 辅助 anchor | 统一使用输入结束边界 `t` |
| A.3 语义 | 预测未来 A.1 事件原因；当前诊断另立任务 |
| disturbance | 作为 cause/context，不自动创建 bottleneck event |
| majority baseline | train 决定类别，固定评估 val/test |
| checkpoint | 暂保留 validation occupancy F1，补报 event F1 |
| 正式主指标 | occupancy PR-AUC/F1 + strict event F1 + onset/duration MAE |

## 12. 推荐实施顺序

```text
Phase 1  冻结 shared episode split
Phase 2  冻结节点目标范围、mask 和时间 anchor
Phase 3  修正 majority baseline，增加统一事件级 evaluator
Phase 4  用现有标签重建并完成第一轮公平模型对照
Phase 5  解耦 disturbance 与 operational bottleneck，发布新 label version
Phase 6  用相同 raw 重建 derived/dataset，重训全部模型
Phase 7  扩充 scenario、seed、订单和布局后开展正式实验
```

Phase 1-6 都可以使用现有 raw。只有 Phase 7 的新场景、订单或布局超出现有采集覆盖时，
才需要重新运行 Isaac Sim。
