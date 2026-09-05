# PDFormer-Baseline 共享实验七点设计评审

## 1. 评审基线

```text
main implementation = dev_tyx@7b2fc02
derived contract    = tyx_bn_agg_unsupervised_v2
prediction target   = factory_ops_event_30m_to_15m_v1
baseline scope      = B2 XGBoost / B3 LSTM / B4 GCN-GRU / B5 GAT-GRU
```

截至 2026-09-01，七点均已有确定设计并进入 baseline 主路径。本文记录决议理由；具体参数、
构建命令和验收条件以《统一Baseline对照实验执行设计》为准。

## 2. Episode Split

**问题**：随机打散窗口会让同一 episode 的高度相关样本跨 train/test，造成泄漏。

**决议**：按 raw run 分组，在每个 run 内用固定 seed 打乱完整 episode，再做 70/15/15
切分。split 过程不读取正负标签、原因分布或模型结果。归一化只拟合 train。

**代码落点**：`factory_baselines/dataset.py::_split_groups`。

## 3. A.1 目标节点

**问题**：全图包含 buffer 和 human，但业务输出要求预测发生瓶颈的生产或运输站点。

**决议**：图输入保留 machine、gantry、human、transport_robot、buffer；A.1 只监督
machine、gantry、transport_robot。buffer 与 human 是因果上下文，不是该任务的目标节点。

**代码落点**：主实验 `occupancy_node_mask`，baseline 共享为 `occ_node_mask`。

## 4. 有效性 Mask

**问题**：只使用未来时间 mask 会把不存在或不允许预测的节点当作容易负样本，跨布局时
还可能输出当前 episode 不存在的资源。

**决议**：所有 loss、cell sampling、threshold scoring 和 event metrics 使用：

```text
valid_future_cell = remain_mask & node_mask & occ_node_mask
```

模型仍可在全部存在节点上编码和传播信息。

## 5. 时间锚点

**问题**：旧辅助 `will/mark/tts` 与输入最后一个窗口重叠，最多泄漏一个窗口边界。

**决议**：统一为：

```text
history = [t-30, t)
future occupancy/event = [t, min(t+15, jobs_done))
```

正式输出使用每站点 `event_will/event_start/event_duration`。旧全局辅助头不作为主实验
checkpoint 或正式结果结论。

## 6. 扰动与瓶颈

**问题**：直接把 machine failure、human unavailable 等扰动合入瓶颈事件，会让模型学习
扰动检测而不是运行瓶颈。

**决议**：A.1 使用 `ops_hot_mask`，只依赖排队、阻塞、饥饿、停机、运输等待、物料短缺
传播和 labor saturation 等 operational evidence。扰动字段仅用于输入上下文、原因分析和
场景分组。`bottleneck_score_s` 不参与 A.1 target。

原因辅助任务只报告六类过程原因：transport delay、material shortage、blocked downstream、
starved upstream、queue buildup、high utilization。注入名称和 score fallback 在 loss/metrics
中忽略。

## 7. Majority Baseline

**问题**：从 validation/test 的真实标签中分别选择该集合多数类，相当于读取评估答案后
再确定 baseline。

**决议**：当前 baseline 评估不再动态计算评估集多数类。后续如需报告 cause majority，
必须只从 train 确定唯一类别，再固定应用到 validation/test，并写入训练 metadata。

## 8. 事件级评分

**问题**：cell-level occupancy F1 不能完整回答哪个站点、何时开始、持续多久。

**决议**：所有模型共同报告：

- `who_precision/recall/f1`；
- `report_precision/recall/f1`，其中 station 正确且 start error 不超过 3 个窗口才算命中；
- `start_mae` 与 `dur_mae`；
- ongoing/upcoming 子集指标；
- occupancy event temporal IoU 指标；
- cell-level occupancy 指标作为辅助。

事件最短持续 8 个窗口，报告阈值在 validation 扫描后冻结到 test，IoU 阈值固定 0.5。
checkpoint 使用 validation `report_f1`，并要求 `report_precision >= 0.80` 且
`report_recall >= 0.35`。

## 9. 数据影响

本次变更不要求重新运行 Isaac Sim。已有 tyx v0.3 raw 包含生成 21 维窗口特征、资源类型、
labor saturation、operational occupancy 和过程原因所需的证据。必须重新执行的是：

```text
bn_agg derived -> shared dataset -> B2-B5 training/evaluation
```

旧 dataset 和旧 checkpoint 对应不同任务，不能与新结果混用。
