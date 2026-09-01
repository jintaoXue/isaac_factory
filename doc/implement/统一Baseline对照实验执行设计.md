# 统一 Baseline 对照实验执行设计

## 1. 当前状态

- 状态：已按主实验最新口径实现，等待服务器重建与 smoke test
- 主实验参考：`dev_tyx@7b2fc02`
- raw 契约：`collector_version=v0.3`
- derived 契约：`tyx_bn_agg_unsupervised_v2`
- dataset：`factory_baseline_dataset_v3`
- 预测目标：`factory_ops_event_30m_to_15m_v1`
- baseline：B2 XGBoost、B3 LSTM、B4 GCN-GRU、B5 GAT-GRU

本文只描述当前主路径。旧的 BSTAN weak v1/v2、12 窗口输入、预测到生产结束的
512 窗口目标均不在当前代码路径保留。

## 2. 公平对照原则

B2-B5 与 PDFormer 固定使用同一组：

```text
raw episode
bn_agg 窗口特征和原因字段
operational occupancy 目标
episode split
节点有效性 mask
预测时域
事件恢复与评分规则
```

模型之间仅允许编码器结构和模型自身超参数不同。扰动日志是上下文和原因证据，不能直接
创建瓶颈正标签。

## 3. 数据链路

```text
tyx raw v0.3
  -> raw quality gate
  -> dev_tyx tools/bn_agg（supervised 模式仅提供 features/cause）
  -> canonical 27 维节点输入
  -> ops_hot_mask 生成无监督 operational occupancy
  -> episode-level 70/15/15 split
  -> B2-B5 shared dataset.pt
```

当前 PDFormer 134-episode cohort 的 raw quality gate 与主实验一致：订单必须完成，
含 `deadlock_reset` 的 episode 必须排除。若生产完成时某个运行期扰动已经 START、但尚未
产生 END，则它属于 episode 右边界上的右删失区间：保留该 episode，将区间截断到最后
可观测时刻，并记录 `runtime_disturbance_right_censored=<event_id>` warning。该 warning
不创建瓶颈标签，也不构成 episode 拒绝原因。

这 9 个 raw 目录共含 142 个 episode；8 个未完成 episode 被门禁排除，最终与 PDFormer
使用相同的 134 个 episode。构建正式数据集前必须同时核对 episode 名称集合，不能只核对
总数。

共享派生目录为：

```text
<raw_run>/shared_bn_agg_unsupervised_v2/episode_XX/env_YY/
```

共享 benchmark 目录为：

```text
output/bottleneck_dataset/experiments/<BENCHMARK_TAG>/
```

## 4. 输入定义

固定参数：

```text
window_size_s               = 60
stride_s                    = 60
input_windows               = 30
input_history_s             = 1800
occupancy_horizon_windows   = 15
occupancy_horizon_s         = 900
hot_min_windows             = 8
hot_gap_windows             = 1
event_report_threshold      = 0.65
event_start_tolerance       = 3 windows
event_iou_min               = 0.5
```

每个节点输入为 27 维：

```text
21 个 bn_agg 连续特征
+ 5 个资源类型 one-hot
+ labor_saturated
```

图保留 machine、gantry、human、transport_robot 和 buffer。A.1 监督节点只包括
machine、gantry 和 transport_robot；human 与 buffer 仍作为上下文节点参与模型编码。

## 5. 目标定义

对 anchor `t`：

```text
history = [t-30, t)
future occupancy = [t, min(t+15, jobs_done))
```

`ops_hot_mask` 只使用运行证据，不使用 `bottleneck_score_s`、turning-point 直接标签或
扰动类型标签。连续 hot run 至少 8 个窗口，允许填补 1 个窗口空洞。

模型共同预测：

1. `y_hot[K,N]`：未来 15 个窗口的站点占用格；
2. `event_will[N]`：该站点是否存在满足持续性要求的事件；
3. `event_start[N]`：事件起始窗口；
4. `event_duration[N]`：事件持续窗口数；
5. `remain_len`：距离订单全部完成的窗口数；
6. `cause`：当前可观测过程原因辅助任务，忽略注入名称和 score fallback。

旧的全局 `will/mark/time-to-start` 不作为正式主指标。

## 6. Split 与归一化

同一 raw run 内先以 seed 42 打乱完整 episode，再按 70%/15%/15% 分为
train/validation/test。一个 episode 的窗口只能进入一个 split。每个 raw run 至少需要
3 个通过门禁的 episode。

split 不读取正负标签或原因分布。连续特征归一化、类别权重等统计只能从 train 拟合。

## 7. 训练与评分

PyTorch B3-B5 的主 loss 权重与主实验一致：

```text
occupancy hot       0.5
remain length       0.4
cause               0.1
event will          2.5
event start         1.5
event duration      1.0
```

checkpoint 主指标为 validation `report_f1`，并要求 `report_precision >= 0.80`。
正式报告至少输出：

- occupancy precision/recall/F1；
- who precision/recall/F1；
- report precision/recall/F1；
- start MAE、duration MAE；
- ongoing/upcoming 分组 recall 和 MAE；
- occupancy event IoU 指标；
- remain length MAE；
- 六类过程原因指标。

B2 没有神经事件头，使用其未来 occupancy 概率恢复 station event，并以同一事件规则评分。

## 8. 服务器执行

从 PDFormer 使用的 9 个 raw 目录重新生成 derived 和 dataset。该 cohort 含 142 个 raw
episode，其中预期 134 个通过门禁，因此显式关闭“零拒绝”模式并锁定入选数量：

```bash
RAW_ROOT="$HOME/work/BNPDFormer/_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset"

BENCHMARK_TAG=factory_pdformer_134_v1 \
STRICT_RAW=0 EXPECTED_ACCEPTED_EPISODES=134 \
  ./batch_factory_baseline_build.sh \
  "$RAW_ROOT/extra_machine" \
  "$RAW_ROOT/extra_human" \
  "$RAW_ROOT/extra_logistics" \
  "$RAW_ROOT/extra_material" \
  "$RAW_ROOT/unsup_n10_i1/n10_machine1.0" \
  "$RAW_ROOT/unsup_n10_i1/n10_human1.0" \
  "$RAW_ROOT/unsup_n10_i1/n10_logistics1.0" \
  "$RAW_ROOT/unsup_n10_i1/n10_material1.0" \
  "$RAW_ROOT/material重采"
```

先做四模型 smoke：

```bash
BENCHMARK_TAG=factory_pdformer_134_v1 RUN_MODE=smoke DEVICE=cuda:0 \
  ./batch_factory_baseline_train.sh ALL
```

smoke 通过后正式训练：

```bash
BENCHMARK_TAG=factory_pdformer_134_v1 RUN_MODE=formal DEVICE=cuda:0 \
  MAX_EPOCHS=100 PATIENCE=15 ./batch_factory_baseline_train.sh ALL
```

## 9. 验收条件

1. manifest 显示 30 个输入窗口、15 个未来窗口和 27 维特征；
2. `target_node_category=machine_gantry_agv`；
3. train/validation/test episode 无交集；
4. `y_hot` 在 human/buffer 列始终不参与 loss 和 metrics；
5. 所有 baseline 输出 `station_report` 与 `occupancy_event`；
6. test 只使用 validation 阶段冻结的模型和固定阈值；
7. raw 不需要重新采集，只重建 derived、dataset 和模型。
