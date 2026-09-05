# 统一 Baseline 对照实验执行设计

## 1. 当前状态

- 状态：共享 dataset 和首轮正式训练已通过；评估协议已对齐主实验最新定义，等待服务器重评/重训
- 主实验参考：`dev_tyx@52e8643` 的 `模型评估指标.md`
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

模型之间允许编码器结构和与该结构匹配的优化超参数不同；模型容量、超参数和训练预算
必须随产物披露。扰动日志是上下文和原因证据，不能直接创建瓶颈正标签。

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
event_report_threshold      = 0.68（validation 扫描的起始值）
report_threshold_sweep      = 0.55,0.60,0.62,0.65,0.68,0.70,0.72,0.75,0.78,0.80,0.82,0.85
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

PyTorch B3-B5 使用同一套 baseline 多任务 loss，以便主要差异落在编码器上：

```text
occupancy hot       0.5
occupancy Dice      0.25
occupancy IoU       0.25
remain length       0.4
cause               0.1
event will          2.5
event start         1.5
event duration      1.0
```

occupancy loss 只在 `remain_mask & occ_node_mask` 上计算，human/buffer 不再
充当负样本。BCE/Dice/IoU 按 machine、workbench、gantry、AGV 分类取均值，
并使用 B3-B5 共同冻结的 baseline 类型权重：

```text
positive: machine=4, workbench=2, gantry=1, AGV=4
negative: machine=1, workbench=2, gantry=2, AGV=2
```

event will loss 同样按四类资源分别归一化后取均值；start 使用与主实验一致的
`sigma=1` 高斯软标签，duration 对窗口数执行 `log1p` 后计算 Smooth L1。

这组权重不是 BNPDFormer 配方的复制，也不是评估协议的一部分。BNPDFormer 当前配方的
reconstruction、contrastive loss、分类型 loss、encoder warm-start、双 will head、
ongoing 强制和 recall lift 都属于主模型专属训练或解码能力，不移植到 baseline。

当前 `modelnote.md` 记录的同口径 PDFormer test 锚点为 report
`precision=0.817`、`recall=0.447`、`F1=0.578`。该数值用于正式实验后的横向核对；
最终表格仍应从同一份 134-episode dataset 对应的模型产物中自动汇总，不能只引用文档数字。

正式训练使用预先冻结的 `baseline_fair_v1` 配置。各模型共享数据、监督目标、loss 形式、
validation checkpoint 规则和评估公式，但优化器参数按结构分别设定：

| 模型 | 容量/树配置 | batch | max/min/patience | 学习率 | weight decay | dropout |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| B2 XGBoost | 500 trees, depth 5, min child 3, lambda 5 | - | - | 0.03 | - | - |
| B3 LSTM | hidden 128, node hidden 128, node embedding 32 | 32 | 60/15/15 | 3e-4 | 1e-3 | 0.25 |
| B4 GCN-GRU | GCN 64, GRU 128 | 24 | 60/15/20 | 3e-4 | 1e-3 | 0.10 |
| B5 GAT-GRU | GAT 64×4 heads, GRU 128 | 16 | 60/15/20 | 1.5e-4 | 1e-2 | 0.20 |

三种神经模型都使用 AdamW、gradient clip 1.0 和 cosine schedule (`lr_min=1e-6`)。
在当前 34 节点、27 特征数据上，B3/B4/B5 默认可训练参数量约为 700k/221k/232k；
参数量不强行做成完全相等，因为 B3 的节点展平输入本身就是非图基线定义的一部分，但每次
训练都会把实际参数量写入 `config.json` 和 `run_summary.json`。

这些超参数不从 test 反推。若后续要做超参数搜索，应为每个模型分配相同数量的候选配置，
只按 validation `report_f1` 和门槛选定，最终 test 只运行一次。不能为了让 baseline 达到
某个预设分数而查看 test 后继续调参。

checkpoint 主指标为 validation `report_f1`，并要求
`report_precision >= 0.80` 且 `report_recall >= 0.35`。每个 epoch 先在 validation
扫描统一阈值集合，在满足 precision 门槛的候选中选择最高 report F1，并将该 epoch
选中的阈值写入 checkpoint。达到双门槛后只在可行 epoch 中选最高 F1；若整轮均
未达到双门槛，则明确记录 `checkpoint_constraint_met=false`，并使用 validation F1
最佳 epoch 作诊断结果。report F1 打平时依次以更高的 occupancy hot F1 和更低的
validation total loss 决胜，避免事件指标全零时错误固定在 epoch 1；early stopping
使用同一排序。occupancy 评估阈值固定为 0.55；test 不再扫描阈值，严格使用所选
checkpoint 在 validation 上确定的阈值。

`event_will` 同时记录不参与 checkpoint 选择的 PR-AUC、ROC-AUC 和预测概率分位数，
用于区分“模型没有排序能力”和“概率整体尚未越过固定报告阈值”。
正式报告至少输出：

- occupancy precision/recall/F1；
- occupancy AP、真实/预测正例率和 machine/workbench/gantry/AGV 分类型指标；
- who precision/recall/F1；
- report precision/recall/F1；
- start MAE、duration MAE；
- ongoing/upcoming 分组 recall 和 MAE；
- occupancy event IoU 指标；
- remain length MAE；
- 六类过程原因的 per-class recall、macro recall，以及仅从 train 计算的多数类基线。

B2 没有神经事件头，validation 对未来 occupancy 概率使用同一阈值集合恢复 station
event，再按同一 who/report 规则选择阈值；test 冻结该阈值。B3-B5 直接扫描其
`event_will_probability`。模型解码方式可以不同，但 8 分钟最短段、3 分钟 start 容差、
监督节点 mask 和指标公式保持一致。

B4 的两层 GCN 使用输入投影残差和逐层 LayerNorm。工厂图连接较密时，纯邻居平均会令
不同工位表示快速趋同，造成按工位 event head 全零；残差保留工位自身状态，GCN 分支仍
负责传播拓扑上下文，因此模型定义仍是标准 GCN-GRU baseline。

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

smoke 通过后，优先单独重跑修正后的 B4：

```bash
BENCHMARK_TAG=factory_pdformer_134_v1 RUN_MODE=formal DEVICE=cuda:0 \
  ./batch_factory_baseline_train.sh B4
```

B4 validation 曲线和产物通过后再运行完整一轮：

```bash
BENCHMARK_TAG=factory_pdformer_134_v1 RUN_MODE=formal DEVICE=cuda:0 \
  ./batch_factory_baseline_train.sh ALL
```

## 9. 验收条件

1. manifest 显示 30 个输入窗口、15 个未来窗口和 27 维特征；
2. `target_node_category=machine_gantry_agv`；
3. train/validation/test episode 无交集；
4. `y_hot` 在 human/buffer 列始终不参与 loss 和 metrics；
5. 所有 baseline 输出 `station_report` 与 `occupancy_event`；
6. test 只使用 validation 阶段冻结的模型和固定阈值；
7. raw 不需要重新采集，只重建 derived、dataset 和模型；
8. 每个产物记录 `training_profile=baseline_fair_v1`、完整模型配置和神经模型参数量；
9. 正式论文结果固定配置后至少运行 seed 42/43/44，报告均值和标准差；同一 seed 的
   episode split 不变，训练随机种子变化。
