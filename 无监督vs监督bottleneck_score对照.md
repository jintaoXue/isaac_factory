# 无监督 vs 监督 `bottleneck_score` 对照

> 日期：2026-09-05  
> 数据：`raw_data/dense_i1`（N=38，F=27，episode 划分 seed=42，窗 20826 / 4331 / 5354，局 140 / 28 / 36）  
> 主指标：`模型评估指标.md` 的 `report_*`（工位对且 `|Δstart|≤3` 分钟）  
> 配方：无监督 `factory_bn/configs/FactoryBN_dense_f1_p80.json`；监督 `factory_bn/configs/FactoryBN_dense_supervised_score.json`

---

## 0. 先回答：现在是不是无监督？

**是。现行主配方是无监督。**

`FactoryBN_dense_f1_p80.json` 里 `"train_mode": "unsupervised"`。占用真值 `y_hot` 来自过程启发式 `ops_hot_mask`（排队 / 阻塞 / 饥饿 / 到货延误等），**训练占用不读** `bottleneck_score`，也不用 TPM 转折点当正例。损失是 `_unsupervised_loss`：未来 X 重构 + 占用格 + 事件头 + 原因 + remain。`w_score=0`，日志里的 `score_mae=0` 表示没评分数回归，不是分数已经拟合好了。

以前那条「用 `bottleneck_score` 的监督」走另一条路径：`train_mode=supervised`，占用 `y_hot = node_hot_mask`（过程规则 ∪ 机台 `score≥0.55` ∪ TPM），损失再加 `w_score * SmoothL1(score_pred, y_score)`。旧格子配方（`ckpt_metric=hot_f1`，包 `n10_i1_20ep`）和这次 A.1 事件对照不是同一套数，不要混引。

---

## 1. 对照在比什么

同一套 A.1 事件契约、同一初始化、同一划分，只换训练目标：

| 项目 | 无监督臂 | 监督臂 |
| --- | --- | --- |
| 配置 | `FactoryBN_dense_f1_p80.json` | `FactoryBN_dense_supervised_score.json` |
| `train_mode` | `unsupervised` | `supervised` |
| 占用 y | `ops_hot_mask` | `node_hot_mask`（过程 ∪ 分数 ∪ TPM） |
| 额外损失 | `w_recon=0.2` 未来 X | `w_score=1.0` 分数回归 |
| 事件头 / 占用格 / 原因 / 对比学习 | 相同 | 相同 |
| STGNPP | 关 | 关 |
| 初始化 | `dense_i1_a1_v3ft/BNPDFormer_best.pt` | 同一份 |
| 存盘 | `ckpt_metric=report_f1`，P≥0.80 且 R≥0.55 | 相同 |
| 跑目录 | `libcity/cache/model_cache/dense_i1_a1_near5` | `.../dense_i1_a1_near5_sup` |

事件契约（两边相同）：H=15 分钟，占用格最短 8 分钟，报告最短 5 分钟，只计 `start≤2`，start 容差 3 分钟，监督节点 = 机台 / 工作台 / 龙门 / AGV。解码：事件头 ∪ 占用格连续段；**不是** persistence-only。

命令：

```text
python -m factory_bn.train --config factory_bn/configs/FactoryBN_dense_f1_p80.json \
  --data_dir raw_data/dense_i1 --save_dir libcity/cache/model_cache/dense_i1_a1_near5 \
  --device cuda --no_wandb --max_epoch 18

python -m factory_bn.train --config factory_bn/configs/FactoryBN_dense_supervised_score.json \
  --data_dir raw_data/dense_i1 --save_dir libcity/cache/model_cache/dense_i1_a1_near5_sup \
  --device cuda --no_wandb --max_epoch 18
```

标签重合统计：`python -m factory_bn.compare_train_modes --phase labels --data_dir raw_data/dense_i1`。

---

## 2. 关键发现：这份包上占用 y 两边一样

在 `dense_i1` 上，8 分钟平滑之后，`ops_hot_mask` 与 `node_hot_mask` **完全重合**（train / val / test Jaccard = 1.0）。官方事件条数也一样：

| 划分 | 窗数 | 事件（ops = score） | 进行中 | 尚未发生 |
| --- | ---: | ---: | ---: | ---: |
| train | 20826 | 2402 | 1944 | 458 |
| val | 4331 | 494 | 396 | 98 |
| test | 5354 | 661 | 549 | 112 |

原始格（平滑前、机台列）并不是「分数没用」，而是分数几乎被过程规则吃掉：

- `score≥0.55` 的机台格：319，**全部**已经在 `ops_occupancy_raw` 里。
- TPM 多出来的 484 格撑不满 8 分钟，平滑后消失。
- 因此本对照 **不是两种占用标签的比赛**，而是 **同一套事件真值下，重构未来 X 对回归 `bottleneck_score`**。

旧包上监督占用可能和过程占用分得开；`dense_i1` 上不能再假设「加上分数就会多一批正例」。

---

## 3. 主结果（官方 `report_*`）

两边都是 18 epoch、cosine、只解冻最后 2 个 encoder block。无监督 best = epoch 18；监督 best = epoch 11（之后 val F1 回落，没再过门）。

### 3.1 Validation（存盘用）

| | 无监督 epoch 18 | 监督 epoch 11 |
| --- | ---: | ---: |
| `rep_p` / `rep_r` / `rep_f1` | 0.808 / 0.605 / **0.692** | 0.803 / 0.601 / **0.688** |
| `up_r` / `on_r` | 0.378 / 0.662 | 0.388 / 0.654 |
| τ | 0.90 | 0.88 |
| `score_mae` | 不评 | 12.60 |

### 3.2 Test（冻结 best ckpt；`_epoch_loop` 仍会再扫一遍 τ）

| | 无监督 | 监督 | 差（监 − 无） |
| --- | ---: | ---: | ---: |
| **`rep_p`** | 0.801 | **0.817** | +0.016 |
| **`rep_r`** | **0.651** | 0.634 | −0.017 |
| **`rep_f1`** | **0.718** | 0.714 | −0.004 |
| `who_*` | 与 report 相同 | 与 report 相同 | 报对的工位 start 几乎都在 3 分钟内 |
| `up_r`（尚未发生） | **0.312** | 0.286 | −0.027 |
| `on_r`（进行中） | **0.719** | 0.705 | −0.015 |
| `start_mae` / `dur_mae` | 0.10 / 1.97 | 0.10 / 2.06 | 时长略差 |
| `n_pred` / `n_true` | 537 / 661 | 513 / 661 | 监督更少报 |
| τ | 0.85 | 0.88 | |
| 门（P≥0.80 且 F1 在 0.70–0.80） | **过** | **过** | |

数字来自各自目录的 `last_metrics.json["test"]`。`who_*` 与 `report_*` 一样，说明错主要在「报不报这个工位」，不在开始时刻。

---

## 4. 附录指标（不当主结论）

| Test | 无监督 | 监督 |
| --- | ---: | ---: |
| 占用格 P / R / F1 | 0.664 / 0.465 / 0.547 | **0.719** / 0.423 / 0.533 |
| IoU 事件 P / R / F1 | 0.739 / 0.417 / 0.533 | 0.744 / 0.421 / 0.537 |
| 机台 / 龙门 / AGV / 工作台 格 P | 0.614 / 0.696 / 0.510 / 0.777 | 0.683 / 0.727 / 0.542 / 0.820 |
| `remain_len` MAE | **10.5** | 12.1 |
| 过程原因 macro recall | 0.961 | 0.960 |
| `score_mae` | 不评 | 12.55 |

监督占用格 **更准、更少报**，和事件头「P 略高、R 略低」一致。分数回归本身没拟合好：`w_score=1.0` 把 train loss 从约 2 抬到约 8–10，`score_mae` 仍在 12 量级（分数原值尺度大，SmoothL1 没按占用那样归一）。它没有把「尚未发生」召回抬上去。

---

## 5. 怎么读

1. **现行主路径继续用无监督。** 同一 A.1 标签上，无监督 test F1 略高（0.718 vs 0.714），尚未发生召回也更好（0.312 vs 0.286）。监督只在精度上多 1.6 个点，靠少报换来的。
2. **在 `dense_i1` 上加 `bottleneck_score` 占用，不会改事件真值。** 分数热格是过程热格的子集；要比「两种标签」得换包，或先证明分数能独立出 ≥8 分钟段。
3. **监督这条臂真正多出来的是分数回归，对 A.1 没帮忙。** 编码器已经被占用 / 事件头拉着走，再塞一个尺度很大的 `y_score` 只会抢梯度。若还要做分数头，应先把 `y_score` 标准化或把 `w_score` 降到 0.05–0.2 再试；那是另一次实验，不是这次的结论。
4. **不要引用旧 s9 格子 P≈0.41 当这次对照。** 包、H、最短段、存盘指标都不同。
5. 两边都过了「P≈0.80 且 F1∈[0.70, 0.80]」。缺口仍是尚未发生：test 112 条里无监督只召回约 31%，监督约 29%。

---

## 6. 产物

| 文件 | 用途 |
| --- | --- |
| `factory_bn/configs/FactoryBN_dense_f1_p80.json` | 无监督现行配方 |
| `factory_bn/configs/FactoryBN_dense_supervised_score.json` | 本次监督对照（不覆盖现行配方） |
| `factory_bn/compare_train_modes.py` | 标签重合 / 双标签复评 |
| `libcity/cache/model_cache/compare_unsup_sup_labels.json` | 占用 / 事件重合 |
| `libcity/cache/model_cache/dense_i1_a1_near5/` | 无监督 ckpt + `last_metrics.json` |
| `libcity/cache/model_cache/dense_i1_a1_near5_sup/` | 监督 ckpt + `last_metrics.json` |

---

## 7. 现行 prefix8 契约上再对照一次

旧对照是 `event_min=5` 的 near5。正式模型换成 prefix8（upcoming≥8、进行中剩余≥1、prefix 解码）之后，监督配置 `FactoryBN_dense_supervised_score.json` 已经过时（min=5、无 prefix、`w_score=1.0`、从 v3ft 起）。

新监督臂对齐 prefix8：**同一套事件契约**，只换训练目标。占用标签在 `dense_i1` 上仍然 Jaccard=1.0，所以比的还是「重构未来 X」对「轻量分数回归」。

| | 无监督（正式） | 监督（新） |
| --- | --- | --- |
| 配置 | `FactoryBN_dense_f1_p80.json` | `FactoryBN_dense_prefix8_sup.json` |
| `train_mode` | unsupervised | supervised |
| 占用 y | `ops_hot_mask` | `node_hot_mask` |
| 额外损失 | `w_recon=0.2` | `w_score=0.15` |
| prefix / TPM / 事件头 | 有 | 同样有（监督路径已补上 `w_prefix` / `w_tpm`） |
| 初始化 | `dense_i1_a1_f180` | **prefix8 best**（微调 8 epoch） |
| best | epoch 8 | epoch 6 |
| 存盘 | `dense_i1_a1_prefix8` | `dense_i1_a1_prefix8_sup` |

### 7.1 Validation

| | prefix8 无监督 | prefix8 监督 |
| --- | ---: | ---: |
| P / R / F1 | **0.883 / 0.881 / 0.882** | 0.875 / 0.882 / 0.879 |
| `up_r` / `on_r` | 0.633 / **0.920** | **0.653** / 0.918 |

### 7.2 Test

| | prefix8 无监督 | prefix8 监督 | 差（监 − 无） |
| --- | ---: | ---: | ---: |
| **P / R / F1** | **0.852 / 0.856 / 0.854** | 0.855 / 0.846 / 0.850 | F1 **−0.004** |
| `up_r` / `on_r` | 0.455 / **0.911** | **0.473** / 0.896 | +0.018 / −0.015 |
| `remain_len_mae` | **10.5** | 10.7 | |
| 原因 acc / macro | 0.965 / 0.971 | **0.969 / 0.975** | |
| `score_mae` | 不评 | 14.3 | 分数头仍没拟合好 |

结论和 near5 那次一样：**监督没有赢。** 轻量 `w_score=0.15` 避免了上次 `w_score=1.0` 把 loss 抬爆，但 test F1 仍低 0.004，进行中召回掉一点，尚未发生只多 2 个点。正式模型继续是 **prefix8 无监督**。
