# Baseline 十二组对齐实验结果（Test 完整指标）

日期：2026-10-02

## 1. 运行状态

本文件整理 B2（XGBoost）、B3（LSTM）、B4（GCN+GRU）和 B5（GAT+GRU）在
Start≤5、10、15 三档上的冻结 Test 评估，共 12 组任务。

- `status`: `frozen_baseline_test_evaluation_completed`
- `test_evaluated`: `True`
- `training_performed`: `False`
- `task_count`: `12`
- Test 评估代码提交：`24002e25cb1d2b679d290046e2146d20afa938ef`
- Test 报告：`baseline_matched_v1_test_20261001_metrics_v3.json`
- Test 日志：`baseline_matched_v1_test_20261001_metrics_v3.log`

评估只加载 validation 已经选定的 checkpoint 和阈值，没有重新训练，也没有使用 Test
重新选择阈值。

## 2. 共同评估口径

12 组实验使用冻结的 `factory_dense_i1_eval_tyx_7b2ab39_v1` 契约：30 个历史窗口、20
个未来网格、Min8、起点容差 3 个窗口、validation 选择阈值后冻结到 Test。`will15_*`
在该协议中是 `who_*` 的同义字段，不表示固定的 15 分钟目标。

`start_mae` 和 `dur_mae` 的原始单位是窗口；当前每个窗口为 60 秒，因此报告同时保留
秒和分钟字段。没有 who 匹配的子集应按样本数和 N/A 解释，不能把空匹配的 0 当作预测
准确。

## 3. Test 整体事件结果

下表数值为原始比例，便于和 JSON 逐项核对。`will15` 与 `who` 在当前匹配协议下相同；
`report_f1` 还要求起点误差满足 report 容差。

| 模型 | Start≤ | 阈值 | will15 P | will15 R | will15 F1 | who F1 | report F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 | 5 | 0.80 | 0.858507 | 0.597583 | 0.704667 | 0.704667 | 0.701817 |
| B3 | 5 | 0.55 | 0.679681 | 0.515408 | 0.586254 | 0.586254 | 0.582818 |
| B4 | 5 | 0.50 | 0.815744 | 0.569789 | 0.670936 | 0.670936 | 0.668801 |
| B5 | 5 | 0.70 | 0.829372 | 0.590332 | 0.689728 | 0.689728 | 0.686904 |
| B2 | 10 | 0.80 | 0.858009 | 0.486978 | 0.621317 | 0.621317 | 0.611285 |
| B3 | 10 | 0.60 | 0.560513 | 0.450614 | 0.499591 | 0.499591 | 0.477254 |
| B4 | 10 | 0.90 | 0.880906 | 0.439803 | 0.586693 | 0.586693 | 0.579482 |
| B5 | 10 | 0.85 | 0.848718 | 0.487961 | 0.619657 | 0.619657 | 0.609048 |
| B2 | 15 | 0.80 | 0.845034 | 0.452960 | 0.589782 | 0.589782 | 0.577233 |
| B3 | 15 | 0.50 | 0.623563 | 0.398348 | 0.486138 | 0.486138 | 0.461495 |
| B4 | 15 | 0.94 | 0.788256 | 0.406609 | 0.536482 | 0.536482 | 0.517711 |
| B5 | 15 | 0.90 | 0.853113 | 0.402478 | 0.546929 | 0.546929 | 0.534456 |

## 4. Test upcoming、时间和原因指标

`strict hits` 是 upcoming 中同时满足 who 命中和起点容差的 report 命中；`upcoming
who recall` 只要求 who 命中。`start_mae`、`dur_mae` 和 `remain_primary` 的单位均为
窗口，当前每个窗口等于 1 分钟。

| 模型 | Start≤ | upcoming strict hits / support | upcoming who recall | upcoming report recall | start MAE | dur MAE | remain primary MAE | cause acc | cause macro recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 | 5 | 27 / 483 | 0.060041 | 0.055901 | 0.030334 | 3.264480 | 4.446202 | 0.990323 | 0.987534 |
| B3 | 5 | 45 / 483 | 0.099379 | 0.093168 | 0.056272 | 3.855116 | 8.839986 | 0.937928 | 0.920129 |
| B4 | 5 | 22 / 483 | 0.045549 | 0.045549 | 0.020148 | 3.321977 | 14.484229 | 0.900165 | 0.882087 |
| B5 | 5 | 44 / 483 | 0.095238 | 0.091097 | 0.042989 | 3.564874 | 10.867311 | 0.957045 | 0.951641 |
| B2 | 10 | 32 / 848 | 0.041274 | 0.037736 | 0.122099 | 3.209577 | 4.446202 | 0.990323 | 0.987534 |
| B3 | 10 | 66 / 848 | 0.112028 | 0.077830 | 0.388222 | 3.768067 | 8.571753 | 0.935804 | 0.930332 |
| B4 | 10 | 37 / 848 | 0.044811 | 0.043632 | 0.117318 | 3.237546 | 15.359733 | 0.906538 | 0.899948 |
| B5 | 10 | 44 / 848 | 0.056604 | 0.051887 | 0.153072 | 3.533372 | 9.936371 | 0.958697 | 0.950773 |
| B2 | 15 | 33 / 987 | 0.038501 | 0.033435 | 0.167173 | 3.169683 | 4.446202 | 0.990323 | 0.987534 |
| B3 | 15 | 71 / 987 | 0.102330 | 0.071935 | 0.452765 | 3.615068 | 7.275809 | 0.944300 | 0.922290 |
| B4 | 15 | 36 / 987 | 0.053698 | 0.036474 | 0.302483 | 3.120506 | 11.082990 | 0.906538 | 0.899725 |
| B5 | 15 | 39 / 987 | 0.043566 | 0.039514 | 0.222348 | 3.421640 | 9.370792 | 0.951381 | 0.935169 |

## 5. 完整 JSON 指标覆盖

终端 `TEST_STAGE_COMPLETE` 行只打印上述 headline 指标；完整结果以每个 task 下的
`test` 对象为准。当前 Test JSON 已实际包含以下指标组：

- 工位事件：who/report P/R/F1、will15 P/R/F1、ongoing/upcoming recall、真实样本数、
  who/report 命中数、起点 exact accuracy、onset bucket accuracy、容差 1/2/3 的
  precision/recall/F1。
- 时间误差：整体、ongoing、upcoming 的 start/duration MAE，以及 seconds、minutes 和
  sample count。
- 剩余时间：普通 MAE、progress-weighted MAE、early/middle/late MAE、各阶段 weighted
  MAE、middle-weighted MAE、primary MAE、秒、分钟和样本数。
- Hot 网格：总体及 machine、workbench、gantry、AGV 的 precision/recall/F1、AP、
  正例率和预测正例率，并保留 hot threshold 与 type harmonic mean。
- 事件附录：event precision/recall/F1/count、`event_will` 和 `occupancy_event`。
- 过程原因：accuracy、macro recall、四个原因类别 recall、majority accuracy 和样本数。

## 6. 结果解读边界

本批 Test 结果已经和冻结的 `dev_tyx@7b2ab39` 主实验评估契约对齐。最新的
`origin/dev_tyx@bbba50a` 还包含主模型专用的 machine/logistics 子集、onset head 附加
指标和辅助 checkpoint 结果；这些不属于本次冻结 baseline 匹配协议，不能混入本表解释。

整体 F1 较高而 upcoming recall 较低的组合是当前模型在 Test 上的实际表现，应在正式结果
中同时报告整体事件指标、ongoing/upcoming 分项、起点和持续时间 MAE，以及原因识别指标。
`score_mae` 仅作为 baseline 辅助指标保存，不能直接与主模型的无监督 `score_mae` 做
横向结论。

## 7. 来源

- Train/validation 汇总：`doc/experiments/baseline_matched_all_20260915_report_data.json`
- Test 原始报告：`baseline_matched_v1_test_20261001_metrics_v3.json`
- Test 日志：`baseline_matched_v1_test_20261001_metrics_v3.log`
- 评估代码：`source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/evaluate_frozen_baseline_test.py`
