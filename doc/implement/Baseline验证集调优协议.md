# Baseline 验证集调优协议

## 1. 目标与边界

本轮调优用于让 B2-B5 在各自模型定义内得到训练充分、概率校准合理的结果。目标不是让
所有 baseline 强行通过 BNPDFormer 的业务门，也不能为了接近主模型分数而改变标签或
评估公式。

以下实验契约冻结：

- 数据集：`factory_pdformer_134_v1`，134 个可训练 episode；
- split：现有整 episode train/validation/test 划分；
- 输入/输出：过去 30 个 60 秒窗口，预测未来 15 个窗口；
- 监督节点：machine、workbench、gantry、AGV；
- 真事件：最短 8 窗口，start 容差 3 窗口；
- 主指标：`report_f1`；
- 业务门：`report_precision >= 0.80` 且 `report_recall >= 0.35`；
- threshold：只在 validation 扫描，test 使用冻结值。

BNPDFormer 的双 will head、ongoing 强制、recall lift、分类型解码阈值和 precursor
加权不移植到 baseline。B3-B5 保持同一种单 event head 和同一种 loss 形式。

## 2. Test 隔离

候选运行必须使用 `--validation_only`，产物满足：

- `run_summary.json.status = validation_completed`；
- `metrics.json` 只有 `validation`；
- 不生成 `occupancy_events_test.csv` 或任何 `test_*` 汇总字段。

`select_baseline_tuning.py` 会拒绝包含 test 指标或缺少指定 seed 的候选。每轮搜索写入新的
`models/tuning/<tag>`，脚本不覆盖已有目录。

当前 seed42 test 已经用于探索性诊断，因此后续不能再根据这些 test 数字选择候选。若论文
需要严格的 untouched holdout，应在调优结束后锁定一组新的 episode，并让 BNPDFormer
与 B2-B5 全部在该 holdout 上只评估一次。

## 3. 统一搜索预算

每个 baseline 固定 8 个候选，每个候选训练 seed 42 和 43。神经模型的 epoch 上限相同为
60，但 min epoch 和 patience 可按结构设置。最终候选冻结后，正式结果用 seed 42/43/44
重训并报告均值与标准差。

候选排名同时使用 validation 两个 seed 的均值和稳定性：

1. 两个 seed 均通过业务门的候选优先；
2. `report_f1` 稳健分数，定义为均值减去总体标准差；
3. `report_f1` 均值；
4. `report_recall_upcoming` 均值；
5. `report_precision` 均值；
6. `hot_f1`、event will AP。

稳健分数在只有两个 seed 时等于两者中较低的 F1，用来防止单个幸运 seed
决定最终配置。主指标仍是 report F1，upcoming 只作次级判断。cause、remain MAE
和只在 who 命中样本上计算的 start MAE 不参与候选选择。

## 4. B2 搜索

B2 第一轮 `b2_search_v1` 显示，`negative_ratio=4, positive_weight=4` 的平均 F1
最高，但 seed42/43 的 F1 为 `0.1069/0.2679`，方差过大；权重降为 1 或只增加负
样本比例都会使 recall 降至约 `1%-4%`。因此第二轮 `b2_search_v2` 改为局部精调：

- negative cell ratio：为 4 时搜索 hot `scale_pos_weight` 2、2.5、3、3.5、4；
- negative cell ratio：为 8 时搜索 hot `scale_pos_weight` 4、6；
- tree depth：4、5；
- min child weight：3、5；
- lambda：5、10；
- report threshold：0.55 到 0.95。

`c0_anchor_r4_w4` 保留第一轮最优均值配置作为锚点，其余候选在高 precision
低 recall 和高方差两个端点之间插值。第一轮产物保留在 `models/tuning/b2_search_v1`，
第二轮写入 `models/tuning/b2_search_v2`，不覆盖旧结果。

第二轮仍由 `c0_anchor_r4_w4` 胜出，但稳健 F1 只有 `0.1069`，所有候选的 upcoming
recall 接近零。这说明限制来自“先预测 occupancy 再恢复事件”的输出形式，不再继续
第三轮同类参数搜索。

B2 现在补齐与 B3-B5 目标等价的 XGBoost `event_will/start/duration` 头：

- will 在所有有效工位节点上训练；
- start 只在 upcoming 正样本上训练；
- duration 只在事件正样本上训练；
- occupancy 头固定为前两轮的 `ratio=4, weight=4`，仅搜索 event will 正类权重
  `1/2/4/6/8/12/16`；
- 产物写入 `models/tuning/b2_event_search_v1`。

执行：

```bash
BENCHMARK_TAG=factory_pdformer_134_v1 TUNE_SEEDS="42 43" \
  ./batch_factory_baseline_tune_b2.sh
```

## 5. B3-B5 共同 event loss 搜索

三种神经模型使用相同三档 event loss 候选：

| 档位 | event loss | FP weight | upcoming weight | ongoing weight |
| --- | ---: | ---: | ---: | ---: |
| incumbent | 2.5 | 2 | 4 | 3 |
| balanced | 3.0 | 3 | 6 | 2.5 |
| strict | 3.0 | 4 | 7 | 2.5 |

提高 upcoming 权重时同步提高 FP 惩罚，避免单纯多报正例。start 继续使用 sigma=1 的高斯
软标签，不增加主模型专属 precursor 标签。

### B3 LSTM

搜索 hidden 64/96/128、node hidden 96/128、embedding 16/32、单/双层 LSTM、dropout
0.25/0.35、学习率和 weight decay。重点控制 751k 参数模型的过拟合。

### B4 GCN-GRU

保留现有输入残差和 LayerNorm，搜索 GCN 64/96、GRU 128/160、dropout 0.1/0.2/0.3、
学习率和 weight decay。现有 regularized 结果作为 `c0_incumbent`。

### B5 GAT-GRU

GAT 两层增加输入残差、层间残差和 LayerNorm，解决其相对 B4 缺少的基础优化稳定性；模型
仍是两层 GAT 加节点级 GRU。搜索 GAT 64/96、2/4 heads、GRU 128/160、dropout、学习率
和 weight decay。

该结构新增了可训练参数，旧 B5 checkpoint 不继续加载。旧实现和结果由 Git 历史提交
`0822ee1` 保留，本轮 B5 候选全部从头训练，不在主代码路径增加兼容兜底。

每次只运行一个模型：

```bash
BENCHMARK_TAG=factory_pdformer_134_v1 TUNE_SEEDS="42 43" DEVICE=cuda:0 \
  ./batch_factory_baseline_tune_neural.sh B4

BENCHMARK_TAG=factory_pdformer_134_v1 TUNE_SEEDS="42 43" DEVICE=cuda:0 \
  ./batch_factory_baseline_tune_neural.sh B5

BENCHMARK_TAG=factory_pdformer_134_v1 TUNE_SEEDS="42 43" DEVICE=cuda:0 \
  ./batch_factory_baseline_tune_neural.sh B3
```

建议顺序为 B2、B4、B5、B3。每轮先审阅 `selection.json` 和 `tuning_summary.csv`，再冻结
该模型正式配置；不根据当前 test 与 BNPDFormer 的差距追加候选。

## 6. 结果判定

优化后的 baseline 可以继续不通过 P80/R35 业务门。合理结果应满足：

- B2 不再依靠大量误报取得 recall；
- B3 相比当前 validation 结果有稳定改善，且不过早过拟合；
- B4 不低于当前 validation 主指标，并出现可观测 upcoming recall；
- B5 经过标准 GAT 稳定化后充分训练，是否超过 B4由 validation 决定；
- BNPDFormer 仍可凭编码器和专属事件解码能力取得领先。

无法在不损害 report F1 或 precision 的情况下提高 upcoming 时，应如实报告该结构的能力
边界，而不是加入主模型规则将其人为抬过业务门。
