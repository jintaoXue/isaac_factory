# Baseline 评估指标对齐说明

## 1. 当前参考

2026-09-11 重新 fetch，`origin/dev_tyx` 为 `20c40e2`（2026-09-06）。
以该提交的 `PDFormer/factory_bn/train.py::_epoch`、`remain.py` 和
`FactoryBN_dense_f1_p80.json` 为共同评估参考，不用旧文档猜配置。

当前张量版本 `factory_baseline_dataset_v6`，事件目标版本
`factory_ops_event_30m_to_15m_dense_v2`，指标契约 `factory_dense_i1_eval_v1`。
`factory_baselines/evaluation.py` 是基线侧事件参数与单位说明的统一入口。
v5 和更早结果留在 Git 历史及归档文件；当前训练入口拒绝旧契约，不静默兼容。

## 2. 共同任务

| 项目 | 最新主实验 / B2–B5 共同口径 |
|---|---|
| 输入 / 预测网格 | 30 个历史窗口 / 15 个未来窗口，窗口 60 秒 |
| 目标资源 | machine、workbench、gantry、AGV；human/buffer 仅作上下文 |
| hot 标签 | 操作状态规则，最短 8 窗口，补 1 窗口间隔 |
| 每站事件 | 未来网格内最长连续 hot 段，等长取最早 |
| upcoming 真事件 | 长度至少 8 窗口，起点不晚于未来索引 2 |
| ongoing 真事件 | 最后历史窗口 hot 且未来段从索引 0 开始，剩余长度至少 1 窗口 |
| who 命中 | 工位正确且预测/真实均有事件 |
| report 命中 | who 命中且起点误差不超过 3 窗口 |
| mask | 未观测未来不计；非目标节点不计 |
| hot 概率阈值 | 0.55，仅用于网格附录指标 |
| 默认事件阈值 | 0.70 |
| validation 扫描 | 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.94, 0.98 |
| 阈值选择 | 优先 precision ≥ 0.80 内最高 report F1；否则最高 F1 |
| checkpoint 可行门 | P ≥ 0.80 且 R ≥ 0.70 |
| test | 冻结 validation 阈值，不在 test 搜索 |

持续不足 8 分钟的 ongoing 尾段现在也计入真事件，较晚开始的 upcoming 被排除。
这改变了任务和分母，不能将旧规则下约 0.3 的 F1 与新 dense F1 直接排序。
起点范围 0–2 小于容差 3，也意味着“始终预测起点 0”可能通过时间门；因此必须同时报告
upcoming recall、起点 MAE 和 ongoing/upcoming 真事件数，不能只看总 F1。

B4/B5 保持 GCN-GRU / GAT-GRU。主模型的 prefix 解码、双 will 头、分类型阈值与
强制提置信度不作为“指标同步”复制进基线。结构、训练优化和后处理的区别要独立记录。

## 3. MAE 与输出

最新提交没有新增另一种必须追逐的 MAE 名称，主要 MAE 含义如下：

| 字段 | 真实含义 / 统计样本 |
|---|---|
| `start_mae` | 起点误差，所有 who 真阳性；不是仅 report 命中 |
| `dur_mae` | 同一批 who 真阳性的事件持续长度误差 |
| `start_mae_ongoing/upcoming` | 对应真事件子集内的 who 真阳性起点误差 |
| `dur_mae_ongoing/upcoming` | 对应真事件子集内的 who 真阳性持续长度误差 |
| `remain_len_mae` | 所有样本锚点到订单全部完成的剩余时间误差，不是事件 duration |
| `score_mae` | 基线辅助 score 网格误差；主无监督训练未计算该项，不能直接横比 |

原始同名时间值单位为 window。v6 同时输出 `*_seconds` / `*_minutes`，按数据窗口长度
换算；当前 60 秒窗口下 window 数值等于分钟数。增加 `time_mae_sample_count` 及两个子集
计数、`remain_len_mae_sample_count`、`n_matched_who` / `n_matched_report`。
无 who 匹配时保留上游原键的 0，但显式单位字段为 JSON null，样本数为 0；报告应显示 N/A，
不能把“没有预测成功”解读成“MAE 为 0，预测完美”。

仍输出 who/report P/R/F1、hot P/R/F1/AP 与资源分项、IoU event 指标、独立事件头 AP、
过程原因 macro recall/accuracy 和 train 多数类基线。MAE 子集不能代替总 recall。
契约与选定阈值随 manifest、checkpoint、summary 和 metrics 保存。

## 4. 上游待讨论点

- `模型评估指标.md` 仍写 recall 门 0.35 且描述不限制起点，最新配置实际是 0.70 / 2。
- `hot_eval_threshold=0.55` 才是网格评价阈值；`event_union_hot_threshold=0.45`
  属于主模型事件融合解码，不能误作共同 hot 指标阈值。
- `train.py` 传入 ongoing 最短 1；部分独立评估脚本未传该参数，可能重算出不同分母。
  本轮明确对齐训练入口，不静默混用独立评估脚本默认值。
- `hist_last_hot` 来源是全 episode 平滑后的标签，可能依赖锚点之后的信息。
  基线不把它输入编码器，但共同指标/起点解码仍使用它。本轮数值是该离线契约下的结果，
  不宣称已证明严格在线预测能力。因果前缀评估须固定真值、只改变可见解码信息后另做对照。
- 主无监督 `_epoch` 的 `score_mae` 保留零累加器，却未实际累计 score 误差；打印的 0
  不是准确率证据。基线自己测到的辅助 score MAE 保留，但标明不可横比。

## 5. 数据与复现

主实验文档记录 dense_i1 共 204 episode、140/28/36 split，但实际数据包和精确 raw 清单
尚未从共享服务器目录找到。不能把自行扩充的数据叫作“已复现主实验 204”。

`rebuild_dense_factory_benchmark.py` 在现有 BSTAN benchmark 目录完成：

1. 原始完整性、完成作业、死锁等质量门；不按标签正例率或模型指标选样本。
2. 核对旧 raw 哈希；保留所有旧 train/validation/test episode 的归属，只划分新增 episode。
3. 使用同一 bn_agg 在内存重聚合，用 canonical exporter 冻结共同 bundle，再构建基线张量。
4. 用带 SHA-256 清单且验证成功的 ZIP 文件保留旧张量、索引和配置，不新建实验目录。
5. 逐样本核对 canonical 导出的输入、锚点、mask、hot、事件目标、cause 和作业剩余量；
   完成后才发布通过状态。未通过的 manifest 不能进入训练。

目前只读审计确认新增 55 个组合扰动全部通过，norm20 为 19/20；合计 74 个新增候选。
原 134 加这批候选预计为 208，最终以完整重建审计为准。新 cohort 需要主模型使用同一份
bundle 和冻结 split 才能作正式横向对照；当前主文档的 dense_i1 分数只能作为背景参考。

`batch_factory_baseline_dense.sh` 记录原目录 audit/build/B4/B5/ALL 命令。
`train_dense_baseline_control.py` 只复用已有输出目录，先验证归档旧权重与指标，
再从头训练 seed42/43 的 history 控制组，只评估 validation。不把旧权重续训混作新控制组。
所有输出留在 BSTAN 仓库，脚本不修改主实验目录，也不创建新实验目录。

## 6. 验证状态

已对 100 组随机输入逐项对比最新 `dev_tyx` 的事件目标、who/report、MAE，结果完全一致
（不启用模型专属强制解码）。测试覆盖 ongoing 短尾、起点 2/3 边界、节点/尾窗 mask、
who 与 report 的 MAE 分母差别、空匹配 N/A、CSV/内存导出数组一致、归档与 split 保留。
相关本地测试共 130 项通过，另有 12 个子测试通过；还核对了上游配置中的实际评估阈值。
服务器扩容重建和新训练尚未执行；后续实际结果记录在《Baseline迭代实验报告》中。
