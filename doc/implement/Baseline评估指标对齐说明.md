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
| ongoing 短尾特例 | 最后历史窗口 hot 且未来段从索引 0 开始，剩余长度至少 1 窗口 |
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
报告中的 ongoing/upcoming 子集实际按真值 start==0 / start>0 划分；历史最后一窗不 hot、
但未来索引 0 起足够长的事件，也会进入 ongoing 指标。因此不能把该指标全部解释为
“锚点前已发生事件”的识别率；历史 hot/cold 的细分应看独立诊断。
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

完整重建审计确认新增 55 个组合扰动全部通过，norm20 为 19/20；合计新增 74 个 episode。
原 134 加这批数据为 208，217 个检查对象中拒绝 9 个。新 cohort 需要主模型使用同一份
bundle 和冻结 split 才能作正式横向对照；当前主文档的 dense_i1 分数只能作为背景参考。

`batch_factory_baseline_dense.sh` 记录原目录 audit/build/B4/B5/ALL 命令。
`train_dense_baseline_control.py` 只复用已有输出目录，先验证归档旧权重与指标，
再从头训练 seed42/43 的 history 控制组，只评估 validation。不把旧权重续训混作新控制组。
所有输出留在 BSTAN 仓库，脚本不修改主实验目录，也不创建新实验目录。

## 6. 验证状态

已对 100 组随机输入逐项对比最新 `dev_tyx` 的事件目标、who/report、MAE，结果完全一致
（不启用模型专属强制解码）。测试覆盖 ongoing 短尾、起点 2/3 边界、节点/尾窗 mask、
who 与 report 的 MAE 分母差别、空匹配 N/A、CSV/内存导出数组一致、归档与 split 保留。
相关本地测试共 138 项通过，另有 12 个子测试通过；还核对了上游配置中的实际评估阈值。
服务器已完成扩容重建，36184 个实际样本全部通过导出一致性检查。B4/B5 四次控制训练已完成，
实际进度见《Baseline迭代实验报告》第 24 节及 `baseline_dense_v6_20260911.json`。

## 7. 正式比较的执行门（2026-09-11 补充）

目标是同一任务下给各模型充分、合理的训练机会，不是通过换数据、改事件定义或放松
匹配标准得到一个接近主模型的数字。指标代码对齐不等于整个实验已对齐。

| 必须冻结并共同使用 | 允许按模型在 validation 上调优 |
|---|---|
| 同一 raw/导出包身份、质量筛选、去重和 episode split | 学习率、batch size、优化器和学习率调度 |
| 同一可用历史信息、窗口/步长、预测范围、节点与时间 mask | 隐藏维度、层数、dropout、weight decay |
| 同一标签、事件门、指标分母、MAE 单位、阈值搜索规则 | 多任务权重、训练集抽样和困难负例策略 |
| 同一 test 使用规则和预先登记的确认 seed | 模型适配的早停；记录实际训练量与搜索预算 |

正式结果保存 bundle 哈希、逐 episode split、最终配置和初始化来源。不能以“集数接近”
或同名目录代替数据身份核对。若采用扩容后的 208 包，主模型也必须在该包和冻结 split 上
重跑；若复现已发表在项目文档中的 204 包成绩，必须先拿到其实际包及划分。

训练阶段也要可审计：主模型 dense 配方接续既有权重，其之前阶段的训练数据和预算不能
被忽略。基线可使用自身的预训练/续训，但只用允许的训练数据，不能加载主模型权重或
借用 validation/test 做梯度训练。报告完整训练量，而不是只比较最后 16/60 个 epoch。
骨干之外的事件头、辅助损失和后处理作为模型方案的一部分披露；若结论仅指骨干优劣，
另做共同预测头与解码器的受控对照。

合理 baseline 的验收包括：无数据泄漏和 mask/归一化错误、事件头可学习、有限搜索后
多 seed 结果稳定、precision/recall 与 upcoming/ongoing 都披露、MAE 带支持数。不能预先
保证必须达到某个 F1，也不能因未达主模型门槛删除结果。按验证集选择配置后冻结测试。

本次再次只读检查 `BNPDFormer/_isaac_factory/PDFormer` 对应的实际嵌套路径及整个
`BNPDFormer` 目录：可见 `raw_data/all` 为 202、`n10_i1_all_usable` 为 134 等包，未找到
`raw_data/dense_i1` 或 `dense_i1_a1_prefix8` 权重。该入口实际指向 `work/isaac_factory`。
当前 208 包仅为探索/迁移控制结果；下一轮 graph_context 搜索尚未启动。
优先取得主实验 `episodes.npz`、`meta.json`、`node_map.json`、实际划分和最终运行配置，
通过同数据同 split 核验后再启动正式基线搜索。其他机器或路径是否存在该包尚未确认。

## 8. Upcoming 探索性训练（2026-09-11）

用户随后要求先改善当前 B4/B5 的 upcoming，因此允许在冻结的 208 包上开展有限的
validation-only 探索，不等待 204 包定位；这不解除第 7 节正式横向比较的执行门。
真实 start==0 / start>0 的子集定义同步用于训练损失：历史曾 hot、未来重新开始的正例
必须属于 upcoming，不能因历史状态被排除起点监督。该修正不改变任何标签或评分。

现有 `batch_factory_baseline_dense.sh` 增加 `diagnose`，对保存的权重读取 train 和
validation，不读取 test。`TRAIN_VARIANT` 可选 `history_control`、`graph_context`、
`upcoming_weighted`，分别是修正后的控制组、仅事件头上下文、仅提前正例权重 4→12。
Profile 使用 v2 区分旧训练损失；保持每模型原架构和预算，先诊断再决定必要的对照。
原 model 目录继续复用，归档旧文件后写入新结果，禁止新建目录或覆盖未归档权重。
执行记录见报告第 25.9、26 节。`d0f7368` 已部署；八次诊断确认当前 train/validation 的
重启冲突数均为 0，因此修正前后本包实际分组相同，跳过无差异的控制重跑。当前已启动
graph_context 两模型两 seed 的单变量对照；新结果必须由实际训练产生后补录。
