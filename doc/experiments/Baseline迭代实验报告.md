# Baseline 迭代实验报告

状态：进行中，尚非正式横向实验结论。当前版本仅整理验证证据；后续补齐实验、
选型、独立评估及对照一致性审计后才能冻结。未将目标标记为完成。

## 1. 目标与不变量

对象为 B2 XGBoost、B3 LSTM、B4 GCN-GRU、B5 BSTAN-style GAT-GRU。
允许模型有各自合理的训练参数，不要求复制 BNPDFormer 的专用结构。
所有修改限定 `dev_xwt`，服务器仓库为 `/home/sci/work/BSTAN_isaac_factory`。

比较必须保持相同的 raw cohort、episode split、标签定义、历史/预报窗、有效节点、
时间锚点及匹配规则。优化只看 validation；不得依照 test 数字继续选择参数。
主要指标为 report P/R/F1，同时报告 ongoing/upcoming recall、who P/R、
start/duration MAE、订单剩余时间 MAE、过程原因 macro recall 与 train-majority 背景。
格子 F1、事件 IoU F1、AP 为补充，不能替代主任务。

P>=0.80、R>=0.35 是业务可行性约束，不是保证每个 baseline 能达到的分数。
不能为了通过约束修改真实标签、起始时间容差或有效性 mask。

## 2. 当前可追溯证据

机器可读证据：[`baseline_validation_20260906_round1_partial.json`](baseline_validation_20260906_round1_partial.json)。
导出时间为 2026-09-06 03:18:41 HKT，仅包含当时已完成的 76 次 validation-only 训练。
此文件不表示所有排队任务已经完成；未读取 `metrics_test.json` 或综合 `metrics.json`。

已核对这 76 次训练：

- 全部使用相同 dataset manifest SHA-256：
  `f9602f3fd9e6b2107d81f0ed78e16780fa2f98cdd6939486e98cc6b9bb8f1f1a`。
- validation 均为 2572 个滑窗样本。
- contract 均为 `tyx_bn_agg_unsupervised_v2`，label 均为 `factory_ops_hot_v1`。
- B2 有 48 次、B4 有 18 次、B5 有 10 次完整记录；快照不含 B3 调参记录。
- 没有一次选中结果同时达到 P>=0.80、R>=0.35。
- 累加各运行记录的 elapsed_seconds 约 5.67 小时；存在并行运行及重复对照，
  这不是独占 GPU 耗时，也不能据此比较模型推理效率。

相同的 baseline manifest 证明这些 baseline 运行内部一致，**不能单独证明与
主实验使用了相同的 episode 分配**，见第 5 节。

## 3. 已完成候选的验证表现

以下均为 seed42/43 的均值，F1 后为 population standard deviation。
同一 validation 的相邻滑窗相关，seed 标准差不是 episode 泛化置信区间。

| 模型与候选 | report P | report R | report F1 | upcoming R | event-will AP | 结论范围 |
|---|---:|---:|---:|---:|---:|---|
| B2 `b2_event_search_v1/candidate_c5_event_w12` | 0.3070 | 0.2727 | 0.2887 ± 0.0125 | 0.0608 | 0.2335 | 该轮完整搜索选中 |
| B4 `b4_search_v1/candidate_c0_incumbent` | 0.4633 | 0.2138 | 0.2916 ± 0.0182 | 0.0026 | 0.2265 | 第一轮完整搜索选中 |
| B5 `b5_search_v1/candidate_c0_stabilized` | 0.5312 | 0.2155 | 0.3066 ± 0.0011 | 0.0106 | 0.2493 | 候选双 seed 完成，整轮未完成 |

不能只看表中最高 precision：例如 B4 新 Focal 候选 seed42 的 P=0.8333，
但 R=0.0337、F1=0.0647，只发出 12 条报告，不能视为解决了误报与漏报平衡。
这个单 seed 结果也不足以判定整个 Focal 方向无效。

## 4. 漏报诊断与当前实验

B4 incumbent seed42 的 validation 共 108 个 ongoing、189 个 upcoming、
32956 个负例站点。阈值 0.55 下，upcoming 中 188 个是概率漏报，0 个是
已经超过概率阈值但起始时间超出容差的漏报。

upcoming 的预测事件概率中位数约 0.0105。阈值降至 0.20，仍有 185 个概率漏报。
不加概率筛选时，约 78.3% 的 upcoming 起始时间预测落在容差内。
这些数字将主要问题定位在事件判别环节，并不证明某个具体模型改动必然有效。

已启动的受控实验为 `context × focal`：

| 候选 | 事件头加入已有历史的全图上下文 | Focal gamma |
|---|---|---:|
| control | 否 | 0（BCE） |
| context | 是 | 0 |
| focal | 否 | 2 |
| context_focal | 是 | 2 |

各候选均从头训练，两颗 seed；不改变 GCN/GAT/GRU 主体或 raw/derived 数据。
新实验存放在独立目录 `models/tuning/b4_event_ablation_v1` 与
`models/tuning/b5_event_ablation_v1`。服务器 tmux 队列名 `baseline_event_ablation`。
先 B4，等待原 B5 搜索结束后再执行新 B5 对照。

尚未完成的候选不参加正式跨候选排名。下一轮模型修改应等本轮完整结果再决定。

## 5. 对照一致性待审计项

### 主实验结果来源

`origin/dev_tyx@52e8643` 的《模型评估指标.md》第 6 节明确把
`rep_p=0.817 / rep_r=0.447 / rep_f1=0.578` 标为历史背景数字。
最终报告必须从当前 `FactoryBN_dense_f1_p80.json` 对应模型产物提取结果，
核对 cohort、split 与评估参数，不能把历史数字当作本次已验证的对手分数。

### 软链接与 split 身份

源码检查发现：主实验 `factory_bn/export_dataset.py` 在解析 raw 路径前保留
`p.name` 作为 episode 前缀；baseline 的 `_discover_groups` 先执行 `Path.resolve()`，
再取 `run_dir.name`。两边随后均按 run 名称排序并依次消耗同一个 RNG 进行 split。

若服务器的 `unsup_n10_i1/n10_*` 是指向 `machine20/human20/...` 的软链接，
别名与实际目录名差异**可能改变分组排序和抽样分配**。目前仅确认源码风险，
尚未用服务器实际主实验 manifest/NPZ 与 baseline split_manifest 逐 episode 验证。
不能把这一风险写成已确认数据泄漏，也不能仅凭 92/17/25 的数量一致认为已排除。

恢复远程访问后的优先动作：核对主实验 export `meta.json` 中的 run_names/run_dirs、
完整 episode 名称，以及双方 train/validation/test 的 episode 身份集合。
若不一致，先修复统一输入协议，在新目录重建派生训练数据并重新训练；不覆盖现有
运行，也不改写 raw 来伪装一致。是否需要重建，以实际审计结果为准。

## 6. 后续选型与停止依据

1. 先完成当前搜索、收齐完整配置与 validation 指标，再决定下一轮。
2. 完成 B3 的合理参数搜索；不能把尚未调参的单次 LSTM 结果写成方法极限。
3. 若某方向无改善，记录失败对照，不盲目增加同类权重或降低报警门槛。
4. 可继续考察一般性的工位身份编码、历史读出与类别不平衡策略，但必须用
   对照实验说明收益，不能借此把 baseline 改成主模型。
5. 停止一个调参方向，需要多颗 seed 上的改善趋势与漏报诊断共同支持。
   “基本达到限制”只能表述为当前数据、模型家族和披露搜索预算下的实测瓶颈，
   不是理论性能上限。
6. 配置冻结后再做独立评估；此前已被反复查看的旧 test 不能恢复成未使用的
   holdout。最终报告明确区分开发期比较和严格独立泛化证据。

最终报告还缺：本轮完整结果、B3 调参、必要的后续改进、主实验产物与 split 审计、
冻结配置及完整主表。当前不具备结束实验并宣称指标达标或方法到顶的证据。
