# Baseline 迭代实验报告

工作状态（2026-09-12，按用户要求收尾交接）：v6 数据重建、全样本对齐、B4/B5
控制组、上下文、提前类别加权和三分类候选均已完成。最后一轮权重的 8 份诊断也已
完成并核验；服务器 `baseline_dense_v6` 与 `baseline_dense_diag` 均 dead=1、exit=0，
未发现对应 baseline 训练或诊断进程。用户要求停止本对话的后续优化，交由新对话接续；
没有新建实验目录、没有删除旧目录、没有启动新的历史特征实验。

当前目标尚未达成：总体 report F1 约 0.75，但 upcoming recall 仍低。
最后权重显示训练集提前类别排序改善而验证集未同步改善，不能简单靠增加 epoch。
新发现主模型源码额外使用编码窗口之前的历史摘要；当前输入信息范围并未完全对齐。
相关结果与证据见第 27.6--27.10 节；新对话先读同目录
`Baseline交接与远端执行指南.md`，再按需查看本长报告。

第 24--27 节是新 dense 开发结果，前 23 节是旧 v5 结果，不能混排。第 25 节主模型
数字来自文档，不是共同数据包重评。正式冻结仍需独立评估与对照一致性审计，
本次收尾不表示优化成功，也不表示已经证明模型能力上限。

上一轮进度（2026-09-07）：第 21 节训练抽样对照 12/12 完成，B4/B5 均选回均匀控制组，
没有综合提升；四个控制组精确复现原 history 的 station_report，保存权重的重评与前缀
历史敏感性复核亦完成。当前 v5 累计 60 次已完成运行，加旧 v3 的 112 次，共 172 次
开发运行/阶段记录，包括重复控制组和续训阶段，不是 172 个独立配置或独立统计样本。
按用户本次要求先整理结果和后续计划，目前没有新训练在运行，也没有启动下一轮调参。

最新前置变化：随后 fetch 到 dev_tyx 的 `20c40e2`，主实验文档已更换数据包和事件定义。
这不是同一任务上的新增高分，不能与本报告 v5 直接排名。具体源码核对、重评参数遗漏
见第 19 节；当前已完成的本地标签/评分迁移及仍待执行部分见第 24 节。

## 0. 当前结果速览

阅读顺序：第 24 节看本次迁移；本节看旧 v5 结果，第 21 节看旧完整对照，第 22 节看旧计划，第 23 节看
“模型限制还是训练限制”的收尾判断。第 1--20 节
保留按时间记录的设计、排查和实验过程；其中“尚在运行”等表述是对应当时的状态，
不覆盖本节最新进度。旧版本/旧 test 分数不与下表混排。

### 统一数据与报告口径

- 数据目录 `factory_pdformer_134_v3`，张量 schema 为 `factory_baseline_dataset_v5`；目录
  名的 v3 不代表仍使用旧张量版本。134 个 episode，train/validation/test 为 92/17/25。
- 契约为 `tyx_bn_agg_unsupervised_v2`，标签为 `factory_ops_hot_v1`，仍是离线规则监督；
  同口径预测分数不等于已经证明扰动因果或真实生产瓶颈的强标签有效性。
- 总计 20060 个窗口，split 为 13813/2589/3658；38 节点、27 特征，30 个历史窗口，
  15 个未来窗口，每窗口 60 秒。保持已审计的共同输入、标签、mask、split 和评分规则。
- 下表全部是 **validation、seed42/43 均值**，F1 的 ± 为两颗 seed 的总体标准差；
  不是 test，不是独立 episode 的置信区间，更不是挑一颗 seed 的最大值。
- report P/R/F1 衡量按当前站点、事件和起始时间规则匹配的报警；upcoming R 单独看
  未来才开始的事件。validation 有 297 个站点窗口事件目标，其中 ongoing 108、
  upcoming 189；滑窗之间相关，不能将这些目标称为 297 个独立物理事件。

### 当前保留方案

| 模型 | 保留方案 | report P | report R | report F1 | upcoming R | will AP |
|---|---|---:|---:|---:|---:|---:|
| B2 XGBoost | c5_event_w12 | 0.2926 | 0.2660 | 0.2785 ± 0.0031 | 0.0635 | 0.2298 |
| B3 LSTM | c0_incumbent | 0.2172 | 0.0724 | 0.1085 ± 0.0088 | 0.0106 | 0.0760 |
| B4 GCN-GRU | history + warm_base | 0.4915 | 0.2172 | 0.3006 ± 0.0025 | 0.0079 | 0.2194 |
| B5 GAT-GRU | history | 0.4876 | 0.2323 | 0.3142 ± 0.0051 | 0.0159 | 0.2500 |

出处：`baseline_validation_v5_20260906_short_hot_complete.json` 中的两次 B2/B3
confirmation、B4 staged/warm_base 和 B5 representation/history。上述方案是当前开发
结果保留点，不代表已冻结正式论文配置。当前四个模型都没有同时满足 P>=0.80、R>=0.35。

B4 的纯 history 为 P/R/F1=0.4835/0.2155/0.2974；续训平均只增加约 0.0033 F1，
只改善 seed43，seed42 保留父权重。不能把这一小差异写成可靠的大幅提升。B5 的续训
选回阶段 0，保留原 history 即可，无需增加一次没有收益的续训步骤。

### Precision 0.7 多的方案是什么

它是 **B5 短热负例权重 4**，不是上表的综合保留方案：

| 统计方式 | P | R | F1 |
|---|---:|---:|---:|
| seed42 | 0.8043 | 0.1246 | 0.2157 |
| seed43 | 0.6719 | 0.1448 | 0.2382 |
| 两颗 seed 均值 | 0.7381 | 0.1347 | 0.2270 |

该方案通过减少报警提高 precision，但漏报增加，F1 和 upcoming recall 均不如 history；
因此保留作“偏高精度工作点”的证据，不替代综合方案，也没有通过双门。

### 已经验证了什么

| 调整方向 | 完整证据 | 判断 |
|---|---|---|
| B2 显式事件头、权重搜索 | 旧数据完整搜索及 v5 两颗 seed 确认 | 保留事件头方案，尚不擅长提前事件 |
| B3 规模、正则化、学习率和权重搜索 | 旧数据 16/16 及 v5 两颗 seed 确认 | 当前 flatten-LSTM 仍弱，不能称所有 LSTM 的上限 |
| B4/B5 历史表示、节点身份 | v5 16/16 | 保留较简单的 history，不扩大无独立收益的 ID 分支 |
| 从头延长、低学习率续训、加大 upcoming 权重 | v5 16/16 | B4 续训仅小幅、单 seed 改善；B5 无净收益 |
| 短热负例加权 | v5 12/12 | 高 precision 以大量召回损失为代价，控制组胜出 |
| 训练事件/提前事件窗口抽样 | v5 12/12 | 两模型均控制组胜出，不继续扩大同方向搜索 |

这些结果支持“已测试训练因素出现平台期”，**不支持已经证明 GCN/GAT 的理论上限**。
训练集 upcoming AUC 已有排序信号，但高置信报警质量不足；验证集对短热负例的区分
能力接近随机，且存在泛化退化。当前主要问题是事件是否会持续、提前事件置信度和
泛化，不能只靠调高阈值或放宽起点容差解决。

### 与主实验的比较边界

历史主实验 P/R/F1≈0.8504/0.3636/0.5094 是一个续训阶段保存 checkpoint 的验证记录，
不是多 seed 均值；当前源码不能严格加载该权重，尚未完成同口径重评。
2026-09-06 核对的 dev_tyx `20c40e2` 文档中，dense_i1 高分又使用不同数据和事件
定义，不能放进上表直接排名。
共同历史 hot 字段存在使用完整 episode 的问题，已有前缀敏感性实测，但尚未共同修正。
因此本报告是可复核的 baseline 开发证据，不冒充已经完成的正式主实验对照表。

## 1. 目标与不变量

对象为 B2 XGBoost、B3 LSTM、B4 GCN-GRU、B5 BSTAN-style GAT-GRU。
允许模型有各自合理的训练参数，不要求复制 BNPDFormer 的专用结构。
所有修改限定 `dev_xwt`，服务器仓库为 `/home/sci/work/BSTAN_isaac_factory`。

目录约定（2026-09-06 用户确认）：后续不按 commit 新建源码副本或构建目录，复用
现有仓库及执行目录。实验结果放在现有 benchmark 的 `models` 下，按实验轮次、候选
和 seed 区分；源码 commit、配置、权重与指标记录在产物和本报告中。已建立的目录
保持原状，由用户决定是否清理，不自动删除或覆盖历史结果。

比较必须保持相同的 raw cohort、episode split、标签定义、历史/预报窗、有效节点、
时间锚点及匹配规则。优化只看 validation；不得依照 test 数字继续选择参数。
主要指标为 report P/R/F1，同时报告 ongoing/upcoming recall、who P/R、
start/duration MAE、订单剩余时间 MAE、过程原因 macro recall 与 train-majority 背景。
格子 F1、事件 IoU F1、AP 为补充，不能替代主任务。

P>=0.80、R>=0.35 是业务可行性约束，不是保证每个 baseline 能达到的分数。
不能为了通过约束修改真实标签、起始时间容差或有效性 mask。

## 2. 当前可追溯证据

最新补充：`baseline_validation_20260906_round2.json` 已收齐原 B5 搜索的 16 次、
B4 context/focal 对照的 8 次训练；总快照包含 88 次已完成 validation-only 运行。
原 B5 最佳仍是 `candidate_c0_stabilized`。B4 新轮按既定稳健排序选中 context，
其 F1=0.2848，低于 control 的均值 0.2916，但跨 seed 波动较小；不能写成平均效果提升。
Focal 单独和与 context 组合的 F1 均值分别为 0.0584、0.2095，upcoming recall 均为 0。

`baseline_main_episode_split_20260906.json` 实测双方 134 个 episode 身份和分配均一致，
train/validation/test 交集分别为 92/17/25；本批数据未触发第 5 节的软链接 split 风险。
`baseline_train_symmetry_20260906.json` 在 13723 个训练样本中只发现 62 个存在所检查的
精确工位交换对称，且对应工位的事件标签无冲突。不支持用所检查的工位交换对称解释
差距；该旧审计没有检查跨窗口的完整输入相同而未来目标冲突。

2026-09-06 11:28 HKT，B5 `b5_event_ablation_v1` 已以 exit=0 完成 8/8；
B3 `b3_search_v1_pyfix` 尚在运行，不对其未完成候选排名。
B3 首次启动在导入 PyTorch 阶段失败，未训练；脚本改为显式使用 `PYTHON_BIN` 后
已成功训练，失败目录 `b3_search_v1` 保留，不计入模型候选分数。

后续输入审计发现一个必须修复的偏移：`validation_jobs_anchor_diagnostic_20260906.json`
记录 153/2572 个 validation 样本的 `jobs_remaining` 不一致。baseline 构建器取
`jobs_remaining[position]`，对应第一未来窗口；主实验取 `position-1`，对应最后历史窗口。
该问题不否定 episode split 审计，但说明 episode 一致不足以证明输入完全对齐。
需要修正输入索引并重建派生张量，不需要重新采集 raw。现有结果只保留为开发期证据。
B3/B5 当前完整轮继续保留；尚未启动的 B4 表示对照等待队列已撤下，避免再启动旧输入实验。
`tools/audit_baseline_validation_contract.py` 用于继续核对 validation 的观测 X、时间锚点、
mask、hot 和事件目标，不读取 test 指标；正式冻结前必须以修正数据通过实际审计。

`baseline_validation_contract_20260906.json` 实测共同的 2572 个 validation 样本：
hot、事件发生/开始/时长、事件节点 mask、历史末尾 hot 及时间锚点一致。
但每个 validation episode 少一个末尾预测样本（共 17 个），所有样本的剩余窗口数少 1，
238 个样本的剩余时间 mask 不同；另有 11 个样本、132 个历史特征格不一致。
不能把这些差异归为模型能力，也不能凭事件目标暂时一致声称输入协议完全对齐。

原因已定位：构建器调用 `bn_agg` 时使用 `closed_windows_only=True`，而主实验 bundle
来自离线 `False` 路径。前者删除末尾不满一分钟的窗口，也把未闭合扰动区间截到 episode
结束；后者保留终末窗口但不采用这些未闭合区间。实测 `material重采__episode_05`
在窗口 123 的两个 workbench 的 `disturbance_active_s` 为 baseline=1、main=0。
在独立探针目录用当前离线代码重建该 episode，136×38×27 个特征与 main bundle 全部一致。
该探针仅证明这个 episode；新数据必须再通过整套 validation 审计。

数据集 v4 修正：订单数使用最后历史窗口起点；离线派生保留终末部分窗口供预测目标使用，
历史输入仍仅允许完整窗口。新派生 CSV 和张量统一写入独立 benchmark 目录，不覆盖 raw
或旧实验。只有当前版本加载路径，不在主代码中兼容旧 v3；旧结果用原 commit 复现。
这是对已保存主实验数据口径的对齐，不意味着主实验对未闭合扰动的忽略在业务上最合理。
此项需与 tyx 一起讨论，若改进应共同重建与重训，不能只给 baseline 换一套特征语义。
当前 B3 整轮结束前，不更新其所在服务器工作目录的 v4 加载器。
v4 修正提交为 `daac38b`（dev_xwt）。通过该提交的独立只读构建快照运行重建，
服务器训练工作目录仍固定在 dev_xwt 的 `35ee064`，未切换分支，也未修改 dev_tyx。
新 benchmark 为 `factory_pdformer_134_v2`，新派生已完成 134 个 episode。第一次构建
启动缺少 PyTorch 即退出，未生成数据，
已保留日志并显式传入经验证的 PYTHONPATH 后重新启动；不是因等待超时重复运行。

张量阶段原实现对重叠历史重复解析与逐格赋值，服务器已观测到长时间单核计算。
优化为每个 episode 转换一次不可变张量，再切片构建历史；没有改变样本、标签或归一化规则。
本地用 1020 个含缺失 buffer 观测的样本，与 Git `daac38b` 实现逐 tensor、sample row、
split 比较完全相同，耗时从 5.960s 降至 0.094s（本地约 63 倍，不是服务器耗时承诺）。
构建入口拒绝覆盖已有张量/manifest，已完成的派生 CSV 可显式通过 `--derived_root` 复用。

确认旧转换 PID=3697518 仍在单核运行、没有任何张量输出后，仅终止这个未完成转换。
复用全部已完成 CSV，以 dev_xwt `7c50640` 的只读快照完成快速张量构建；未重跑 raw 聚合，
没有中断 B3。快照仅用于运行，所有代码修改/提交仍在 dev_xwt，服务器训练工作目录未切换分支。

### v4 实测审计与剩余 A.3 问题

新张量已完成：20060 个样本，train/validation/test=13813/2589/3658，episode=92/17/25。
134 个 episode 的身份及 split 匹配。2589 个 validation 样本的历史特征、评分、hot、
事件发生/开始/时长、订单数、剩余长度、各 mask 及时间/窗口索引全部匹配，末尾样本差异为零。
但 A.3 `y_cause` 有 128 个样本不匹配，所以整体 `comparison_match=False`，还不能正式训练。
服务器产物为 `baseline_episode_split_v4_20260906.json` 和
`baseline_validation_contract_v4_20260906.json`（现已随 `64366a6` 同步入 Git）。

已查到具体原因：主实验原有 `n10_human1.0/derived/episode_06/env_00` 特征 CSV
没有 `labor_saturated_s`，新离线聚合新增了这一列。窗口 32 的共同 CSV 字段逐项一致。
主实验 NPZ 已通过 `ensure_labor_saturated_feature` 补算第 27 维，因此模型输入和 A.1
可一致；但保存的 A.3 原因标签没有同步采用新增列。比如窗口 32/37/41，当前聚合为
`starved_upstream`，主 bundle 仍为 -1；窗口 97 则分别为 `starved_upstream` 和
`transport_delay`。两边原因类别字典相同，不是编码顺序错位。

原派生 min_event_windows=1、新派生=8 也不同，但源码中 A.3 当前窗口原因的判定不依赖
这个事件段长度过滤，不能简单改成 1 就宣称修好。下一步应使用主实验实际冻结 bundle
作为共同标签来源，并验证字段/身份/哈希；不添加缺列回退到旧规则的兼容分支。
若双方要修正劳动饱和的 A.3 语义，则必须共同更新 bundle 并重训主模型和 baseline，
不能把新 A.3 baseline 与旧 A.3 checkpoint 当作完全同口径。当前 v4 原型保留为失败审计证据。

### B5 context/focal 完整轮

本轮验证指标（seed42/43，仍是旧 v3 数据的开发期对照）：

| 候选 | report P | report R | report F1 均值 ± std | upcoming R | will AP |
|---|---:|---:|---:|---:|---:|
| context | 0.4959 | 0.2071 | 0.2921 ± 0.0064 | 0.0079 | 0.2321 |
| control | 0.5290 | 0.2020 | 0.2920 ± 0.0134 | 0.0053 | 0.2415 |
| focal | 0.6235 | 0.1044 | 0.1779 ± 0.0227 | 0 | 0.2218 |
| context_focal | 0.5299 | 0.1212 | 0.1838 ± 0.0734 | 0 | 0.1815 |

按既定稳健排序选中 context，但均值增加仅约 0.0001，不支持声称有实质改善；
也未超过第一轮 `c0_stabilized` 的 0.3066。两个 Focal 方向均不再追加同类搜索。
四个候选全部未通过 P/R 双门，事件提前预测仍未解决。
服务器已导出 `doc/experiments/baseline_validation_20260906_round3_partial.json`，
含 100 次已完成 validation-only 运行（其中 B3 仅 4 次，尚非完整搜索）；现已随 `64366a6` 同步入 Git。

首次快照：[`baseline_validation_20260906_round1_partial.json`](baseline_validation_20260906_round1_partial.json)。
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
| B5 `b5_search_v1/candidate_c0_stabilized` | 0.5312 | 0.2155 | 0.3066 ± 0.0011 | 0.0106 | 0.2493 | 完整搜索选中 |

不能只看表中最高 precision：例如 B4 新 Focal 候选 seed42 的 P=0.8333，
但 R=0.0337、F1=0.0647，只发出 12 条报告，不能视为解决了误报与漏报平衡。
该单 seed 本身不足以下结论；现已补齐的 B4 双 seed 对照见第 2 节。

## 4. 漏报诊断与当前实验

B4 incumbent seed42 的 validation 共 108 个 ongoing、189 个 upcoming、
32956 个负例站点。阈值 0.55 下，upcoming 中 188 个是概率漏报，0 个是
已经超过概率阈值但起始时间超出容差的漏报。

upcoming 的预测事件概率中位数约 0.0105。阈值降至 0.20，仍有 185 个概率漏报。
不加概率筛选时，约 78.3% 的 upcoming 起始时间预测落在容差内。
这些数字将主要问题定位在事件判别环节，并不证明某个具体模型改动必然有效。

受控实验为 `context × focal`，B4/B5 均已完整结束：

| 候选 | 事件头加入已有历史的全图上下文 | Focal gamma |
|---|---|---:|
| control | 否 | 0（BCE） |
| context | 是 | 0 |
| focal | 否 | 2 |
| context_focal | 是 | 2 |

各候选均从头训练，两颗 seed；不改变 GCN/GAT/GRU 主体或 raw/derived 数据。
新实验存放在独立目录 `models/tuning/b4_event_ablation_v1` 与
`models/tuning/b5_event_ablation_v1`。旧队列结束后 B5 未启动，已确认目录不存在且
无相应训练进程，再以独立 tmux `baseline_b5_event` 启动；未重跑已完成候选。

尚未完成的候选不参加正式跨候选排名。下一轮模型修改应等本轮完整结果再决定。

## 5. 对照一致性待审计项

### 主实验结果来源

已取得一个主实验保存产物：`n10_i1_all_usable_unsup_p80decode/BNPDFormer_best.pt`，
见 `main_validation_reference_20260906.json`。它是 seed42 单次训练、该阶段 epoch6 的
checkpoint（由 `unsup10` 权重继续训练），不是多 seed 均值或已证实的全局最好结果。
保存的 validation P/R/F1 为 0.8504/0.3636/0.5094，upcoming recall=0.2698。
该产物记录固定报告阈值 0.65、recall gate=0，与当前正式配置的阈值扫描及双门选型不同；
所以只能作为有来源的参考，不直接作为最终多 seed 主表。当前服务器主实验源码为
`f322dbf`，远端最新文档为 `52e8643`；当前源码不等于训练时源码的证明。

后续服务器只读核对：该 p80decode 目录只有 `BNPDFormer_best.pt` 和 `last_metrics.json`。
配置指向的初始 `unsup10` 目录目前不存在；在 PDFormer/output 的已有四组 train.log 中
未找到 p80decode 或 init_ckpt/unsup10 的对应记录，尚不能恢复这次训练的完整曲线和总预算。
当前源码 `_load_init_ckpt` 只加载形状匹配的模型权重，随后创建 AdamW，不是恢复优化器的
完整断点续跑。这说明当前实现的热启动机制，但仍不能单凭当前源码证明历史每一步操作。

`origin/dev_tyx@52e8643` 的《模型评估指标.md》第 6 节明确把
`rep_p=0.817 / rep_r=0.447 / rep_f1=0.578` 标为历史背景数字。
最终报告必须从当前 `FactoryBN_dense_f1_p80.json` 对应模型产物提取结果，
核对 cohort、split 与评估参数，不能把历史数字当作本次已验证的对手分数。

### 软链接与 split 身份

源码检查发现：主实验 `factory_bn/export_dataset.py` 在解析 raw 路径前保留
`p.name` 作为 episode 前缀；baseline 的 `_discover_groups` 先执行 `Path.resolve()`，
再取 `run_dir.name`。两边随后均按 run 名称排序并依次消耗同一个 RNG 进行 split。

若服务器的 `unsup_n10_i1/n10_*` 是指向 `machine20/human20/...` 的软链接，
别名与实际目录名差异**可能改变分组排序和抽样分配**。最初仅发现源码风险，
现已用服务器实际主实验 manifest/NPZ 与 baseline split_manifest 逐 episode 验证通过。
证据是逐 episode 差异为空，而不是仅凭 92/17/25 的数量一致。

本次无需因此重建。后续数据迁移仍应重复审计，不能仅凭路径别名认定一致；
若发现不一致，应在新目录修复统一输入协议，不改写 raw 或覆盖现有实验。

只读审计工具为 `tools/audit_baseline_episode_split.py`：传入 baseline 数据集目录、
主实验 bundle、可信训练 checkpoint，以及主实验所用 `PDFormer` 源码目录。
工具直接调用该目录的 split 函数，用 checkpoint 中的 seed/比例重建分配，
通过 raw `episode_config.csv` 将软链接别名映射到同一 episode 身份。
输出逐 episode 差异和输入 SHA-256，不读取预测指标或 NPZ 中的未来标签。
其 `episode_split_match` 仅证明提供的这些产物在 episode 分配上是否一致；
若主实验 bundle/源码在训练后变动，仍需历史快照才能证明当次实际分配。
这也不能替代 raw 字节、特征、标签和样本时间锚点的一致性审计。

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

最终报告还缺：B2/B3 在当前 v5 数据上的确认重训、B4/B5 后续改进、主实验正式多 seed
结果、冻结配置及完整主表。共同 validation 输入/目标审计已完成，见第 8 节；它不替代
独立泛化评估。当前不具备结束实验并宣称指标达标或方法到顶的证据。

## 7. B4/B5 下一轮表示对照

B4 context/focal 整轮完成后，新增 `b4_representation_v1`，不改变数据、损失权重或评分。
保持两层 GCN、node-wise GRU 和同一事件头，比较以下四个从头训练的候选，各 seed42/43：

| 候选 | 已知工位 ID 嵌入 | GRU 读出 |
|---|---|---|
| control | 无 | 末状态 |
| identity | 16 维，投影后加入图卷积残差 | 末状态 |
| history | 无 | 末状态与全历史均值拼接，线性投影回原维度 |
| identity_history | 16 维 | 末状态与全历史均值 |

假设分别是固定工位的不同动态可受益于显式身份，以及均值读出能减轻末状态对近期状态的偏重。
这不是已证实的修复或性能上限解释，必须等待完整对照；不使用未来信息，不增加主模型专属解码。
工位身份绑定同一 manifest 的 node_ids，只支持当前已知工位，不据此声称跨布局泛化。
预算保持 max_epochs=60、min_epochs=10、patience=10、batch=24、lr=3e-4、weight_decay=0.01。
入口已统一为 `bash batch_factory_baseline_representation.sh B4` 或 `B5`，旧脚本不留别名。
两模型都比较 control、identity、history、identity_history，每组 seed42/43，从头训练。
B5 保留两层 GAT + GRU，采用相同的身份/历史读出对照；batch=16、lr=1.5e-4、
min_epochs=15、patience=20，其他参数在该模型四个候选间不变。B4 保持上述原定预算。
新数据 `factory_pdformer_134_v3` 上先跑 control，避免把输入修复混成结构收益。
脚本校验通过的 episode/validation 审计和 manifest 哈希，拒绝覆盖旧搜索目录。
默认关闭 B5 新选项时，与 Git `805efeb`
相同 seed 的权重及全部输出 tensor 逐位一致。完整实验结束前不宣称选项改善指标。
本轮本地模型/掩码/梯度/checkpoint/命令路由共 33 项测试与 9 个 subtest 通过；
脚本路由测试验证每模型 8 次候选、一次完整排名、全部 validation-only，并拒绝未通过或
哈希失效的输入审计和已有输出目录。测试中的模拟训练仅用于命令路由，不计入实验成绩。
实验开关用于正在执行的消融，不是旧版本兼容路径。最终冻结时应清理无收益的候选实现，
历史实验的完整代码继续由 Git commit 保留。

## 8. 共同 A.3 标签绑定与 v5 验证

v5 构建 API 必须显式传入主实验冻结 bundle；A.3 只从该 bundle 的 cause[t-1] 读取，
不再读取 baseline 重算的 bottleneck_label.csv。两边原因字典、物理 episode 身份、
cohort、窗口索引和窗口起点须严格匹配；NPZ/meta 文件哈希与 episode 名称记入 manifest。
原始 raw 和主实验 bundle 均不修改，旧 v4 失败审计保留，v5 输出另存
`factory_pdformer_134_v3`。这是一条明确共同标签路径，不是根据缺字段切换旧规则的兜底。

窗口特征与 A.1 仍按已对齐离线规则构建，完整验证仍逐项比较 X、A.1、A.3、mask、订单数、
时间锚点和 episode split；不通过删除失败检查来获得通过。当前本地测试覆盖冻结标签值、
类别顺序、cohort、无标签 -1、非法值、时间错位及缺文件；相关 33 项测试与 5 个 subtest 通过。
服务器实际审计结果见本节下文。

2026-09-06 12:16 HKT，B3 为 8/16 完成，活动训练 PID=3731066；原工作目录不更新
v5 加载器，避免打断旧数据的完整轮。下一步先完成 v5 服务器构建与审计，再跑 B4/B5
新数据上的 incumbent 和表示对照，不把当前约 0.3 的开发分数视为方法上限。

v5 提交 `26893a1` 已在服务器独立只读快照成功构建 20060 个样本，split 数量为
13813/2589/3658。随后 episode 审计因共享模块提前导入 baseline 自带 factory_bn，
触发“参考 splitter 非指定 checkout”的来源保护而退出；并非数据构建失败。
修正为先加载指定主实验 splitter，再导入身份辅助函数，保留来源检查。
测试增加独立主实验代码目录，避免只在同目录通过而漏掉服务器场景。
已有张量不覆盖、不重建，仅重新执行两项审计。

2026-09-06 12:33:54 HKT，`baseline_v5_audit` 以 exit=0 完成。由 `b06728b` 独立
只读快照重跑来源正确的审计后，134 个 episode 的 split 一致；2589 个 validation 样本
位置差异为 0，以下字段均为 0 差异：y_score、y_hot、y_cause、remain_mask、target_remain_len、
occ_node_mask、hist_last_hot、event_will/start/duration、jobs_remaining/total、raw_x_observed、
anchor_time_s、first_future_start_s、input_window_indices。`comparison_match=True`。
这是完整 validation 观测输入/目标审计，并非 test 性能评估或主模型历史训练溯源证明。

证据存于新 benchmark 的 `episode_split_audit.json`、`validation_contract_audit.json`，
并复制为服务器 `doc/experiments/baseline_episode_split_v5_20260906.json` 与
`baseline_validation_contract_v5_20260906.json`，待 B3 结束、服务器工作目录可快进后同步。
上述五份服务器证据已改由独立 dev_xwt 执行 checkout 在开训前提交为 `64366a6`，
并拉回本地；不需要中断 B3 或更新其旧工作目录。
构建控制台日志移入新 benchmark，未遗留在仓库根目录。v5 数据现在可作为下一轮 B4/B5
共同协议的起点；新数据上的模型对照与多 seed 完整结果仍未完成，不宣称方法达标。

### 续训对照原则

B4/B5 可以从自身 checkpoint 进行低学习率微调，不因此失去 baseline 身份。
它是额外待验证的训练方案，不保证提高 upcoming。须与相同总训练预算的从头训练对照，
固定 train/split/标签，仅 validation 选初始 checkpoint 与参数，逐 seed 独立完成两个阶段。
产物应记录父 checkpoint 哈希、数据版本、每阶段 epoch/更新步数/耗时/学习率及选型规则；
均值覆盖完整训练流程，不能只报续训阶段的最好 seed 或 epoch。当前先完成共同数据审计，
不从旧 v3 不同输入协议的权重热启动并把它冒充新协议从头训练的公平对照。

## 9. 新数据模型整轮已启动

表示对照实现提交 `161327c`，加上服务器证据后的实际执行提交固定为
`64366a62b075f7abfd3b9861176e2e866ec8443d`。
独立 checkout `/home/sci/work/BSTAN_baseline_dev_xwt_v5` 的分支是 dev_xwt，
原 `/home/sci/work/BSTAN_isaac_factory` 仍在 dev_xwt 的旧提交运行 B3，未改代码或切分支。
此执行 checkout 在整轮完成前不 pull、不提交修改；新文档提交不改变运行中的代码。

tmux `baseline_graph_v5` 的 pane PID=3760105，顺序执行 B4 整轮，再执行 B5 整轮；
每模型 control/identity/history/identity_history x seed42/43。已实测 B4 control seed42
训练 PID=3761319 完成 epoch1/2，确实开始训练，而非只创建队列或跑 smoke。
此时 B3 已完成 11/16，活动 PID=3759057。主机可用内存约 24 GiB，GPU 占用约
8.5/32 GiB；没有为此停止其他训练或 Isaac Sim 进程。存在并发，不将耗时当作独占推理效率。

输出继续写入原仓库的 `factory_pdformer_134_v3/models/tuning/b4_representation_v1`
和 `b5_representation_v1`，不是写到主实验或执行副本的数据目录。
整轮日志为新 benchmark 内 `representation_round_v1_console.log`。
绑定 dataset manifest SHA-256 为
`66c6554d0ae1c7a293e482a201829a4c324f1d9d0b88934c1c2a6c1e27333183`。
先等各完整轮结束再比较均值、稳健分数及 upcoming 诊断；目前不对早期 epoch 排名，
也不声称新表示已提高 F1。后续仍需完成 B3 整轮、必要重训、阶段微调对照及最终报告。

## 10. B4 新数据表示对照完整结果

2026-09-06 13:58 HKT 已确认 `b4_representation_v1` 为 8/8 完成，
`selection.json` 状态为 `validation_selection_completed`、`test_evaluated=False`。
同一 tmux 随后开始 B5 表示对照；下表不是 B5 的结果，也不是旧 v3 数据的跨版本收益。
全部使用第 9 节固定代码与 v5 manifest，各候选 seed42/43：

| 候选 | report P | report R | report F1 均值 ± std | hot F1 | upcoming R | will AP |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.4176 | 0.2323 | 0.2978 ± 0.0063 | 0.3900 | 0.0159 | 0.2257 |
| history | 0.4835 | 0.2155 | 0.2974 ± 0.0007 | 0.3952 | 0.0026 | 0.2230 |
| identity | 0.4395 | 0.2020 | 0.2768 ± 0.0089 | 0.3891 | 0.0079 | 0.2328 |
| identity_history | 0.4448 | 0.2037 | 0.2780 ± 0.0027 | 0.3906 | 0.0106 | 0.2145 |

按预先固定的 mean-minus-std 排序选中 history：稳健分数 0.2966，高于 control 的
0.2915。但平均 F1 下降约 0.0004，不能称为总体性能提升。precision 增加约 0.0659，
伴随 recall 下降约 0.0168、upcoming recall 下降约 0.0133。hot F1 仅增加约 0.0052。
两个 seed 的标准差不代表跨 episode 的置信区间。全部候选未通过 P/R 双门。
本轮不支持继续扩大节点 ID 嵌入搜索，也不能据此宣称 B4 已达到方法上限。

### 新数据漏报诊断

使用既有 `diagnose_baseline_events.py` 对 control/history 的四个 best.pt 在 CPU 上
完成 validation-only 推理，未改权重或训练作业。每次覆盖 2589 个 validation 样本，
含 189 个 upcoming 站点目标；这些滑窗目标相关，不是 189 个独立物理事件。

| 候选 / seed | best epoch | upcoming 概率中位数 | 起始时间在容差内 | 概率漏报 @0.55 | 时间漏报 @0.55 | 概率漏报 @0.20 |
|---|---:|---:|---:|---:|---:|---:|
| control / 42 | 13 | 0.03269 | 0.8201 | 183 | 2 | 171 |
| control / 43 | 18 | 0.01870 | 0.8519 | 187 | 0 | 174 |
| history / 42 | 10 | 0.00577 | 0.7778 | 189 | 0 | 185 |
| history / 43 | 12 | 0.01956 | 0.7619 | 188 | 0 | 179 |

时间在容差内比例是不经过报警概率筛选、使用既定起始时间解码的诊断值；不是正式
提前召回率。时间漏报只统计已经通过概率阈值但起始时间不匹配者。阈值扫描仅用于
诊断，不据此更换正式选型规则。即使降至 0.20，仍有 171–185/189 个概率漏报，
支持优先检查事件判别的训练信号、泛化与阶段训练，而非单独降低报警阈值。

原始诊断位于新 benchmark 的
`models/tuning/b4_representation_v1/diagnostics/{control,history}_seed{42,43}.json`。
已完成运行的配置、validation 指标及哈希另由导出工具归档；B5 整轮结束后再作其完整
比较，并决定后续受控调整。当前 B3 仍为 15/16、进程存活，不对最后未完成候选排名。

### 训练集事件信号分布

通过正式 `FactoryBaselineTensorDataset` 生成 train 的事件目标后统计，避免把动态生成的
event_will 误当成 dataset.pt 中的静态字段。覆盖 13813 个训练样本，不使用 validation/test
决定类别比例；计数单位为滑窗站点目标，不能视作独立样本数：

| 资源类型 | negative | ongoing | upcoming |
|---|---:|---:|---:|
| machine | 63247 | 357 | 491 |
| workbench | 24315 | 198 | 177 |
| gantry | 43807 | 151 | 312 |
| agv | 44324 | 39 | 66 |
| 总计 | 175693 | 745 | 1046 |

训练中 future_start>0 且 hist_last_hot=1 的正例为 0，因此这批数据上没有证据支持
“loss 将大量 upcoming 错当成 ongoing”这一解释。原搜索的 base、balanced、strict
分别使用 upcoming/negative 权重 4/2、6/3、7/4，比值为 2、2、1.75；后两组并未增加
upcoming 相对负例的权重。总 loss 系数和 ongoing 权重也有变化，所以不能仅凭比值
断言梯度完全相同，或断言提高 upcoming 权重必然改善泛化。

此证据支持下一轮隔离“提前事件信号强度”与“增加训练阶段”的影响，而不是复用旧搜索
标签称已经充分搜索正负平衡。先等 B5 表示对照完整结束，再固定有限候选与预算；不在
运行中改损失、阈值或样本。原始统计产物为新 benchmark 的
`train_event_balance_diagnostic_20260906.json`，包含数据 manifest 哈希。

以上 B4 完整配置/指标、selection、四份漏报诊断及训练分布，以及下节 B3 完整轮，
已由服务器生成并随 `b81503b` 归档到本地 `doc/experiments`。未提交 raw、模型权重
或 test 预测；执行中的 B5 checkout 仍保持 `64366a6`，未跟随报告提交更新。

## 11. B3 旧数据搜索完整结果

2026-09-06 14:23 HKT，`b3_search_v1_pyfix` 为 16/16，selection 状态为
`validation_selection_completed`，确认没有仍运行的 B3 Python 训练进程。
以下仍为旧 v3 数据、2572 个 validation 样本的开发期结果，不与当前 v5 的 B4
表格合并成同口径正式排名。原始配置、逐 seed 指标及哈希存于
`baseline_validation_v3_20260906_b3_complete.json`，完整排序见
`baseline_b3_selection_20260906.json`：

| 候选 | report P | report R | report F1 均值 ± std | upcoming R | will AP |
|---|---:|---:|---:|---:|---:|
| c0_incumbent | 0.2462 | 0.0892 | 0.1310 ± 0.0087 | 0.0212 | 0.0966 |
| c1_balanced | 0.2291 | 0.0471 | 0.0780 ± 0.0094 | 0.0159 | 0.0596 |
| c3_compact96 | 0.1805 | 0.0539 | 0.0829 ± 0.0145 | 0.0185 | 0.0644 |
| c2_strict | 0.2408 | 0.0455 | 0.0763 ± 0.0099 | 0.0079 | 0.0659 |
| c6_regularized | 0.1892 | 0.0471 | 0.0755 ± 0.0162 | 0.0132 | 0.0724 |
| c4_compact64 | 0.2447 | 0.0337 | 0.0567 ± 0.0008 | 0.0291 | 0.0580 |
| c7_low_lr_strict | 0.3111 | 0.0135 | 0.0257 ± 0.0003 | 0.0106 | 0.0722 |
| c5_two_layer | 0.1298 | 0.0101 | 0.0187 ± 0.0126 | 0.0053 | 0.0348 |

按既定排序选中 c0_incumbent，全部未通过 P/R 双门。加深、缩小和加大正则等候选
没有超过对照，不据此宣称 LSTM 的理论上限。后续先在共同 v5 数据确认原配置，
避免继续扩大旧数据搜索；有限训练信号实验可与 B4/B5 共用合理思路，但不改成图模型。

16 次运行的 manifest 哈希相同，但记录了五个不同 git_commit：761380d、0a1c98b、
7a10e71、38d4b5b、35ee064。对这些提交分别读取 Git blob 并作 SHA-256 比对，
确认搜索脚本、B3 入口、B3 模型、共享 trainer/heads/loss/dataset/metrics/schema、
remain 指标模块均字节相同。提交间改动为 B4 表示选项、审计工具/产物和文档；不能
直接声称整仓库固定同一 commit，但没有发现 B3 核心实现跨候选变化。

确认 B3 结束后，原服务器仓库快进到 dev_xwt 当前提交。五份以前生成、后来已入 Git
的未跟踪审计副本先与 origin/dev_xwt 字节比对，全部一致后保留到旧 benchmark 的
`audit_copies_before_pull_20260906`，没有删除 raw 或用户修改。B5 在独立 dev_xwt
checkout 中继续运行，不修改其代码、不重启其进程。

## 12. B2/B3 共同 v5 数据确认轮

旧数据两模型的搜索完整结束后，启动 `baseline_confirm_v5`。它顺序执行 B2 两颗 seed，
再执行 B3 两颗 seed；不增加候选，不 warm-start，不读取 test。和 B5 共用只读执行
checkout 的 `64366a6` 代码及 v5 manifest。启动前核对分支、提交和 manifest 哈希，
两个新输出目录均必须不存在。该队列只在各模型两次训练均成功后生成 selection；
这里只有一个候选，selection 仅用于验证完整性和汇总，不表示做了新超参搜索。

- B2：沿用 c5_event_w12，500 棵树、深度 5、lr=0.03、subsample/colsample=0.8、
  min_child_weight=3、reg_lambda=5、negative_cell_ratio=4、hot 权重=4、event 权重=12、
  n_jobs=4；输出 `models/tuning/b2_v5_confirmation_v1/candidate_c5_event_w12/seed{42,43}`。
- B3：沿用 c0_incumbent，单层 LSTM128、node_hidden128、node_embedding32、dropout0.25、
  batch32、lr=3e-4、weight_decay=0.001、max_epochs60、min_epochs12、patience12，
  原 base loss；输出 `models/tuning/b3_v5_confirmation_v1/candidate_c0_incumbent/seed{42,43}`。

本轮保持各模型旧轮的验证阈值候选，不把阈值搜索变化混成输入修正收益：B2 原有 16 点
扫描（0.55 至 0.95 的显式列表），B3/B4/B5 原有 12 点扫描（0.55 至 0.85 的显式列表）。
它们的 P/R 约束、标签与匹配规则相同，但阈值搜索范围不同，最终正式横向协议冻结时
必须明确统一候选范围或披露模型特定阈值调参预算；当前不能只写成所有评估配置完全相同。
不得为提高某个 baseline 数字临时换评分规则；正式协议调整须共同、预先固定并重评。

启动实测 `baseline_confirm_v5` pane PID=3825863，B2 seed42 Python PID=3828116，
CPU 活跃；B5 原 PID=3819541 继续运行，2/8 完成。两队列均尚未产生本轮完整结果。
每个运行保留 `training.log`、config、validation 指标及 provenance；并发耗时不用于
声称独占训练效率。下一次修改训练方案仍等待 B5 表示对照整轮完成。

## 13. 新数据整轮收齐

2026-09-06 15:59 HKT，`baseline_graph_v5` 实测 dead=1、exit=0，B5 为 8/8，
没有残留 B5 训练进程。B2/B3 确认队列此前已输出 `B2_B3_V5_CONFIRMATION_COMPLETE`。
全部 20 次 validation-only 结果已由服务器导出并随 `ba915ce` 归档为
`baseline_validation_v5_20260906_round4_complete.json`，同时保存 B2/B3/B5 selection
及 B5 四份漏报诊断。逐运行验证 validation 样本均为 2589、manifest 哈希均为
`66c6554d0ae1c7a293e482a201829a4c324f1d9d0b88934c1c2a6c1e27333183`，无 test 评估。
18 次神经模型产物内 git_commit 均为 `64366a6`；B2 两次产物没有写入 git_commit，
其执行版本仅有第 12 节启动预检记录，不能伪称产物内版本审计也通过。B2 后续须补齐
版本/库版本元数据，不回写旧结果来假装当时已记录。

### B5 完整表示对照

| 候选 | report P | report R | report F1 均值 ± std | hot F1 | upcoming R | will AP |
|---|---:|---:|---:|---:|---:|---:|
| control | 0.4614 | 0.2256 | 0.3002 ± 0.0115 | 0.4034 | 0.0106 | 0.2510 |
| history | 0.4876 | 0.2323 | 0.3142 ± 0.0051 | 0.4040 | 0.0159 | 0.2500 |
| identity | 0.4871 | 0.2189 | 0.2993 ± 0.0087 | 0.4018 | 0.0053 | 0.2533 |
| identity_history | 0.4630 | 0.2340 | 0.3100 ± 0.0009 | 0.4054 | 0.0132 | 0.2509 |

按既定稳健排序选中 history，其稳健分数约 0.3091，identity_history 约 0.3090，
两者很接近。history 相对 control 平均 F1 增加约 0.0140、precision 增加约 0.0262、
recall 增加约 0.0067；属于小幅改善，不是提前预测能力的突破，也不是显著性结论。
全部候选仍未通过双门。后续优先保留较简单的 history，不扩大无独立收益的 ID 搜索。

B5 对照与 history 的 validation-only 诊断再次指向低事件概率：

| 候选 / seed | best epoch | upcoming 概率中位数 | 起始时间在容差内 | 概率漏报 @0.55 | 时间漏报 @0.55 | 概率漏报 @0.20 |
|---|---:|---:|---:|---:|---:|---:|
| control / 42 | 12 | 0.02776 | 0.8783 | 187 | 0 | 184 |
| control / 43 | 17 | 0.03876 | 0.8254 | 187 | 0 | 175 |
| history / 42 | 13 | 0.02482 | 0.8254 | 187 | 0 | 177 |
| history / 43 | 17 | 0.03144 | 0.7989 | 185 | 0 | 168 |

每次 upcoming 分母为 189 个相关滑窗站点目标；诊断比例与正式 recall 的区别同第 10 节。
这些结果支持继续针对事件判别做受控训练实验，不支持仅调时间容差或降低报警阈值。

### B2/B3 确认结果

| 模型 / 既定候选 | report P | report R | report F1 均值 ± std | hot F1 | upcoming R | will AP |
|---|---:|---:|---:|---:|---:|---:|
| B2 / c5_event_w12 | 0.2926 | 0.2660 | 0.2785 ± 0.0031 | 0.3117 | 0.0635 | 0.2298 |
| B3 / c0_incumbent | 0.2172 | 0.0724 | 0.1085 ± 0.0088 | 0.2101 | 0.0106 | 0.0760 |

两模型均未通过双门。B2 两颗 seed 的 F1 为 0.2816/0.2754，B3 为 0.1173/0.0998。
这是输入/标签对齐后的确认，不是新的超参搜索胜出结果；不能把旧 v3 的更高分直接拿回
当前 v5 主表。B3 当前按固定顺序展平全图，再经单个 LSTM 状态生成各节点输出；
如后续保留非图节点级时间编码对照，必须作为明确的新架构候选，不暗改已报告模型。

## 14. 下一轮受控训练预案

本轮已完整结束，下一轮针对 B4/B5 的 history 结构检验以下四种训练方案，各 seed42/43。
这是预先定义的下一轮方案，当前实现已通过下述测试、服务器已启动，尚无完整轮分数，
不把计划当作已完成的实验。

| 候选 | 初始化 | upcoming / negative 权重 | 本阶段 epoch 上限 |
|---|---|---|---:|
| scratch_base | 从头 | 4 / 2 | 80 |
| scratch_signal | 从头 | 16 / 2 | 80 |
| warm_base | 同 seed 自身 history best.pt | 4 / 2 | 20 |
| warm_signal | 同 seed 自身 history best.pt | 16 / 2 | 20 |

warm 方案计入原第一阶段最多 60 epoch 的预算，故完整流程上限同为 80。实际第一阶段
B4 seed42/43 已训练 20/22 epoch、选中 10/12；B5 已训练 33/37、选中 13/17。
不能只计算选中 epoch 之前的耗时，也不能把重复使用父权重视作预训练免费。
逻辑每候选预算与实际复用同一父 checkpoint 的独占计算量分开报告；不同早停导致实际
更新次数可不同，这比较的是训练方案，不单独证明热启动的因果收益。

scratch 沿用每模型初始学习率、batch 与早停参数，cosine 上限改为 80；warm 重建优化器，
学习率为原值的 1/4（B4=7.5e-5，B5=3.75e-5），min_epochs=10、patience=10。
其余结构和 loss 权重不变，保持完整多任务输出；不使用 Focal、ID 或全图事件 context。
共同 v5 数据、两颗 seed、同一神经模型阈值列表、P/R 约束及事件匹配规则保持不变。
有限候选完成后按同一稳健排序选型，不从 test 选择，不以通过业务约束为由追加无限搜索。

实现必须严格校验父 checkpoint 的模型配置、seed、manifest、完成状态和完整 history；
只读同模型、同数据、自身 validation-only 第一阶段的权重，拒绝跨模型/数据借权重。
这是 weights-only 微调，不冒充恢复优化器/调度器的断点续跑。保留阶段 0 的验证与权重，
若后续没有改善须明确报告，而非隐藏父模型结果或只展示某个更好 seed。
记录父权重/配置/history 哈希、各阶段 epoch/更新次数/耗时及累计预算。
实现通过测试后才启动；旧实验在 Git 和不可变产物中保留，不增加旧数据兼容加载路径。

### 阶段训练实现

新增 `batch_factory_baseline_staged.sh` / `run_staged_baseline.py`，严格执行上表的
4 方案 x seed42/43，完整完成后才选型；数据审计、父选型不完整或输出已存在会直接拒绝。
神经训练器增加同模型同 seed 的 weights-only 入口，保存阶段 0、完整训练预算及父产物
哈希；既定阈值、loss mask、标签及评分函数不变。实现与实际服务器训练完成状态分开记录，
本节不提供尚未完成的阶段训练分数。

本地验证：暖启动、选型、导出、图表示、执行路由、事件消融、数据集、B3/B4/B5 模型、
对称性和事件诊断测试合计 88 passed、12 subtests passed；包含两个模型的真实小数据
第一阶段/第二阶段训练、父权重不变、阶段 0 保留及预算累计测试。shell 语法与
`git diff --check` 通过。2026-09-06 16:41 HKT 服务器只读预检未见 B2-B5 训练进程，
两个 dev_xwt checkout 均干净，确认不会通过更新代码影响上一轮正在运行的作业。

### 服务器启动记录

实际执行提交固定为 `83db56913323c252ef81c0fee0c37786929d7d7f`，独立执行目录仍是
`/home/sci/work/BSTAN_baseline_dev_xwt_v5`，分支 dev_xwt。原仓库也已 fast-forward 到
该提交；后续报告提交不更新运行中的执行目录。真实服务器 torch=2.7.0+cu128，CUDA 可用。
四个父 checkpoint 的完整性预检全部通过，manifest 仍为第 9 节的共同 v5 SHA-256。

| 父模型 / seed | 实际训练 epoch | 实际 optimizer steps | 选中 epoch |
|---|---:|---:|---:|
| B4 / 42 | 20 | 11520 | 10 |
| B4 / 43 | 22 | 12672 | 12 |
| B5 / 42 | 33 | 28512 | 13 |
| B5 / 43 | 37 | 31968 | 17 |

tmux `baseline_b4_staged_v1`、`baseline_b5_staged_v1` 并发运行，各自从
`scratch_base / seed42` 开始。进程检查见两个 `run_staged_baseline.py`，
PID=3908371/3908376，观察时均已存活约 1 分 15 秒；控制台已打印候选启动行。
输出在共同 v5 数据集的 `models/tuning/{b4,b5}_staged_training_v1`，每模型预定义 8 次。
存在并发，不将本轮耗时解释为独占运行效率。

随后本机锁屏，UU 工具明确报告无法自动解锁；尚未读取到首个 epoch，不能把进程启动
认定为完整实验成功。恢复窗口访问后先检查既有会话、训练日志和完成标记，不重复启动，
不修改这两个运行目录的代码；待各模型全部 8 次完成后再进行结果比较。

## 15. 阶段训练完整结果与 precision 复核

UU 访问恢复后观察到首轮 epoch，之后按原进程句柄等待：B4 2/8、6/8、8/8，
B5 1/8、3/8、5/8、7/8、8/8；最终两会话均 dead=1、exit=0，原进程均已结束。
未重启作业或中途改参数。完整产物已归档于 `baseline_validation_v5_20260906_staged_complete.json`
及两模型的 `baseline_b*_staged_{selection,protocol}_20260906.json`。

新增 16 次均为同一 v5 manifest、执行提交 83db569、2589 个 validation 样本且 evaluate_test=False；
累计预算逐项满足 parent_steps + stage_steps。新增实际训练 290 epoch，阶段耗时求和约
7198 秒；因并发，这不是墙钟总时长，也不包含已经归档的父阶段费用。

| 模型 / 方案 | P | R | F1 均值 ± std | upcoming R | will AP |
|---|---:|---:|---:|---:|---:|
| B4 / scratch_base | 0.4149 | 0.2323 | 0.2975 ± 0.0034 | 0.0079 | 0.2209 |
| B4 / scratch_signal | 0.3487 | 0.2020 | 0.2558 ± 0.0069 | 0.0370 | 0.2067 |
| B4 / warm_base | 0.4915 | 0.2172 | 0.3006 ± 0.0025 | 0.0079 | 0.2194 |
| B4 / warm_signal | 0.4835 | 0.2155 | 0.2974 ± 0.0007 | 0.0026 | 0.2230 |
| B5 / scratch_base | 0.4783 | 0.2323 | 0.3121 ± 0.0037 | 0.0159 | 0.2495 |
| B5 / scratch_signal | 0.3777 | 0.2441 | 0.2929 ± 0.0206 | 0.0476 | 0.2284 |
| B5 / warm_base | 0.4876 | 0.2323 | 0.3142 ± 0.0051 | 0.0159 | 0.2500 |
| B5 / warm_signal | 0.4876 | 0.2323 | 0.3142 ± 0.0051 | 0.0159 | 0.2500 |

全部未通过双门。选型器均选 warm_base，但须正确解释阶段 0：B4 只有 seed43 选中新阶段
epoch3，F1 从父阶段 0.2966 增至 0.3032；seed42 保留父权重。B5 两个 warm 方案、两颗 seed
全部选中 epoch0，数值就是原 history 的结果，不能宣称续训带来提升。后续使用 B5 原 history
即可得到相同权重，不要求部署时额外执行一段没有收益的训练。

这轮不支持继续无边界提高 upcoming 权重或重复低学习率全模型续训。更大的 upcoming 权重
提高部分提前召回，同时 precision/F1 下降。有限搜索未证明模型理论上限，也未达到业务要求。
当前累计已完成新 v5 验证运行 36 次，加旧 v3 的 112 次，共 148 次开发运行；不能只披露赢家预算。

### 下一步依据误报来源，而不是追逐单个 precision

用户指出 precision 偏低。它衡量报出的站点事件中有多少匹配真实事件及起始时间；低分
说明当前报警质量不足，但不存在“低于主模型就不是有效 baseline”的统计学门槛。
有效对照依赖实现、共同数据与输出、充分且有限的调参、透明预算和独立评价，不依赖强行过门。

扩展 `diagnose_baseline_events.py`，保持全部预测、mask、阈值和规范评分不变，只将 report
误报分成互斥项：真实事件但起始时间超容差；其余负例按可观测未来少于 8 窗口、未来有 hot
但不构成合格事件、未来完全无 hot，再按历史末帧 hot/cold 分层。输出逐资源预测/命中/误报
数量和预测时长分位数，同时核对分解重建的 precision 等于规范评分。
未来 hot 和剩余观测长度只用于回顾诊断，不可成为部署特征或过滤报警的规则。

主实验历史 P=0.8504/F1=0.5094 仍是单次保存结果。当前服务器主代码固定 f322dbf，train.py
Git blob 为 f3756caf2ca64db53a94bd0060f2a3340c15b2f7，已与本地该 Git 提交核对相同；
不同于 origin/dev_tyx 后续代码。重评时应使用该实际源码、保存的 scaler/graph/权重及已审计
validation episode，不重新拟合 scaler，也不重训主实验或修改 dev_tyx。原始与共同解码结果
需分别保留；尚未完成此重评，不声称评估设置解释了实际性能差距。

## 16. 本轮误报定位与正式对比前置问题

四个已选 checkpoint 的 validation 误报诊断已完成，使用原保存阈值，不重新选择阈值。
原脚本误把 manifest 的 5 类资源类型列表按节点编号索引，服务器报 IndexError；
已改为读取并严格核对 node_catalog.csv，新增实际多节点/少类型回归测试后重跑。
该问题仅影响诊断报表输出，没有改动训练、预测和评分。
完整 JSON 已归档为 baseline_false_alarm_{b4,b5}_seed{42,43}_20260906.json。

| 模型 / seed | 保存阈值 | report P | 全部误报 | 历史 hot、未来有 hot 但不构成合格事件 | 未来无 hot | 真实事件起点错位 |
|---|---:|---:|---:|---:|---:|---:|
| B4 / 42 | 0.55 | 0.5210 | 57 | 41 | 15 | 1 |
| B4 / 43 | 0.60 | 0.4621 | 78 | 46 | 31 | 1 |
| B5 / 42 | 0.55 | 0.5077 | 64 | 47 | 17 | 0 |
| B5 / 43 | 0.55 | 0.4675 | 82 | 51 | 31 | 0 |

计数单位为相关的“预测窗口 x 站点”，不是独立物理事件。四次合计 281 个 report 误报中，
185 个（65.8%）属于历史 hot 但未来不满足连续至少 8 窗口的事件要求，94 个未来无 hot，
只有 2 个是真实事件的起点错误。短观测 horizon 类误报为 0。
因此当前优先问题是事件持续性判别，不是起点容差；不能通过放宽容差解决主要误报。
已有 upcoming 概率诊断则显示另一端的提前事件置信度过低。单纯增大正例权重已在第 15 节
完整试验中降低总体 P/F1，故不能继续重复同方向搜索。

### 主实验保存分数尚未在当前源码复现

新增只读工具 reevaluate_main_validation.py，仅选择已审计的 17 个 validation episode，
核对 2589 个窗口锚点、已有数据审计哈希、保存 scaler 与节点顺序；不拟合数据、不重训，
不向 model.predict 传入未来目标张量。主源码从指定服务器目录隔离导入，baseline 的
评分实现作为独立模块载入，计划对相同预测交叉核对评分。原解码与共同解码分开输出，
不进行 threshold 选型或 test 评分。

首次实测在 strict=True 加载 checkpoint 时失败：当前主源码 f322dbf 要求
cluster_emb.weight，但冻结 checkpoint 中缺失该权重。没有使用 strict=False、随机补权重、
跳过层等方式继续评分。Git 历史显示 cluster_emb 曾在 7b2fc02 引入，但尚未确定该权重
实际对应的完整历史源码与训练环境。历史 P=0.8504/R=0.3636/F1=0.5094 仍保留为保存记录，
不应据此声称已完成当前源码上的同口径重评，也不代表原历史分数必然错误。

工具将严格加载失败与已完成的 cohort 前检单独记录，并以失败状态退出，不产生伪造的
重评分数。下一步先定位真实匹配的主实验源码或取得可严格加载的新 checkpoint，再比较
完整相同口径的模型结果；不能仅为复现一个漂亮数字而选择兼容补丁。

### 历史状态的因果性问题

共同数据路径把完整 episode 的 ops_hot_mask(...)[t-1] 当作 hist_last_hot；该 mask 经过
完整序列的短段过滤和间隙填充。最小反例：相同的前 3 个 hot 窗口，后续继续到 10 窗口时，
完整序列会把第 3 窗口标为 1；若之后立刻结束则为 0。只看已观察的前缀时，8 窗口持续性
还未满足，不能知道完整序列的这个值。本地反例测试已通过。

服务器重评前检进一步确认：共同 validation 的 33,473 个可评估站点窗口中，有 199 处
完整序列与前缀重算的历史 hot 值不同。结果与严格加载失败一起归档于
main_validation_replay_attempt_20260906.json（status=failed、model_scored=False）。
199 不是误报数、正例数或指标下降量；本次没有完成完整输入的因果性审计，也没有测量
替换历史状态后的 P/R/F1，不把这一差异比例解释为实际性能影响。

使用未来信息生成训练目标本身并非此问题；问题是把完整序列平滑后的值用于预测输入或
解码判定。当前 baseline 不把 hist_last_hot 输入网络，但解码用它将起点置 0，loss 也用来
划分正例权重；主实验还有依据该状态的 ongoing 强制报警策略。该问题属于共同数据/解码
协议，不应只为提高 baseline 分数而单方面修改，也不能直接把这个字段增加为网络输入。
尚未量化其对当前各模型 P/R 的影响，不宣称它解释了主模型与 baseline 的全部差距。

推荐与主实验统一确认：历史状态只能从截至锚点的原始 operational 状态计算，明确区分
“当前有拥堵迹象”与“过去已持续达到门槛”，必要时提供过去连续时长、队列变化等因果特征。
未来的 hot/event 标签可保持离线监督定义。仅此历史字段修正不要求重采 raw，但需要重建
相关派生字段并让所有模型在同一版本重训或按影响范围重评；正式表不能混用两种协议。

### 后续有限调整方案

在上述共同协议确认前，不追加大规模正式调参。保留原 checkpoint 和结果，不声称已到
方法理论上限，也不以低于主模型为由将 baseline 判为无效。共同协议确认后的下一轮应有限：

1. 优先检验训练集中的困难负例：未来有短暂 hot、却不构成合格事件的站点。仅作为训练
   监督分层或采样依据，不作为推理过滤条件；与原训练方案做固定预算对照，检查 precision
   改善是否以 recall 大幅下降为代价。
2. 如保留显式历史状态特征或 continuation/onset 分支，必须使用前缀可计算的状态，作为
   新的清楚命名的架构候选，不能悄悄改变 B4/B5 定义或借用主实验权重。
3. 每轮预先固定少量候选、seed42/43 与预算，完整结束再比较。以 P/R/F1、upcoming recall
   共同判断，不追逐某一颗 seed 的 precision；定型后补独立 seed，并在未参与调参的独立
   episode 上最终评价。此前反复查看过的 test 不再宣称完全未触碰。

本轮本地事件诊断和主重评工具测试合计 8 passed；覆盖资源目录、误报分解、只选 validation、
评分模块隔离、原解码/共同解码区别及上述历史状态反例。真实 checkpoint 已触发严格加载
拒绝，当前并未成功生成主实验重评分数。

## 17. 历史状态敏感性实测与困难负例对照预案

2026-09-06 实测完成，产物为 baseline_history_causality_20260906.json。保持原 X、目标、
权重和保存阈值，仅比较完整序列历史 hot 与截至锚点的前缀重算值，不进行 test 评分。

| split | 样本 | 可评估站点窗口 | 历史标志差异 | 正例中的历史标志差异 | ongoing 损失分组差异 |
|---|---:|---:|---:|---:|---:|
| train | 13813 | 177484 | 1146 | 451 | 0 |
| validation | 2589 | 33473 | 199 | 68 | 0 |

当前损失把 event_start=0 的正例也纳入 ongoing，故上述历史标志变化没有改变事件损失分组；
loss 的其他项不读取 hist_last_hot，网络输入也不包含该字段。本地测试验证分组不变时当前
loss/梯度不变，但这不证明重新训练后 checkpoint/threshold 选型不变。

| checkpoint | 原 report P/R/F1 | 前缀 report P/R/F1 | F1 差值 |
|---|---|---|---:|
| B4 warm_base / 42 | 0.5210 / 0.2088 / 0.2981 | 0.5042 / 0.2020 / 0.2885 | -0.0096 |
| B4 warm_base / 43 | 0.4621 / 0.2256 / 0.3032 | 0.4000 / 0.1953 / 0.2624 | -0.0407 |
| B5 warm_base / 42 | 0.5077 / 0.2222 / 0.3091 | 0.5077 / 0.2222 / 0.3091 | 0.0000 |
| B5 warm_base / 43 | 0.4675 / 0.2424 / 0.3193 | 0.4481 / 0.2323 / 0.3060 | -0.0133 |

who P/R、报警数、upcoming recall 均不变；变化来自 ongoing 的起点置零规则。
这不是历史状态问题已解决的证明，也不是主模型差距的解释；正式口径仍需双方确认。
现有主模型 Git 历史中，7b2fc02 之前尚无 checkpoint 所需事件头，之后又包含 cluster_emb。
目前没有找到能无补参数地匹配该保存权重的完整 Git 版本，已询问对应源码快照/可加载权重。

### 下一轮：仅调整短 hot 事件负例的损失权重

作为有限开发试验，继续冻结 v5 的已有共同数据、标签和评分规则，不将结果直接写成新的
正式因果 benchmark。针对第 16 节占多数的误报，使用训练目标识别困难负例：站点有效、
未来可观测至少 8 窗口、可观测未来有 hot、但 event_will=0。future hot/观测长度只决定
loss 权重，不进入输入或解码，不用于过滤验证报警，不改变任何标签。

| 候选 | event_short_hot_fp_multiplier | 初始化 | 上限 |
|---|---:|---|---:|
| weight1_control | 1 | 从头 | 60 epoch |
| weight2 | 2 | 从头 | 60 epoch |
| weight4 | 4 | 从头 | 60 epoch |

B4/B5 各三个候选 x seed42/43，共 12 次。结构固定为已选 history-only；各模型的原学习率、
batch、dropout、weight decay、早停及正例/一般负例权重不变。父 history 产物仅提供配置及
溯源，绝不加载权重。本轮不复用旧 warm 控制组，而是同一新源码重跑从头 control。
沿用原 12 点阈值列表、P/R 约束及完整轮稳健选型，所有候选完成后才作判断。
不同 loss 权重下的 total loss 不直接跨候选比较，核心依据为共同 P/R/F1 与 upcoming recall。

倍数 1 跳过加权分支，保留原算法精确行为；测试覆盖正例、无 hot 负例、padding、未观测
未来、短 horizon 不被错误加权，梯度方向及标签不变。运行器测试覆盖六次全部从头训练后
才选型，保证不会误走 warm 路由。本轮结束后对选择结果做前缀解码敏感性复核，不把高
precision、低 recall 的候选视为成功，也不在结果不理想时无限追加倍数。

执行入口（执行目录和环境由服务器实际配置）：

```bash
STUDY=hard_negatives bash batch_factory_baseline_staged.sh B4
STUDY=hard_negatives bash batch_factory_baseline_staged.sh B5
```

默认输出 models/tuning/{b4,b5}_short_hot_negative_v1，已存在则拒绝覆盖。本段是预先固定的
方案与实现说明，实际启动、完成状态和结果另行记录；不能将计划当作完成。

实现验证：相关损失、训练路由、续训、因果历史审计、评分重放、事件诊断、图对称性、
调参与数据集测试合计 62 passed、8 subtests passed；shell 语法和 git diff --check 通过。

## 18. 困难负例完整结果：precision 提升不等于综合改善

### 执行与可复现性

本轮训练源码固定为 `265801c30d5e1264fc98828d5a9c807ce1deb7b4`，始终在 dev_xwt 的
独立执行 checkout 运行。B4/B5 各三个候选、seed42/43，共 12 次全部完成，两个 tmux
训练进程均正常退出（exit=0）。期间未修改执行 checkout，也未更换参数、标签或评分。

产物：

- `baseline_validation_v5_20260906_short_hot_complete.json`：包含当前 v5 全部 48 次已完成验证运行。
- `baseline_b4_short_hot_selection_20260906.json`、`baseline_b5_short_hot_selection_20260906.json`：完整排名。
- `baseline_b4_short_hot_protocol_20260906.json`、`baseline_b5_short_hot_protocol_20260906.json`：开训前写入的六次配置。
- `baseline_short_hot_history_causality_20260906.json`：四个赢家 checkpoint 的保存分数复现和前缀敏感性复核。

12 次产物的源码、manifest 哈希、60 epoch 上限、evaluate_test=False、随机初始化及
parent_epochs_trained=0 均已核对。实际新增 332 epoch，各运行耗时合计 8099.58 秒；
因为两个模型并行，该合计不是墙钟耗时。新控制组的两颗 seed 在每个模型上均精确复现
此前 candidate_history 的全部 validation 指标、best_epoch 和 epochs_trained。
这支持默认倍数 1 没有改变旧算法行为，不是另一个不可比控制组。

### 同轮比较

以下均为 validation 的 seed42/43 均值，F1 后为总体标准差，不是 test，也不是独立事件
重复实验的置信区间。采用同一既定阈值搜索与选型规则；各运行实际阈值记在原产物中。

| 模型 | 短 hot 负例倍数 | report P | report R | report F1 | upcoming R | will AP |
|---|---:|---:|---:|---:|---:|---:|
| B4 | 1（控制） | 0.4835 | 0.2155 | 0.2974 +/- 0.0007 | 0.0026 | 0.2230 |
| B4 | 2 | 0.4860 | 0.1936 | 0.2762 +/- 0.0017 | 0.0079 | 0.2178 |
| B4 | 4 | 0.5260 | 0.1566 | 0.2412 +/- 0.0004 | 0.0053 | 0.2102 |
| B5 | 1（控制） | 0.4876 | 0.2323 | 0.3142 +/- 0.0051 | 0.0159 | 0.2500 |
| B5 | 2 | 0.5506 | 0.1936 | 0.2847 +/- 0.0010 | 0.0132 | 0.2438 |
| B5 | 4 | 0.7381 | 0.1347 | 0.2270 +/- 0.0112 | 0.0053 | 0.2407 |

所有运行均未同时达到 P>=0.80、R>=0.35。两模型完整轮次都选择 candidate_weight1_control，
因此不把倍数 2/4 设为默认，也不替换此前保留的跨轮工作候选。

以 B5 两颗 seed 的站点滑窗计数之和说明代价：控制组报警 284 次，其中匹配正确 138 次、
误报 146 次；倍数 4 报警 110 次，正确 80 次、误报 30 次。误报减少的同时丢失了 58 次
原本匹配正确的报告。单看 seed42，倍数 4 已达到 P=37/46=0.8043，但 R=37/297=0.1246，
仍远未通过双门。这里的计数是同一 validation 在两颗 seed 上的重复站点滑窗预测，
不能当成两倍独立事件数。

结论：困难负例权重确实使报警更保守，但没有充分学会分辨哪些热点将持续；提前预测
仍弱，AP 也未改善。本轮不能写成“B5 已达到 0.8 的合理效果”，也不能据此扩展无界
权重搜索。低 precision 反映当前能力不足，但不自动否定 baseline 的对照价值。

### 前缀解码复核

保持相同权重、X、目标、mask 和保存阈值，四个赢家的原保存 report 指标均严格复现。
仅将历史 hot 换为截至锚点的前缀计算结果，得到：

| 模型 / seed | 原 P/R/F1 | 前缀 P/R/F1 |
|---|---|---|
| B4 / 42 | 0.5210 / 0.2088 / 0.2981 | 0.5042 / 0.2020 / 0.2885 |
| B4 / 43 | 0.4459 / 0.2222 / 0.2966 | 0.3919 / 0.1953 / 0.2607 |
| B5 / 42 | 0.5077 / 0.2222 / 0.3091 | 0.5077 / 0.2222 / 0.3091 |
| B5 / 43 | 0.4675 / 0.2424 / 0.3193 | 0.4481 / 0.2323 / 0.3060 |

前缀 F1 均值 B4=0.2746、B5=0.3076；报警数、who 指标、upcoming recall 不变，
变化仍来自起点解码。该复核不是替换共同协议，也不证明所有输入都已通过因果性审计。
前缀标志差异和 ongoing 损失分组差异计数与第 17 节完全一致。

### 阶段判断与下一步边界

在已经尝试的配置范围内，B4/B5 进入了经验上的平台期：历史汇聚只有小幅改善；
继续训练未稳定改善两模型；增加正例信号或本轮困难负例权重没有改善综合表现。
这不构成方法理论上限，更不能证明所有合理结构/训练方案都已穷尽。

因此不再以“precision 没有主模型高”为理由立即追加又一轮权重或 epoch 搜索。
继续正式实验前，优先完成以下前置事项：

1. 找到能严格加载主 checkpoint 的实际源码快照，或获得与现有源码严格匹配的主模型
   权重；历史单 checkpoint 的 P=0.8504 不能直接与 baseline 双 seed 均值等同比较。
2. 双方统一决定历史 hot 是否必须前缀可计算，并固定解码和阈值选型协议。当前完整序列
   历史平滑有未来依赖，不能在主模型和 baseline 上使用不同口径后称为正式对比。
3. 在协议冻结后固定工作配置，补独立 seed 和未用于调参的 episode 评估；此前反复查看
   的 test 必须披露为开发中已接触，不作为新的完全独立验证。
4. 只有这些检查仍指向模型表达能力不足时，才为 B4/B5 设计一个预先命名、有对照和
   预算的新结构候选。不能改成主模型结构却继续用原 baseline 名称。

本报告保留不理想结果与全部搜索预算，不承诺每个 baseline 都能达到业务 P/R 门槛。
当前是可追溯的开发阶段证据，尚不具备冻结最终横向论文指标的条件。

## 19. dev_tyx 新提交改变了比较对象

### 已拉取并核对的源版本

2026-09-06 拉取 `origin/dev_tyx` 后，引用从 `52e8643` 更新为
`20c40e230aedee6aef2429d352413fbcf0fa571a`（提交时间 2026-09-06 12:37:46 +08:00）。
仅更新远端引用并读取 Git 对象，当前分支仍是 dev_xwt，未 merge、checkout 或修改主实验。

新《F1提升实验历程.md》把 `dense_i1_a1_prefix8` 标为正式保留，声称其数据包为
`raw_data/dense_i1`，140/28/36 个 episode、20826/4331/5354 个 train/val/test 窗口。
这些数量和新分数目前是文档记录，不是本代理已用实际 checkpoint 复现的结果。
它不再是此前 n10_i1_all_usable 的 134 episode 对照。

### 定义差异而非仅超参数变化

下表的新列直接取自该提交的 `factory_bn/configs/FactoryBN_dense_f1_p80.json` 和函数实现。

| 项目 | 当前 baseline v5 | 新主配置 |
|---|---|---|
| 输入 / 占用预报 | 30 / 15 个 60s 窗口 | 仍为 30 / 15 |
| 占用平滑 | hot_min=8、gap=1 | 相同 |
| 新发生事件 | 未来最长热段至少 8 窗口 | 仍至少 8，但只计 start<=2 |
| 进行中事件 | 预报窗内剩余也须至少 8 | 历史已热且最长段 start=0 时，剩余至少 1 |
| 报告起点容差 | 3 窗口 | 相同 |
| checkpoint 双门 | P>=0.80、R>=0.35 | P>=0.80、R>=0.70 |
| 神经验证阈值表 | .55/.60/.62/.65/.68/.70/.72/.75/.78/.80/.82/.85 | .55/.60/.65/.70/.75/.80/.82/.85/.88/.90/.94/.98 |
| 解码 | 事件头概率、历史热时起点归零 | 另有 prefix 头、占用并集、ongoing 抬升等 |
| 本次阶段初始化 | baseline 各自从头或已披露的同模型续训 | 从 dense_i1_a1_f180 权重微调 16 epoch 上限 |

例如，同一段已经持续很久、未来只剩 3 分钟的热状态，在旧任务里是负例，新定义中是正例。
第 18 节加权压制的部分“困难负例”在新定义下可能变为正例，因此不能把旧调参结论原样
迁移到新任务，也不能用新旧 F1 的差值来归因模型能力。近期 onset 与剩余时长可以是合理
业务任务，但必须明确命名，不等同于覆盖整个 15 分钟起始范围的预测任务。

从新旧 Git 源码提取并实际执行 `node_event_targets`，四个固定小例全部符合如下结果：

| 输入热段与历史状态 | 当前 v5 will/start/duration | 新配置 will/start/duration |
|---|---|---|
| 已热，未来从 0 起剩 3 窗口 | 0/0/0 | 1/0/3 |
| 当前冷，从 5 开始持续 8 窗口 | 1/5/8 | 0/0/0 |
| 当前冷，从 2 开始持续 8 窗口 | 1/2/8 | 1/2/8 |
| 当前冷，未来从 0 起仅 3 窗口 | 0/0/0 | 0/0/0 |

这只是源码级定义反例测试，没有重标真实数据、训练模型或读取新 test 指标。

### 需与 tyx 统一的三个问题

1. 文档和配置冲突。新版《模型评估指标.md》的表格仍写事件最短 5、start<=2，而正文
   又说配置没有 start 上限、所有正例至少 8、recall gate=.35；实际 JSON 是最短 8、
   ongoing 最短 1、start 上限 2、recall gate=.70。《F1提升实验历程.md》的最终表与
   JSON 更接近。建议把最终 checkpoint 所携契约作为可核验来源，修订矛盾文档，并由双方
   明确是否采用这套新任务，不由 baseline 私自挑一个容易得高分的版本。
2. 重评入口漏传参数。该提交 `train.py:861` 明确传
   `event_ongoing_min_windows=cfg.get(...,1)`，但 `eval_ckpt.py:123` 调 `_epoch_loop`
   时没有该参数，`compare_train_modes.py:332` 的 `_event_kw` 也没有。
   `_epoch_loop` 在 `train.py:349` 默认 None，最终 `node_event_targets` 把 None 解释为
   与 event_min 相同，故剩余 1--7 窗口的 ongoing 可能训练评估算正例、重评却算负例。
   上表的 ongoing 剩 3 窗口即可复现这种差异。建议各入口使用同一个显式契约参数对象，
   并加入训练评估/重评目标和分母相等的回归测试。这里只记录发现，未修改 dev_tyx。
3. `prefix8` 不是因果历史前缀修复。它的 prefix_mlp 预测未来从首窗开始连续热多久；
   新 `dataset.py:477` 仍从整局 hot 取 hist_last_hot。对比 AST 确认 ops_hot_mask、
   ops_occupancy_raw、smooth_occupancy_runs 与当前基线版本均相同，所以第 17/18 节的
   历史可观测性风险并未因名称含 prefix 而消失。主模型新 split/force/prefix 解码对这个
   历史标志的使用更广，不能以旧模型敏感性数字推断新模型影响大小，需新权重实测。

### 服务器实际状态与待提供产物

只读检查了两个用户此前提供的路径：

- `/home/sci/work/isaac_factory`
- `/home/sci/work/BNPDFormer/_isaac_factory`

用户补充主实验在 BNPDFormer 后，再检查外层 `/home/sci/work/BNPDFormer`：其 HEAD
是 `ba69f23`、分支 main，为论文/文档仓库，存在用户未提交的 README.md 修改，未触碰。
`ls -ld` 和 `realpath` 确认 `_isaac_factory` 是指向 `/home/sci/work/isaac_factory` 的
符号链接。因此上述两个入口实际是同一训练仓库，不应描述为两份独立主实验。

这两个入口的 Git HEAD 均显示 `f322dbf`。对应 hc_factory/PDFormer 下的
`raw_data/dense_i1`、`libcity/cache/model_cache/dense_i1_a1_prefix8` 和
`libcity/cache/model_cache/dense_i1_a1_f180` 均不存在。对已知 PDFormer 目录的限定深度
源码快照检查仅找到现行 factory_bn/model.py；这不证明服务器其他未知路径也没有产物。
已询问新数据/权重是否在其他机器或尚未上传。用户随后要求先继续当前 baseline 优化，
因此新主实验产物不作为当前 v5 开发诊断的阻塞条件，但仍是新口径正式对比的前置条件。

不能用旧 134 episode 冒充新 dense_i1，不能只把旧目录改名，也不能自行重训/改写 tyx
主实验来填补这些证据。现有开发结果和两套 Git 引用均保留。

### 获取并确认后才能执行的迁移

1. 固定实际 bundle、checkpoint、config、完整训练源码/依赖、训练历史和 split 身份哈希。
   先做严格加载及 validation 重放；不补随机权重、不使用 strict=False。
2. 在新 benchmark 目录构建统一目标，明确记录 min_hot、min_event、ongoing_min、max_start、
   历史状态来源、节点集合、阈值选型。若新任务包含额外的较远历史特征，明确各模型相同的
   可用信息范围；可选择模型不使用某类输入，但不能宣称它们都只看了相同 30 分钟。
3. 逐 episode 和逐锚点核对输入/目标/mask；训练评估与离线重评必须得到同一事件分母。
4. B2--B5 使用各自已披露的合理初始方案在新契约下重训，不把改标签的收益算作结构提升。
   是否引入额外 prefix 辅助头需作为显式模型候选，保持其非图/LSTM/GCN/GAT 身份。
5. 在统一定义下做有限验证对照，定型后补独立 seed 和独立 episode，最终报告区分数据/
   标签变更、训练优化、解码优化的贡献。当前不能跳过这些步骤直接出新的最终主表。

## 20. 当前 v5 的训练集判别能力检查

用户确认先继续优化现有 baseline，不等待新主实验产物。当前开发继续固定第 9 节 v5
数据和原事件定义，不迁移到 dense_i1，不修改标签、split、阈值门槛或匹配容差。

前三轮已分别检查历史表示、续训/增加正例权重、短热负例加权。最后一轮提高负例权重
减少误报的同时也丢失大量真阳性，没有超过对照。下一步先区分两种原因：同一已选权重
是否在训练集也不能识别 upcoming，还是训练集识别良好、验证集失效。没有这一步证据，
不继续重复加权搜索，也不直接把低召回归因于网络容量或过拟合。

`diagnose_baseline_events.py` 增加显式 `--split train|validation`，拒绝 test；
保持既有目标、mask 和报告函数，读取同一模型的最佳 checkpoint，不重新选 epoch。
在 all_events、ongoing_vs_negative、upcoming_vs_negative 和 events_vs_short_hot_negative
四个事后子集中增加 tie-aware AP、ROC-AUC、正例率和 AP/正例率。ongoing/upcoming 仍使用
同一个模型 will 分数，并非新增了预测头。子集划分使用真实未来，仅用于误差诊断，
不能作为推理门控；这里的 tie-aware AP 不替换既有正式评分实现。

预先固定检查 B4/B5 的 history seed42/43，共四个权重、每个 train 和 validation，
合计八份诊断。验证集报告阈值使用各权重已保存值；其他阈值只作曲线诊断，不选阈值。
训练集结果不当作泛化成绩，也不按它筛选模型。根据完整八份结果决定是否值得增加
一个事件判别结构候选，或以已测试范围的限制结束当前轮次。

本地六项测试通过，覆盖两种允许的 split、test 拒绝、无效节点剔除、空集合、并列分数
的 AP、误报互斥分组和原报告一致性。

### 实际诊断结果

服务器 `baseline_fit_audit_v1` 于 2026-09-06 22:20:12 HKT 正常退出（status=0），
8/8 诊断完成，代码固定为 `0a8c0b5`。每份 train 为 13813 窗口、validation 为 2589；
四份 validation 的 P/R/F1 和 ongoing/upcoming recall 与原权重保存值全部一致。
聚合产物 `baseline_history_fit_20260906.json` 记录输入权重及原诊断文件哈希，随
`862d366` 归档。未重新训练、选择权重或评估 test。

| 模型 / seed | train upcoming R @原阈值 | val upcoming R | train upcoming AP | val upcoming AP | train 短热区分 AUC | val 短热区分 AUC |
|---|---:|---:|---:|---:|---:|---:|
| B4 / 42 | 0.0306 | 0.0000 | 0.0803 | 0.0214 | 0.6585 | 0.4869 |
| B4 / 43 | 0.0593 | 0.0053 | 0.1019 | 0.0271 | 0.7106 | 0.5503 |
| B5 / 42 | 0.0306 | 0.0106 | 0.0690 | 0.0320 | 0.6498 | 0.5471 |
| B5 / 43 | 0.0707 | 0.0212 | 0.1067 | 0.0327 | 0.7115 | 0.5625 |

upcoming AP 的负例为所有有效无事件站点，排除 ongoing 正例；train/validation 正例率
分别约 0.00592/0.00566。短热区分 AUC 比较真正事件与“未来有热但不足事件条件”的
负例，排除全冷和观察期不足。后者的验证 AUC 接近 0.5，说明当前事件头在这一困难组
没有稳定的排序优势；这不是以子集过滤方式修饰正式评分。

训练 upcoming AUC 为 0.922--0.943，高于验证 0.808--0.839，因此不能声称网络完全
没有学会提前信号。更准确的结论是：排序已有信号但高分区精度不足，原阈值下训练召回
本就很低，并伴随泛化退化。低召回既不是仅由时间起点误差造成，也不能仅归因于过拟合。

## 21. 训练窗口抽样受控对照

源码核对发现共享神经 trainer 当前使用均匀 shuffle；此前的事件权重、Focal 和困难
负例权重实验都改变 loss，没有改变每个 epoch 抽到哪些训练窗口。本轮单独检验训练
样本暴露方式，不添加新事件头、不复制主模型结构，也不改评估协议。

| 候选 | 训练窗口权重 | 每 epoch 抽取数 |
|---|---|---:|
| uniform_control | 原均匀、无放回 shuffle | 13813 |
| event4 | 有任一有效事件的窗口权重 4，其余 1；有放回 | 13813 |
| upcoming4 | 有任一有效 start>0 事件的窗口权重 4，其余 1；有放回 | 13813 |

两个模型各 3 候选 x seed42/43，共 12 次从头训练，全部 60 epoch 上限；沿用各模型
history 方案的网络、loss、batch、学习率、早停及验证阈值表。每个 epoch 的 optimizer
更新次数不变，但抽样可能重复窗口、实际早停 epoch 可不同，不能声称独立样本增加，
也不能把逻辑 epoch 数当作相同数据覆盖率。所有未被加权的负窗口仍有非零抽取概率。

目标权重仅从 train split 的现行 Dataset 生成的 event_will/event_start/occ_node_mask
计算；不读 validation/test 标签决定抽样。校验无有效目标时直接报错，不借用评估集
事件补足。validation/test 的窗口、顺序、权重和分母不变；同一个窗口的 X/标签/mask
不因被重复抽到而变化。抽样不做 importance correction，因而明确改变了所有任务的
训练暴露分布，不把它误述为只改 event loss 的等价实现。

`STUDY=sampling bash batch_factory_baseline_staged.sh B4|B5` 沿用现有执行入口和执行副本，
不新建源码目录。study_config 在开跑前固定三组配置；每个运行的 metadata 保存
抽取数、目标窗口数和期望抽中比例。默认 factor=1 保持旧 loader 的精确 shuffle
顺序，不额外遍历标签；B3 也可显式使用同一训练选项，但本轮仅运行 B4/B5。

验证：67 tests、7 subtests passed，含真实 B4/B5 一轮训练、默认顺序复现、seed
确定性、固定抽样长度、有效节点门禁、禁止借评估集事件和 train-only 产物记录。
完成全部候选后按现有 validation 稳健排序选型，重做保存分数和前缀历史敏感性核验。
以上为开训前预案，不预设这一训练因素会提高 precision 或 F1；完整结果见下文。

### 启动记录

源码固定为 `009fe42`，复用 `/home/sci/work/BSTAN_baseline_dev_xwt_v5`，没有新建
源码或构建目录。结果分别写入现有 `factory_pdformer_134_v3/models/tuning` 下的
`b4_event_sampling_v1` 和 `b5_event_sampling_v1`；运行期间不更新执行目录源码。

最初直接执行脚本因文件没有执行权限而在进入 Python 前退出，两次均未产生训练结果。
确认原 pane 已退出、输出目录不存在后，复用同名 tmux 会话并改为 `bash` 调用；
未修改文件权限，未覆盖结果。另一次 B5 命令粘贴在外层 shell 被拒绝，未进入 tmux。
以下为纠正调用后验证到的实际进程，失败的启动尝试不计作训练候选。

2026-09-06 22:53:08 HKT 核查：

| 模型 | tmux 会话 | pane PID | 训练 Python PID | 当时输出 |
|---|---|---:|---:|---|
| B4 | baseline_b4_sampling_v1 | 4119696 | 4119986 | epoch 7，进程存活 |
| B5 | baseline_b5_sampling_v1 | 4119701 | 4124609 | epoch 2，进程存活 |

上述为当时的启动检查，不是本轮选型结果；后续已等待全部 12 次训练完成再作阶段判断。

启动后的记录工具补充：`export_baseline_validation.py` 同时导出已保存的
`training_sampling` metadata，保留抽样方式、目标窗口数和期望抽中比例；仅使用
validation 专用指标文件。相关导出与抽样测试 14/14 通过。该改动不影响 trainer，
也没有同步到本轮仍在运行的执行目录。

### 完整结果与复核

2026-09-06 23:29 HKT 已确认 B4 6/6；2026-09-07 00:07 HKT 已确认 B5 6/6，
两模型均输出完成标志，两个训练 Python 进程已结束。12 次均使用 `009fe42`，
没有重启已完成候选或在中途改参数。全部分数是 validation 的 seed42/43 均值。

| 模型 | 抽样方案 | P | R | F1 ± std | upcoming R | will AP |
|---|---|---:|---:|---:|---:|---:|
| B4 | uniform_control | 0.4835 | 0.2155 | 0.2974 ± 0.0007 | 0.0026 | 0.2230 |
| B4 | event4 | 0.3702 | 0.2340 | 0.2866 ± 0.0027 | 0.0053 | 0.1931 |
| B4 | upcoming4 | 0.2363 | 0.2189 | 0.2230 ± 0.0230 | 0.0608 | 0.1537 |
| B5 | uniform_control | 0.4876 | 0.2323 | 0.3142 ± 0.0051 | 0.0159 | 0.2500 |
| B5 | event4 | 0.4260 | 0.2374 | 0.3033 ± 0.0013 | 0.0238 | 0.2201 |
| B5 | upcoming4 | 0.2700 | 0.1953 | 0.2144 ± 0.0027 | 0.0476 | 0.1443 |

均未通过 P/R 双门。既定 mean-minus-std 排名和 F1 均值均选择控制组：event4 略增
召回但降低 precision，F1 分别下降 0.0107/0.0109；upcoming4 提前召回增加，但总体
precision 和 F1 显著下降，不把局部指标上升写成全面改善。

四个 uniform_control 的 station_report 与相同 seed 的原 history 逐字段完全相同。
训练 metadata 实测：13813 个训练窗口中，any_event 目标窗口 1645 个，权重 4 时
期望抽中比例约 0.3508；upcoming 目标窗口 984 个，期望比例约 0.2348。每 epoch
仍抽取 13813 次，不作 importance correction。12 次实际累计 273 epoch，运行时长
合计 6520.61 秒；这是进程用时相加，不是并发任务的墙钟耗时或独占性能测试。

只读复核复用 `baseline_fit_audit_v1`，2026-09-07 00:13:41 HKT 以 status=0 结束。
保持四个选中权重、X、目标、mask 和保存阈值不变，用 `c5249b5` 的审计工具复现原
保存分数，再仅替换历史 hot 的计算范围，未训练、未选新阈值、未评估 test。

| 模型 / seed | 原 F1 | 仅用已观察前缀的 F1 | 变化 |
|---|---:|---:|---:|
| B4 / 42 | 0.2981 | 0.2885 | -0.0096 |
| B4 / 43 | 0.2966 | 0.2607 | -0.0360 |
| B5 / 42 | 0.3091 | 0.3091 | 0.0000 |
| B5 / 43 | 0.3193 | 0.3060 | -0.0133 |

这再次说明历史字段并非完全无影响；它是共同协议待修问题，不据此把新旧结果混排。
完整 12 次配置/指标/抽样记录、两组开训协议与选型、四次控制复现和四权重复评已合并
归档至现有 `doc/experiments/baseline_event_sampling_20260907.json`，提交 `4be69cd`。
没有按 commit 新建源码目录，也没有删除或覆盖旧训练结果。

## 22. 后续计划与停止条件

### 当前决策

先完成本报告和计划，不启动新训练。B4/B5 保留第 0 节的当前综合方案和 B5 高精度
工作点；不再扩大已连续无净收益的正负例权重、抽样倍率或全模型续训搜索。
B2/B3 保留当前结果，不为让排名更好看而更改指标或刻意削弱/增强某个模型。

当前证据足以停止这些已测试方向，但还不能认定整个模型家族达到方法上限。若继续
优化，应检验一个明确的新假设，并设定有限预算；不以“必须接近主模型 precision”
作为停止条件，也不承诺 baseline 一定能达到该数值。

### 第一步：先锁定正式对比口径

与 tyx 确认最终采用哪一份数据和哪一版任务，取得其实际训练配置、可严格加载的
checkpoint 及对应完整源码，而不是只凭文档数字决定实验。明确冻结：

- episode 身份和 split、30/15 历史/未来窗口、节点集、观测特征及仅 train 拟合的归一化。
- hot/event 连续门槛、ongoing 的最短未来持续时间、upcoming 最晚起点、有效性 mask。
- 历史状态只使用截至锚点的观测；共同修正完整 episode 平滑造成的 lookahead，
  不只给 baseline 单方面改规则。字段修正通常不需重采已有 raw，但须重新构建受影响张量。
- 报警阈值的候选表、validation 选型方法、P/R 门槛、时间容差及解码规则。B2 与神经
  模型目前的阈值搜索范围不同，正式比较前统一或预先给出有依据的区别。
- 主实验续训父权重、阶段时长及累计训练/调参预算；各模型可用适合自己的学习率、
  batch、正则化和训练上限，但信息、目标、评估和可获得的调参预算必须透明。

验收产物：一份冻结协议、逐 episode/锚点输入-标签-mask 审计、主模型严格加载与
保存分数复现。若采用新 dense_i1，全部模型都在新协议下重建/重训；旧 v5 仅作为开发
记录，不把换任务后的涨分计成结构改进。若暂时缺少主实验产物，则先停留在当前开发
结论，不盲目重采，也不发布貌似同口径的最终排名。

### 第二步：B4/B5 最后一个有限结构对照

口径固定后，先在共同版本复现现有对照，再考虑事件判别的类别混合问题。预案是：
保持 GCN-GRU / GAT-GRU 编码器和其他任务头，把单一 will 二分类头与
`none / ongoing / upcoming` 三分类事件头对照；事件概率由两个事件类别的概率之和
得到，继续输出同一 will/start/duration 契约。该方案尚未实现、尚无收益证据。

ongoing/upcoming 只用训练目标生成监督，推理不能用真实未来类别或完整序列的
hist_last_hot 选择分支；不增加主实验专用 cluster、传播或 prefix 模块，不把新变体
冒称为原论文的精确复现。新头必须清楚命名为实验变体，同时保留标准头结果，不能
静默替换已经报告的 B4/B5。先核对解码、loss 和梯度，再做数据/随机种子可控的消融。

预算建议：每模型仅“现有头 / 三分类头”两组，各 seed42/43，合计 8 次；沿用已披露
的模型专属训练预算和共同评估。完整结束后比较 P/R/F1、upcoming recall、AP 与误报
构成。若收益只出现在一颗 seed、或靠召回塌缩换 precision，保留原方案；不再扩展
成无边界架构搜索。若双方确认新主实验协议不需要这项结构对照，则直接进入配置确认。

### 第三步：冻结后做新的初始化确认

选型仅使用预先规定的 train/validation，冻结后不再追着结果改配置。建议预先指定
新的 seed44/45/46，对 B2--B5 及主实验按相同协议运行，报告每颗 seed 与均值/标准差；
开发用 seed42/43 的选型成绩单列，不把新 seed 当成新独立 episode。

主实验采用续训时须明确是同 seed 父阶段加子阶段，并计入完整预算；baseline 的
续训也保留阶段 0 对照。单模型的 epoch 最优值、多个 seed 的均值和阈值搜索最优值
分别标注，不能相互替代。

### 第四步：独立评估与最终报告

正式泛化结论还需共同保留未用于调参的 episode。旧 test 已被查看，不能因重新命名
或重新划分就宣称再次成为未接触 holdout；若没有新独立数据，明确只报告开发期比较。
冻结后只评估，不根据该结果回头改模型。按 episode 做不确定性分析，不能把相关滑窗
当成独立样本来压低置信区间。

最终表同时列 report P/R/F1、ongoing/upcoming recall、报警/命中数、who、will AP、
hot 和事件 IoU 指标，以及有有效匹配样本时的时间/时长误差。说明误差单位和有效样本数，
不能把无匹配时输出的 0 MAE 解释为预测准确。附代码版本、数据哈希、配置、seed、
父权重及完整搜索预算，同时保留无收益的结果。

若最后有限结构对照及新 seed 确认仍无稳定提升，结论写为“在本数据、任务定义、
模型家族和披露预算下观察到的性能限制”，停止调参并交付完整报告；不写成普遍理论
上限，不隐藏与主模型的差距，也不为追平分数改动评价规则。

## 23. 本轮收尾判断：模型限制还是训练限制

用户决定先收尾，白天审查报告和后续计划。本节给出基于现有实测的判断，不为结束
工作而把“尚未找到更好方案”改写成“已经证明模型上限”。第 22 节仍是待执行计划，
并未启动其中的三分类头实验、新 seed 确认或正式 test 评估。

| 假设 | 已有证据 | 当前判断 |
|---|---|---|
| 主要因为训练轮数不够 | 延长从头训练、低学习率续训完整对照已结束；B4 只在一颗 seed 上小幅改善，B5 保留阶段 0 | 不支持继续仅靠加 epoch 解决问题 |
| 主要因为正负例权重或窗口抽样不合适 | 增加 upcoming 权重、短热负例权重、事件/提前事件抽样均完成多 seed 对照 | 所测训练策略已出现平台期；不能据此声称所有优化器/参数均已穷尽 |
| 模型完全学不到提前信号 | train upcoming AUC 约 0.922--0.943，AP 约 0.069--0.107；该子集正例率约 0.0059 | 已学到部分排序信号，不是完全没有学习能力 |
| 困难负例判别和泛化不足 | upcoming 的验证 AP 降至约 0.021--0.033；短热负例验证 AUC 约 0.487--0.563，低于训练约 0.650--0.711 | 有较直接证据支持当前事件判别与泛化存在瓶颈 |
| 单一事件头混合 ongoing/upcoming 导致性能受限 | 分层诊断显示两类表现不同，但尚无三分类头的受控对照 | 合理的下一步假设，不是已证实原因 |
| GCN/GAT 架构本身已到极限 | 当前未证明所有合理事件头/训练配置都无收益；共同历史字段及新主实验协议仍有待处理事项 | 不能作此结论，也不能将与主模型的全部差距归因于架构 |

**倾向判断：不是单纯“没训够”，而是当前任务与表示下的事件持续性判别、提前事件
高分区质量和泛化共同受限。训练方法的已测方向没有继续带来稳定收益，但模型本体的
极限尚未被证明。**因此先停下无边界调参是合理的，停下不等于承认某模型永远只能达到
当前分数，也不等于已经取得可靠的正式主实验优劣结论。

2026-09-07 收尾前再次 fetch 的 dev_tyx 仍为 `20c40e2`；服务器已知主入口实际指向
`/home/sci/work/isaac_factory`，源码仍为 `f322dbf`。在该入口下再次检查的
`raw_data/dense_i1`、`libcity/cache/model_cache/dense_i1_a1_prefix8` 和
`libcity/cache/model_cache/dense_i1_a1_f180` 均不存在。这只是已知路径的检查，
不宣称其他机器或其他目录没有这些产物；不为找不到文件而修改主实验仓库。

白天审查时优先决定：正式使用旧 v5 还是新 dense_i1、取得主实验真实产物与冻结
配置、是否值得执行第 22 节唯一的有限结构对照。之后才考虑配置冻结和独立确认。
当前训练/审计结果和报告全部保留在既有目录，旧目录不清理，不再启动新的训练任务。
收尾实查三个已知会话 `baseline_b4_sampling_v1`、`baseline_b5_sampling_v1`、
`baseline_fit_audit_v1` 均为 `dead=1, status=0`，相应 Python PID 均已退出。

## 24. 新一轮 dense 口径与数据扩充（2026-09-11）

### 24.1 已核对与已实现

最新 fetch 仍为 `origin/dev_tyx@20c40e2`。本地 `dev_xwt` 已按其训练入口和 dense 配置
统一 ongoing 最短 1、upcoming 最短 8 且起点不晚于 2、起点容差 3、hot 阈值 0.55、
事件阈值搜索及 checkpoint P>=0.80/R>=0.70。张量契约升级 v6，旧数据必须重建。
这属于任务/分母更新，不是简单降低报警阈值；新 F1 的增幅不能全部归因于模型提升。

完整说明见 `doc/implement/Baseline评估指标对齐说明.md`。start/duration MAE 统计所有
who TP；remain_len_mae 统计到全部作业完成的剩余时间，分开保存秒、分钟、支持样本数。
零匹配时显式单位 MAE 为 null；不拿上游未实际计算的 score_mae=0 作横向对照。
全 episode 平滑历史标签可能含未来依赖，已明确记录为共同离线契约的限制。

### 24.2 实际 raw 审计

只读检查主 raw 根目录：
`/home/sci/work/BNPDFormer/_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset`。
该仓库入口指向 `/home/sci/work/isaac_factory`；本轮没有修改其代码、raw 或 derived。

| 新增分组 | 检查数 | 通过数 |
|---|---:|---:|
| 二维：human+log、human+mach、human+mat、mach+log、mach+mat、mat+log | 30 | 30 |
| 三维：human+log+mat、mach+human+log、mach+human+mat、mach+log+mat | 20 | 20 |
| 四维：four_dim | 5 | 5 |
| 无扰动：norm20 | 20 | 19 |
| 合计 | 75 | 74 |

norm20 被拒一例含未完成作业、开放任务与死锁重置。选择只根据 raw 质量，不根据正例数
或模型成绩。原 134 加通过的 74 预计为 208；完整重审计仍可能拒绝 raw 已变动的案例。
保留原 92/17/25 的物理 episode 归属，只划分新增数据，预计最终 138/30/40。

**这不是主文档记录的 204 个 episode 或 140/28/36 split。** 目前未取得主文档对应的
精确导出包和 raw 清单。新的共同导出包可供主模型复用，但双方须使用同一 bundle/split
重新评估才能正式比较。此前已查看的 test 不会因扩容或改名变成新的未接触 holdout。

### 24.3 原目录执行流程

新增 `batch_factory_baseline_dense.sh` 记录主要命令，显式检查服务器仓库名和 dev_xwt。
默认复用 `factory_pdformer_134_v3`（目录名保留，实际张量版本以 manifest 为准）。

- `audit` 只重审计并保存独立 JSON；`build` 重聚合、导出、建张量并逐样本核对。
- 在内存聚合，不向主仓库写 derived。旧数据先归档为经 SHA-256 复核的 ZIP 文件。
- `B4` / `B5` / `ALL` 复用已经存在的 representation/history/seed42、seed43 目录；
  不存在则报错，绝不新建。旧权重和指标先归档为 ZIP，保留独立运行记录 JSON。
- `ARCHIVE_TAG` 必须唯一，audit 与 build 用不同 tag；已有归档不覆盖。

第一步先跑 **fresh initialization、history readout、均匀训练抽样** 的 B4/B5 控制组，
各 seed42/43；只读 validation 选 checkpoint/阈值，不输出 test 指标，不续用旧 v5 权重。
B4 使用 LR 3e-4、batch24、min10/patience10；B5 使用 LR 1.5e-4、batch16、min15/patience20；
共同 max60、weight_decay0.01、dropout0.2，结构和多任务头保持不变。
这是数据/口径迁移的控制实验，不宣称该配置已是新任务最优值。

控制组完成后优先核对真实事件数、误报、ongoing/upcoming recall、AP、MAE 支持数和训练曲线。
随后只对诊断支持的瓶颈做有限、同预算验证集对照；没有稳定多 seed 增益时停止并讨论。
不为达到主模型某一分数而挑数据、看 test 调参或隐性更换模型结构。

### 24.4 当前执行边界

本地已做与最新 upstream 100 组随机事件目标/报告指标逐项一致性检查，以及边界、归档、
保留 split、CSV/内存导出与基线张量对齐测试。用户解锁后已通过 UU/Leooo 的 SSH 重新
连接服务器，只在 BSTAN/dev_xwt 拉取 `5394034`。未修改或停止主仓库的仿真任务。

2026-09-11 20:18，重建进程正常退出 0：217 个 raw episode 中 208 个通过、9 个被拒，
split 为 138/30/40；36184 个窗口（23859/5439/6886），事件正例窗口 6274。
全部样本的输入、目标、mask、时间锚点和作业量均与共同导出一致，manifest 为 passed。
原数据已保留于 `dataset_before_dense20260911.zip`；没有新建实验目录或修改主 raw。

四次从头训练在 `baseline_dense_v6` tmux 会话顺序运行。目前 B4 seed42 完成，
best epoch2 / total12，validation P/R/F1=0.8023/0.6895/0.7417，ongoing R=0.7926，
upcoming R=0.0138；新规则下验证目标为 950 ongoing +145 upcoming。
start/duration MAE 为 0.0397/2.3020 分钟（755 个 who TP），remain_len MAE 28.798 分钟。
同时启动该已保存权重的只读验证集分层诊断，定位低 upcoming recall 的原因。
**这不是 v5 同任务上的翻倍提升，也未通过 R>=0.70 门槛。**
完整阶段记录保存在 `doc/experiments/baseline_dense_v6_20260911.json`，不读取 test 成绩。

### 24.5 首个控制组诊断与预注册下一项对照

B4 seed42 的独立诊断正常退出：ongoing vs negative AP=0.8425/AUC=0.9825；
upcoming vs negative AP=0.0064/AUC=0.7534。145 个 upcoming 目标的 will 分数中位数
约 0.0066，90 分位约 0.0483。阈值从 0.65 降到 0.30，仍有 143 个置信度漏报、
0 个起点漏报。因此不能把问题归因于起点容差或继续盲目降低阈值。

预注册一个单变量 `graph_context` 对照（执行前提更新见 24.6）：保留图编码器、last_mean、
loss、优化器、epoch 预算和 split，仅启用已有 `event_context` 投影。该投影使用同一输入
中的 masked 图级历史表示、可用 global features 和历史锚点的 jobs_remaining/jobs_total，
不增加未来标签或额外 raw 字段。每模型仍 seed42/43 从头训练，validation-only。
这检验“事件头只读单节点历史，缺少工厂阶段上下文”的假设，不预先声称一定有效。

执行为 `TRAIN_VARIANT=graph_context`，使用新的唯一 ARCHIVE_TAG；目录仍复用既有
candidate_history/seed42、seed43，真实变体以 config/profile/运行记录为准，不以目录名推断。
完成前不部署到正在执行的控制组；先保留各控制组完整指标，再归档权重运行该项对照。

### 24.6 四次控制完成与主实验身份核验优先

用户再次明确：基础配置与评估方式要对齐主实验，然后优化 baseline，不能只得到不同
任务上更高的分数。当前 208 包保留为探索/迁移控制，不标记为正式主实验可比结果。
`graph_context` 尚未在服务器启动；先取得主实验实际数据身份及冻结 split。

四次控制训练均完成，tmux `baseline_dense_v6` 为 dead=1 / exit=0。仅验证集：

| 模型 / seed | best / total epoch | report P | report R | report F1 | upcoming R | remain MAE (min) |
|---|---:|---:|---:|---:|---:|---:|
| B4 / 42 | 2 / 12 | 0.8023 | 0.6895 | 0.7417 | 0.0138 | 28.7980 |
| B4 / 43 | 5 / 15 | 0.8133 | 0.7041 | 0.7548 | 0.0069 | 15.2020 |
| B5 / 42 | 10 / 30 | 0.8378 | 0.6795 | 0.7504 | 0.0138 | 12.0084 |
| B5 / 43 | 5 / 25 | 0.8217 | 0.6858 | 0.7476 | 0.0138 | 14.0061 |

2026-09-11 再次 fetch，dev_tyx 仍为 20c40e2。通过 UU/Leooo 只读检查主目录中的
所有已发现 `episodes.npz` 和 `raw_data/*/meta.json`：`all` 为 202、`evt` 为 120、
`n10_i1_all_usable` 为 134、`n10_mix_ood` 为 55，另有 38--92 个 episode 的旧包；
未发现 `raw_data/dense_i1`，model_cache 中也没有 dense 命名目录。BNPDFormer 顶层
`experiments` 只有绘图脚本/README，没有发现目标导出包。已知主入口是指向
`/home/sci/work/isaac_factory` 的符号链接；不修改该仓库，不宣称其他机器没有数据。

正式比较需核验主实验的包哈希、逐 episode split、实际最终配置和续训来源。
若直接用主实验 204 包，就按其原划分重建 baseline 输入；若双方改用当前 208 包，
则主模型必须同包同 split 重跑，不拿文档 0.854 直接排名。学习率/模型容量/正则化不硬套
主模型；验证集搜索预算、训练量和初始化过程则必须披露。执行细则已记入
`doc/implement/Baseline评估指标对齐说明.md` 第 7 节。

## 25. B4/B5 与最新 BNPDFormer 结果快照（2026-09-11）

**本节是同评估定义下的阶段结果并列记录，不是同数据同 split 的受控排名。**
再次 fetch 确认 `origin/dev_tyx` 仍为 `20c40e230aedee6aef2429d352413fbcf0fa571a`。
主模型取该提交文档正式保留的 `dense_i1_a1_prefix8`，不取已作废的 v7、不混用 near5、
f180、uphist、remain 或 STGNPP 辅助实验的最佳单项。B4/B5 取第 24 节四个已完成控制组，
每次都是 validation 选定的最佳 checkpoint；本轮没有运行 baseline test。

### 25.1 共同评估定义

| 项目 | 当前 B4/B5 与主实验 dense 训练入口共同规则 |
|---|---|
| 历史 / 预测范围 | 30 分钟历史，未来 15 分钟；每窗口 60 秒 |
| 目标与有效性 | machine/workbench/gantry/AGV；human/buffer 仅作上下文；未观测未来不计 |
| Hot 标签 | ops 状态规则，整局最短 8 窗口，补 1 窗口间隔 |
| 每站事件 | 未来网格内最长连续 hot 段，等长取最早；每锚点每站一条 |
| Upcoming 事件 | 持续至少 8 个窗口，起点不晚于未来索引 2 |
| Ongoing 短尾 | 历史末窗已 hot 且未来继续，剩余至少 1 个窗口也计入 |
| Report 命中 | 预测和真值均有事件、工位正确，起点误差不超过 3 个窗口 |
| Precision | 命中的报告数 / 预测报告数 |
| Recall | 命中的报告数 / 真实事件目标数 |
| F1 | 2PR/(P+R)，不是简单平均 precision 和 recall |
| 起点 / 持续时间 MAE | 在所有 who TP（成功匹配工位的事件）上计算，不仅限 report TP |
| 剩余完工时间 MAE | 所有样本锚点到订单全部完成的剩余时间误差 |
| 阈值 / checkpoint | validation 扫阈值，优先 P>=0.80 内最高 F1；checkpoint 门 P>=0.80/R>=0.70 |

持续时间不参与 report TP 判定，误差单独用 MAE 衡量。所有下表 MAE 均为分钟。
ongoing/upcoming 指标分组按真实起点为 0 / 大于 0；历史末窗不 hot、但未来索引 0 起
足够长的事件，也在 ongoing 指标内。共同规则中的历史 hot 来自整局平滑标签，存在
前视依赖待审查，当前结果不宣称严格在线预测已验证。上述规则与实际模型的解码策略
分开记录：主模型有 prefix 门控等后处理，基线本次未启用主模型专属的置信度抬升。

### 25.2 数据与结果来源

| 项目 | B4 / B5 当前控制组 | BNPDFormer 正式保留档 |
|---|---|---|
| 数据 | v6 扩容包，208 episode | 文档 `raw_data/dense_i1`，204 episode |
| Episode train/validation/test | 138 / 30 / 40 | 140 / 28 / 36（文档记录） |
| 窗口 train/validation/test | 23859 / 5439 / 6886 | 20826 / 4331 / 5354（文档记录） |
| Validation ongoing/upcoming 目标数 | 950 / 145，共 1095 | 624 / 98，共 722 |
| 初始化与训练 | 每模型 seed42/43，从头训练 | seed42 配方，从 f180 接续训练，当前阶段 best epoch8 |
| 汇总方式 | 两颗 seed 最佳 checkpoint 的指标均值与标准差，同时保留逐次结果 | 单个最佳 checkpoint，不是多 seed 均值 |
| 数据证据 | 服务器 metrics + manifest，36184 样本导出核验通过 | Git 文档记录；对应 dense 包、权重和完整 metrics 尚未定位 |

真实事件数为相关窗口/节点上的事件目标，不是独立物理事件数量。主模型文档结果保留
其原始三位小数精度；基线均值对每颗 seed 等权计算，不拼接两次预测重新算 F1。
标准差用总体定义（ddof=0）描述这两次波动，不是置信区间，也不代表已有充分重复实验。

### 25.3 Validation 主指标

| 模型 / 统计口径 | Precision | Recall | Report F1 | Ongoing recall | Upcoming recall | P/R 门 |
|---|---:|---:|---:|---:|---:|---|
| B4 seed42，best epoch2 | 0.8023 | 0.6895 | 0.7417 | 0.7926 | 0.0138 | 未通过 |
| B4 seed43，best epoch5 | 0.8133 | 0.7041 | 0.7548 | 0.8105 | 0.0069 | 通过 |
| B4 均值 +/- 标准差 | 0.8078 +/- 0.0055 | 0.6968 +/- 0.0073 | 0.7482 +/- 0.0066 | 0.8016 +/- 0.0089 | 0.0103 +/- 0.0034 | 1/2 次通过 |
| B5 seed42，best epoch10 | 0.8378 | 0.6795 | 0.7504 | 0.7811 | 0.0138 | 未通过 |
| B5 seed43，best epoch5 | 0.8217 | 0.6858 | 0.7476 | 0.7884 | 0.0138 | 未通过 |
| B5 均值 +/- 标准差 | 0.8298 +/- 0.0081 | 0.6826 +/- 0.0032 | 0.7490 +/- 0.0014 | 0.7847 +/- 0.0037 | 0.0138 +/- 0.0000 | 0/2 次通过 |
| BNPDFormer prefix8，文档单 checkpoint | 约 0.883 | 约 0.881 | 约 0.882 | 约 0.920 | 约 0.633 | 文档值通过 |

B4/B5 使用各自 validation 选中的阈值：B4 为 0.65/0.55，B5 为 0.60/0.60。
主模型该 checkpoint 的最终实际阈值未取得，不把配置默认 0.70 当作已验证的选定阈值。
这些数值显示的差距是现有记录间的差距，不能在数据不同、主模型经历续训且仅单次报告
的情况下解释为骨干结构的净收益。主模型与 baseline 的 test 分数不放进本表混排。

### 25.4 Validation MAE 与支持数

| 模型 / seed | 起点 MAE | 持续时间 MAE | Who TP 支持数 | 剩余完工时间 MAE | 完工时间支持数 |
|---|---:|---:|---:|---:|---:|
| B4 / 42 | 0.0397 | 2.3020 | 755 | 28.7980 | 5439 |
| B4 / 43 | 0.0285 | 2.2885 | 771 | 15.2020 | 5439 |
| B4 两次指标均值 | 0.0341 | 2.2953 | 逐次如上，不合并 | 22.0000 | 每次 5439 |
| B5 / 42 | 0.0323 | 2.1999 | 744 | 12.0084 | 5439 |
| B5 / 43 | 0.0240 | 2.4233 | 751 | 14.0061 | 5439 |
| B5 两次指标均值 | 0.0281 | 2.3116 | 逐次如上，不合并 | 13.0072 | 每次 5439 |
| BNPDFormer prefix8 文档 | N/A | N/A | N/A | 仅给约 11.6--12.8；另处约 12.0 | 未取得指标文件核实 |

N/A 表示未取得该正式 checkpoint 的数值，不是 0。主模型旧 near5 的 start/duration MAE
不能补进 prefix8 行。剩余时间也不采用另一个 remain 辅助实验的 9.9。
B4 的完工 MAE 两次差异较大；B5 在当前同包控制组上更低。小起点 MAE 是 who TP 条件
误差，且命中主要来自起点 0 事件，不能据此声称 baseline 的提前预测比主模型准确。

### 25.5 Test 记录单列

| 模型 | P | R | F1 | Ongoing recall | Upcoming recall | 剩余完工 MAE (min) |
|---|---:|---:|---:|---:|---:|---:|
| B4 当前 v6 控制组 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |
| B5 当前 v6 控制组 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |
| BNPDFormer prefix8 文档 | 约 0.852 | 约 0.856 | 约 0.854 | 约 0.911 | 约 0.455 | 约 10.5 |

主模型 test 真值文档为 828 ongoing +112 upcoming。当前 baseline 不为了补表而打开
test，不将旧 v5 test 指标搬进新 v6。拿到共同包/划分并冻结模型选择后再做正式测试。

### 25.6 当前结论与追溯

- B4/B5 两次 precision 均超过 0.80；当前控制组的误报率已不再是旧结果约 0.5 precision
  的水平，但这不是旧任务上的直接提升证明。
- 在同一份当前 baseline 验证集上，B4/B5 平均 report F1 均约 0.75，二者非常接近；
  不凭两颗 seed 宣称其中一个有统计显著优势。
- 最需继续优化的是 upcoming：B4 仅命中 2/145、1/145，B5 两次均 2/145。
  主模型文档 upcoming recall=0.633，提示应重点核验并对照提前事件判别，而非只追总 F1。
- B4 完工时间预测波动明显；B5 相对稳定。两者均需在共同数据及正式多 seed 流程中确认。
- 缺少同一包和初始化/训练预算的统一审计，当前不能作为最终论文主模型胜出结论。

数据来源（固定 `dev_tyx@20c40e2`，不是依赖未来会移动的分支）：

1. `modelnote.md` 第 4 节及“当前 bestmodel”：正式模型、Val/Test P/R/F1、分层 recall 与计数。
2. `F1提升实验历程.md` 开头及第 4 节：数据规模与 split、prefix8 初始化和 best epoch。
3. `聚类_剩余时间_原因_STGNPP.md` 第 2.1 节：prefix8 完工时间 MAE 的近似 Val/Test 记录。
4. `无监督vs监督bottleneck_score对照.md` 第 7 节：正式无监督臂的 Test 指标及 remain MAE。
5. 本地 `baseline_dense_v6_20260911.json`：服务器四次 baseline 完整精度指标及 bundle 哈希；
   新增 `main_experiment_reference` 与 `comparison_snapshot` 保存本次文档摘录和汇总口径。

只更新现有 dev_xwt 报告和 JSON，不新建目录、不改训练参数、不触发训练或 test。

### 25.7 指标补全状态与当前可核验全表

上一版快照只列了主指标和总体 MAE，并非完整 metrics。2026-09-11 再次尝试读取服务器
原始 `metrics_validation.json` 时，UU 连续显示“无法连接至服务器，错误码：0”，无法进入
现有 SSH 终端。下表补齐已保存信息能核验的项目；未读出的项目明确列为待补，不写 0。
本次没有重训、没有重评 test，也没有执行新的诊断作业。

表中 B4/B5 是 seed42/43 两次指标等权均值，主模型是文档单 checkpoint；仍有 208/204
数据包差别。MAE 单位分钟，主模型源数值只有三位小数或近似范围。

| Validation 指标 | B4 均值 | B5 均值 | BNPDFormer prefix8 文档 |
|---|---:|---:|---:|
| Report precision | 0.8078 | 0.8298 | 约 0.883 |
| Report recall | 0.6968 | 0.6826 | 约 0.881 |
| Report F1 | 0.7482 | 0.7490 | 约 0.882 |
| Who precision（计数核验推导） | 0.8078 | 0.8298 | 未取得 |
| Who recall（计数核验推导） | 0.6968 | 0.6826 | 未取得 |
| Who F1（计数核验推导） | 0.7482 | 0.7490 | 未取得 |
| Ongoing recall | 0.8016 | 0.7847 | 约 0.920 |
| Upcoming recall | 0.0103 | 0.0138 | 约 0.633 |
| Event-will AP（所有事件） | 0.7579 | 0.7561 | 未取得 |
| 起点 MAE | 0.0341 | 0.0281 | 未取得 |
| 持续时间 MAE | 2.2953 | 2.3116 | 未取得 |
| 剩余完工时间 MAE | 22.0000 | 13.0072 | 仅约 11.6--12.8 |
| 通过 P>=0.80/R>=0.70 的次数 | 1/2 | 0/2 | 文档单次通过 |
| 验证样本数（每次） | 5439 | 5439 | 文档 4331 |
| 真实事件目标数（每次） | 1095 | 1095 | 文档 722 |
| Ongoing / upcoming 目标数（每次） | 950 / 145 | 950 / 145 | 文档 624 / 98 |

计数核验：用已保存的 recall 与 1095 真目标恢复 report TP，再由 precision 四舍五入恢复
整数预测数；who TP 使用已保存的 `time_mae_sample_count`。四次 who TP 恰好等于
report TP，所以这四次 who P/R/F1 与 report 相同。这是数学恢复，不冒充新读取的原始字段。
B4 seed42 的已存 precision 与 755/941 相差约 4.53e-8，需重新读取原始 JSON 排除抄录
误差；本次恢复计数采用 1e-6 绝对容差，其余三次相符，四位小数展示不受影响。

| 模型 / seed | 预测数 | Report TP / Who TP | Report FP / FN | Upcoming 命中 / 目标 | 起点门额外淘汰的 who TP |
|---|---:|---:|---:|---:|---:|
| B4 / 42 | 941 | 755 / 755 | 186 / 340 | 2 / 145 | 0 |
| B4 / 43 | 948 | 771 / 771 | 177 / 324 | 1 / 145 | 0 |
| B5 / 42 | 888 | 744 / 744 | 144 / 351 | 2 / 145 | 0 |
| B5 / 43 | 914 | 751 / 751 | 163 / 344 | 2 / 145 | 0 |

完整指标仍需从对应的 `metrics_validation.json` 补回以下组：

| 待补项目 | 原始位置 | 当前限制 |
|---|---|---|
| Hot P/R/F1/AP、真实/预测正例率、资源分项、type hmean | `remain` | 本地只有最佳 checkpoint 的事件摘要，不用训练中某一轮 hot 值替代 |
| Event-will ROC-AUC、概率分位数/最大值、原始计数 | `event_will` | AP 已保存；其他字段待远程读取 |
| IoU event P/R/F1 与匹配数 | `occupancy_event` | 不能用 report F1 代替；此附录仍按完整 8 窗口事件匹配 |
| Ongoing/upcoming 各自 start/duration MAE 与支持数 | `station_report` | 总体 MAE 不能反推出分组 MAE |
| 原因 accuracy/macro F1/macro recall、各类指标、train 多数类、混淆矩阵 | `cause`、`confusion_matrix_validation.csv` | 不用旧 v5 或主模型 test 数据补 baseline validation |
| 辅助 score MAE | `remain.score_mae` | 主无监督实验未实际计算该项，不能横比它打印的 0 |
| 验证总 loss 及各任务 loss、训练时间和参数量 | `loss`、`run_summary.json`、`config.json` | 这是训练诊断与成本，不替代主事件指标 |

### 25.8 Upcoming recall 低的已知证据与待验证原因

**已确认：** 不是百分比打印或均值计算错误，四次实际只命中 1--2/145；也不是“没有
事件头”，B4/B5 已有 event-will、start、duration 头。上述计数说明在各自选定阈值处没有
额外的起点容差损失，但不表示那些未报警的目标起点都预测正确。

B4 seed42 已保存的独立验证诊断提供更直接的证据：

- 145 个 upcoming 的事件概率中位数约 0.0066，90 分位约 0.0483，远低于阈值 0.65。
- 阈值降至 0.30，仍只有 2/145 命中，143 个置信度漏报、0 个起点漏报。
- Ongoing vs negative AP=0.8425，而 upcoming vs negative AP=0.0064；后者 AUC=0.7534，
  说明存在一些排序信号，但高置信度提前事件判别很差。分组 AP 属回顾性诊断，不是新评分规则。
- 总体 event AP 约 0.75 不能当作 upcoming AP。总 report 指标同样受 950/1095 个起点 0
  目标主导；仅识别这些目标就能取得较高总体分数。

**训练与结构上的合理假设，尚未由新 dense 对照证实：**

1. 单个 will 头同时处理“当前瓶颈延续”与“尚未开始”，模型可能优先拟合容易的延续信号。
   当前选择指标是总体 report F1，没有要求单独的 upcoming recall 下限；这与主实验共同规则
   一致，但并不保证每个模型会自然学好提前子任务。
2. 当前是迁移控制配置：从头训练、均匀采样、未启用事件头 graph_context；事件头只读
   GCN/GAT-GRU 编码后的节点历史向量，没有直接读取 pooled 全图和作业阶段上下文。
   图编码器本身已有邻居信息，不能说模型完全没有图上下文。
3. 代码已经给 upcoming 正例权重 4、ongoing 权重 3，并非完全没处理不平衡；但验证集
   950/145 比例不能代替训练集分布。仍需检查训练集各组数量、有效权重和分组拟合情况。
   Loss 中 ongoing 分组含 `hist_hot` 分支，而报告按 start==0 分组，需统计历史 hot 但未来
   事件 start>0 的案例是否存在及影响，不能未检查就把它定为根因。
4. 主模型有分阶段训练、prefix/事件融合等额外机制，和当前控制组不是只差 GCN/GAT
   骨干；文档较高 upcoming 值不证明任意一个头改动必然使 baseline 达到相同水平。

**下一次诊断顺序：** 恢复连接后先取完整指标，对 B4/B5 各 seed 都做独立概率/漏报分解，
并对比 train 与 validation 的 upcoming AP、概率分布和混淆。train 也差才优先定位目标、
权重或可学习性；train 好而 validation 差则优先定位泛化和 episode 分布。保持事件定义、
split 和评分不变，再按既有执行门做有限单变量对照，不先加 epoch 或盲目降低阈值。
B5 的同类低 recall 是已观察事实，但 B4 的具体概率分位数不能当成 B5 的实测结果。

### 25.9 远程恢复后的完整指标补录

2026-09-11 已重新通过 UU 的 Leooo 终端登录服务器。第 25.7 节的连接阻塞已解除；
四次控制组的完整 config、summary 和 validation metrics 保存到现有 benchmark 目录中的
`baseline_dense_control_metrics_20260911.json`（97305 字节）。本地同名日期快照记录其路径，
补充指标按终端六位小数提取；精确原值以服务器完整 JSON 为准。没有读取 test。
B4 seed42 的 precision 已重新核实为 `0.8023379383634431`，恰为 755/941；修正此前
抄录差异，四位小数展示不变。四次原始计数均确认 who TP 等于 report TP。

| Validation 补充指标 | B4 seed42 | B4 seed43 | B5 seed42 | B5 seed43 |
|---|---:|---:|---:|---:|
| Hot precision | 0.5858 | 0.6089 | 0.6759 | 0.6089 |
| Hot recall | 0.3323 | 0.3394 | 0.3072 | 0.3311 |
| Hot F1 | 0.4240 | 0.4358 | 0.4224 | 0.4290 |
| Hot AP | 0.3645 | 0.3807 | 0.3772 | 0.3726 |
| Hot type hmean | 0.4504 | 0.4620 | 0.4544 | 0.4556 |
| Event-will ROC-AUC | 0.9522 | 0.9596 | 0.9582 | 0.9613 |
| IoU event precision | 0.2891 | 0.2911 | 0.3154 | 0.2965 |
| IoU event recall | 0.3141 | 0.3187 | 0.3049 | 0.3129 |
| IoU event F1 | 0.3011 | 0.3043 | 0.3100 | 0.3045 |
| Cause accuracy | 0.8858 | 0.9149 | 0.9644 | 0.9531 |
| Cause macro F1 | 0.3371 | 0.3457 | 0.4374 | 0.3731 |
| Cause macro recall | 0.6969 | 0.6706 | 0.8637 | 0.7376 |
| Train 多数类 cause accuracy | 0.3096 | 0.3096 | 0.3096 | 0.3096 |
| 辅助 score MAE（不可与主模型零占位横比） | 0.3401 | 0.2779 | 0.2232 | 0.1375 |
| Validation total loss | 1.0116 | 0.9938 | 0.7360 | 0.7547 |
| Upcoming start MAE (min) | 0.5000 | 1.0000 | 0.5000 | 0.5000 |
| Upcoming duration MAE (min) | 5.7255 | 6.3232 | 5.5512 | 5.8194 |
| Upcoming MAE 支持数 | 2 | 1 | 2 | 2 |
| Ongoing start MAE (min) | 0.0385 | 0.0273 | 0.0310 | 0.0227 |
| Ongoing duration MAE (min) | 2.2929 | 2.2833 | 2.1909 | 2.4142 |
| Ongoing MAE 支持数 | 753 | 770 | 742 | 749 |

每次 event 有 1095 正例、67331 负例；hot 真正例率均约 0.013204。
IoU 附录仍按完整 8 窗口事件过滤，真目标数为 866，不能拿它的分母替换 report 的 1095。
Hot 四类资源 P/R/F1、预测正例率和 cause 分项 recall 保存在本地 JSON 的
`supplemental_validation_metrics`；完整任务 loss、配置和其他原始字段保留在服务器导出。
主模型对应完整 metrics 未取得，不以这些新读数填造主模型缺失项。Upcoming MAE 仅
1--2 个匹配支持，虽提示持续长度预测薄弱，但不能当作稳定的泛化估计。

## 26. Upcoming 定向优化（准备阶段）

用户要求先在当前数据上改进 B4/B5 的提前预测。当前冻结的 208 包用于探索性对照，
正式与主模型比较仍需第 25.2 节的数据身份核验。新实验不重建数据、不改 mask、事件
定义、阈值搜索或 checkpoint 门，不打开 test，也不为了达到某个数字调整评分标准。

### 26.1 修正与诊断

共享训练损失之前按历史末窗 hot 将部分未来 start>0 的重启事件归为 ongoing，且不计
其 upcoming start loss。现按有效正例的 `start==0` / `start>0` 分组，与评估子集一致。
这是训练监督分组修正，标签和评分不变；实际数据中的暴露数量及其对指标的影响尚待诊断。

`batch_factory_baseline_dense.sh diagnose` 在原 benchmark 目录输出 B4/B5、seed42/43
的 train/validation 独立诊断，检查分组数量、历史状态冲突、AP、概率分布和漏报类型。
先诊断现有权重再归档重训，避免失去旧模型的训练集拟合证据。不会用 train 的诊断值
选定推理阈值，验证阈值规则保持原样。

### 26.2 已登记的有限候选

| 配置名（profile 后缀 v2） | 相对修正后控制组的唯一变化 | 用途 |
|---|---|---|
| history_control | 无 | 从头重跑，隔离监督修正影响 |
| graph_context | 事件头读取 pooled 图与已有全局/作业历史上下文 | 检查局部表征是否不足 |
| upcoming_weighted | Upcoming 正例权重 4 改为 12 | 检查容易的 ongoing 是否主导梯度 |

三者保留 GCN-GRU / GAT-GRU 骨干，均匀采样、原学习率和每模型训练预算不变。不是同时
堆叠所有改动，也不保证三个都必须跑：先看现有模型诊断，再跑必要的单变量对照。
每轮复用已有 model 目录，先用带哈希核验的 ZIP 保存旧权重和指标；不新增目录。
每模型 seed42/43，仍由 validation 选 checkpoint；同时检查 upcoming AP/recall、总体
P/R/F1、ongoing、MAE 与支持数，不以精度崩溃换取表面提前召回。

本地相关测试 138 项及 12 个子测试通过，shell 语法检查通过。此节写入时修改尚未部署、
新诊断和候选训练尚未启动；完成状态与结果需在实际运行后另行补录。

### 26.3 已部署与实际诊断结果

2026-09-11 已提交并推送 `d0f7368`，服务器 `BSTAN_isaac_factory/dev_xwt` fast-forward
同步成功。八次 train/validation 诊断在原 `baseline_dense_diag` 会话完成，退出码 0；
文件为现有 benchmark 内的 `*_diagnostics_upcoming20260911.json`。旧权重先完成诊断，
再归档用于后续训练，未打开 test。

每次 train 为 4191 ongoing、595 upcoming、297152 negative；validation 为
950、145、67331。两个 split 的历史 hot 但未来 start>0 正例都为 **0**。因此前述分组
修正对当前冻结包不改变实际 loss 分组，不能把低召回归因于该 bug。跳过没有差异的
重复控制训练，旧控制组仍可作为当前候选的训练语义对照；未来数据出现此类重启时修正有效。

| 旧最佳 checkpoint 诊断 | B4 seed42 | B4 seed43 | B5 seed42 | B5 seed43 |
|---|---:|---:|---:|---:|
| Train upcoming AP | 0.0069 | 0.0100 | 0.0244 | 0.0090 |
| Validation upcoming AP | 0.0064 | 0.0086 | 0.0087 | 0.0077 |
| Train upcoming recall（使用保存的 validation 阈值） | 0.0000 | 0.0101 | 0.0420 | 0.0000 |
| Validation upcoming recall | 0.0138 | 0.0069 | 0.0138 | 0.0138 |
| Train upcoming 置信度漏报 / 起点漏报 | 595 / 0 | 589 / 0 | 570 / 0 | 595 / 0 |
| Validation upcoming 置信度漏报 / 起点漏报 | 143 / 0 | 144 / 0 | 143 / 0 | 143 / 0 |
| 整段旧训练历史最高 validation upcoming recall | 0.0345 | 0.0276 | 0.0345 | 0.0345 |

训练集提前目标也未充分学到，不能只解释为验证分布或过拟合。后期最高 upcoming recall
同样很低，说明仅换成更晚 checkpoint 不能解决当前问题。这仍不能单凭观察断言是
骨干容量限制；接下来区分事件头输入上下文和正例训练权重。

已在已有 `baseline_dense_v6` 会话启动 `TRAIN_VARIANT=graph_context`，依次运行
B4/B5 seed42/43，归档标签 `upcontext20260911`。写入时 B4 seed42 已到 epoch3，
其余按队列执行；还没有这轮最终结果，不把中途分数称作提升。旧控制参数量 B4=253470、
B5=266398，各次旧训练耗时约 433/535/1255/1060 秒，不能沿用早期 v5 的参数量。
保留原数据、原阈值/评分及训练预算，不额外启用加权候选，不新增目录。下一步先检查
完整上下文对照，未完成前不选择最终配置或运行 test。

## 27. Dense Upcoming 对照推进（2026-09-12）

### 27.1 上下文候选完成

重新 SSH 登录后确认 `baseline_dense_v6` 已退出，status=0，四份运行记录均为
`validation_completed`。完整 config/summary/metrics 保存于原 benchmark 目录的
`baseline_dense_context_metrics_20260912.json`（111938 字节，SHA-256 见本地 JSON）。
数据 manifest 哈希与原控制组相同，未评估 test；不是将训练中某个最好单项拼成一行。

| Validation 最佳 checkpoint | B4 seed42 | B4 seed43 | B5 seed42 | B5 seed43 |
|---|---:|---:|---:|---:|
| Best / total epoch | 3 / 13 | 6 / 16 | 7 / 27 | 5 / 25 |
| 选定阈值 | 0.65 | 0.55 | 0.55 | 0.60 |
| Precision | 0.8531 | 0.8440 | 0.8439 | 0.8084 |
| Recall | 0.6575 | 0.6621 | 0.6813 | 0.7014 |
| Report F1 | 0.7427 | 0.7421 | 0.7539 | 0.7511 |
| Ongoing recall | 0.7579 | 0.7611 | 0.7821 | 0.8063 |
| Upcoming recall | 0.0000 | 0.0138 | 0.0207 | 0.0138 |
| Upcoming 命中 / 目标 | 0 / 145 | 2 / 145 | 3 / 145 | 2 / 145 |
| 全事件 will AP | 0.7432 | 0.7550 | 0.7509 | 0.7609 |
| Hot F1 | 0.4292 | 0.4274 | 0.4253 | 0.4290 |
| 起点 MAE (min) | 0.0472 | 0.0276 | 0.0456 | 0.0260 |
| 持续时间 MAE (min) | 2.2739 | 2.2338 | 2.1115 | 2.3944 |
| 订单剩余时间 MAE (min) | 19.9698 | 14.6196 | 12.7886 | 14.5973 |
| Upcoming 起点 MAE (min) | N/A | 0.5000 | 0.6667 | 0.5000 |
| Upcoming 持续时间 MAE (min) | N/A | 4.2917 | 2.9871 | 4.9516 |
| Upcoming MAE 支持数 | 0 | 2 | 3 | 2 |
| P>=0.80/R>=0.70 | 未通过 | 未通过 | 未通过 | 通过 |

B4 均值 F1=0.7424、upcoming R=0.0069；B5 均值 F1=0.7525、upcoming R=0.0172。
相对原控制组分别约 -0.0059/+0.0035 的 F1 差异，且提前事件命中仍极少，不构成稳定
解决 upcoming 的证据。B4 seed42 的原始零 MAE 无匹配支持，表内明确为 N/A。
参数量 B4=286878、B5=299806；四次耗时约 455.7/557.5/1117.1/1035.9 秒。
保留结果而不把该候选提升为正式方案，也不与不同 cohort 的主模型分数作净收益比较。

### 27.2 正例权重单变量对照完成

复用已有会话和模型目录，启动 `TRAIN_VARIANT=upcoming_weighted`，归档标签
`upweight20260912`，代码 `516b71e`。相对原 history 控制组仅将 upcoming 正例权重
4 改为 12，event_context 仍关闭，其他 loss、采样、学习率、预算和评分规则不变。
四次已正常完成，tmux exit=0；完整结果已保存于原 benchmark 的
`baseline_dense_weighted_metrics_20260912.json`（114055 字节，哈希见本地 JSON）。
数据 manifest 未变，四份 metrics 均只有 validation，未打开 test。

| Validation 最佳 checkpoint | B4 seed42 | B4 seed43 | B5 seed42 | B5 seed43 |
|---|---:|---:|---:|---:|
| Best / total epoch | 1 / 11 | 5 / 15 | 3 / 23 | 5 / 25 |
| 选定阈值 | 0.75 | 0.60 | 0.55 | 0.55 |
| Precision | 0.8466 | 0.8373 | 0.8463 | 0.8084 |
| Recall | 0.6603 | 0.6813 | 0.6740 | 0.6895 |
| Report F1 | 0.7419 | 0.7513 | 0.7504 | 0.7442 |
| Ongoing recall | 0.7611 | 0.7832 | 0.7747 | 0.7926 |
| Upcoming recall | 0.0000 | 0.0138 | 0.0138 | 0.0138 |
| Upcoming 命中 / 目标 | 0 / 145 | 2 / 145 | 2 / 145 | 2 / 145 |
| 全事件 will AP | 0.7490 | 0.7669 | 0.7524 | 0.7604 |
| Hot F1 | 0.4296 | 0.4366 | 0.4229 | 0.4266 |
| 起点 MAE (min) | 0.0360 | 0.0322 | 0.0393 | 0.0238 |
| 持续时间 MAE (min) | 2.4424 | 2.3150 | 2.2467 | 2.3974 |
| 订单剩余时间 MAE (min) | 40.4087 | 14.7983 | 21.1399 | 14.1687 |
| Upcoming 起点 MAE (min) | N/A | 0.5000 | 0.5000 | 0.5000 |
| Upcoming 持续时间 MAE (min) | N/A | 3.1971 | 5.7311 | 5.8783 |
| Upcoming MAE 支持数 | 0 | 2 | 2 | 2 |
| P>=0.80/R>=0.70 | 未通过 | 未通过 | 未通过 | 未通过 |

B4 均值 F1=0.7466、upcoming R=0.0069；B5 均值 F1=0.7473、upcoming R=0.0138。
仍没有稳定的提前召回提升，且两次较早选中的 checkpoint 完工时间 MAE 较差。
中途 upcoming R=0.0414 不是所选 checkpoint 结果；也不拼接其他 epoch 的 MAE。
不提升为正式配置，不扩展多倍率搜索。本轮仅检验新 dense 标签和数据下一个已登记权重。

### 27.3 三分类事件头候选与归档诊断工具

依第 22 节已登记的假设，实现独立 `three_class` 候选，profile 为 `dense_three_class_v2`。
GCN-GRU / GAT-GRU 编码器、其他任务头和同一观察信息不变。事件分类输出
`none / ongoing / upcoming` 三类，正例按真实 start==0 / start>0 生成训练监督；
分类损失改为三类交叉熵，权重沿用 2/3/4，不同时叠加上下文、抽样或加权候选。
同 seed 下共享模块的初始参数与二分类臂完全相同，新增分类输出层在独立 RNG 作用域
初始化，避免改变其他预测头及后续训练随机数流；不从其他模型或评估集加载权重。
推理不读取真实类别，统一事件概率是预测 ongoing 与 upcoming 概率之和，其 logit 为
`logsumexp(event_logits) - none_logit`，继续由共同阈值和 report 函数评分。
原 start/duration、hot、cause 和 remain 目标与损失保留，不修改标签或评估分母。

这属于清楚标注的事件头消融，不冒称原论文精确复现；标准二分类头仍是有效对照，
不是旧版本兜底。三分类交叉熵比二分类 BCE 多了类别区分要求，不能直接以二者 total
loss 的大小判断谁更好。候选预算每模型 seed42/43、每次最多 60 epoch，仍按原共同
validation 选择规则；不保证一定改善，不在本轮加权训练尚未结束时部署训练模块。

`diagnose_baseline_events.py --archive_member best.pt` 可直接读取当前格式的归档文件，
先验证 ZIP manifest 内的 SHA-256，再在内存加载，不解压覆盖正在用的模型路径。
诊断记录源文件和权重成员哈希，仍只允许 train/validation。此工具用于核查已完成
候选的分组学习情况，不更改或筛选正式事件目标。

本地 149 项及 12 个子测试通过，包括两种骨干真实一轮训练/加载、三类边际概率、三类可分合成样本拟合、
CE 权重和 mask 梯度、共享初始化/RNG 一致、无有效目标、禁止未来标签参与推理，以及归档完整性。另对
提交前的二分类头做 10 组精确参数/前向/反向对照均一致。

### 27.4 三分类候选运行记录

确认权重对照全部完成并保存后，服务器 clean `dev_xwt` fast-forward 至 `33628fc`。
服务器额外运行 5 项无文件训练的三分类前置测试，全部通过；未修改主实验仓库或其进程。
复用 `baseline_dense_v6`，归档标签 `upclass20260912`，依次训练 B4 seed42/43、
B5 seed42/43。最新核验 B4 两次已 `validation_completed`（13 / 15 epoch），
B5 seed42 PID4006578 存活，seed43 排队。manifest 哈希与控制组一致。
旧权重在原模型目录先核验归档，不新增实验目录。
本轮还没有最终指标，不能填入结果表；保持同一 validation 选模规则和 test 封存。
目标尚未达到，后续需核查完整三分类结果与 train/validation 分组学习情况。

同步复用 `baseline_dense_diag`，以 CPU 两线程依次诊断上下文候选的四份归档权重，
每份读取 train 和 validation。来源为 `model_before_upweight20260912.zip` 内的
`best.pt`，输出标签 `upcontext20260912`；不解压、不覆盖当前训练路径、不读取 test。
八份均已完成，诊断会话 exit=0；逐份核验数据哈希、归档成员哈希、graph_context profile
和保存 epoch 均一致。每次 train/validation 样本数为 23859/5439，未打开 test。

| 上下文候选分层诊断 | B4 seed42 | B4 seed43 | B5 seed42 | B5 seed43 |
|---|---:|---:|---:|---:|
| Train upcoming AP | 0.007318 | 0.014483 | 0.011602 | 0.009032 |
| Validation upcoming AP | 0.006518 | 0.009206 | 0.006183 | 0.007868 |
| Train upcoming recall @保存的 validation 阈值 | 0.0000 | 0.0151 | 0.0151 | 0.0017 |
| Train 概率漏报 / 时间漏报 | 595 / 0 | 586 / 0 | 586 / 0 | 594 / 0 |
| Validation 概率漏报 / 时间漏报 | 145 / 0 | 143 / 0 | 142 / 0 | 143 / 0 |

分母分别为 595/145 个相关窗口站点目标。训练集也弱，不能仅归因于验证过拟合；
当前上下文候选没有解决提前事件分类。先完成已登记三分类对照，不扩展上下文组合搜索。

### 27.5 三分类诊断补充

诊断脚本在三分类模型存在对应输出时记录 `none/ongoing/upcoming` 的混淆矩阵、
各真实类别下预测概率均值、argmax recall 与支持数，以及 upcoming 类概率对负例的 AP。
此分析用于区分类别学习和总事件概率的表现，**不把类别 argmax 当成报警规则**。
原 report 仍使用两种事件类别概率之和，阈值、选模、事件分母和时间容差均未改变。
脚本验证三类分布归一化及边际事件概率一致；某类别没有支持样本时，其召回和概率均值明确为 N/A。

152 项及 15 个子测试通过，包括 B4/B5 实际训练、checkpoint 加载、完整诊断 CLI，
并核验诊断 report F1 与保存的 validation 值相同、无 test 输出。只增加诊断和测试，
不改训练模块。训练全部完成后才将服务器同步至 `79a5df6`；服务器另运行三项测试和
三个子测试，全部通过。复用 `baseline_dense_diag` 启动四份已保存权重各自的
train/validation 诊断，输出标签 `upclass20260912`。没有启动新一轮调参或 test。

### 27.6 三分类完整结果

四份运行记录均为 `validation_completed`；`baseline_dense_v6` exit=0，最后 PID4017331
已退出。完整记录保存为 `baseline_dense_three_class_metrics_20260912.json`（114448 字节，
哈希见本地 JSON）。四次均在 `33628fc` 训练，数据、split 和评测契约均与控制组相同。

| Validation 最佳 checkpoint | B4 seed42 | B4 seed43 | B5 seed42 | B5 seed43 |
|---|---:|---:|---:|---:|
| Best / total epoch | 3 / 13 | 5 / 15 | 7 / 27 | 5 / 25 |
| 选定阈值 | 0.60 | 0.60 | 0.60 | 0.65 |
| Precision | 0.8408 | 0.8185 | 0.8528 | 0.8389 |
| Recall | 0.6703 | 0.7041 | 0.6721 | 0.6849 |
| Report F1 | 0.7459 | 0.7570 | 0.7518 | 0.7541 |
| Ongoing recall | 0.7726 | 0.8105 | 0.7726 | 0.7874 |
| Upcoming recall | 0.0000 | 0.0069 | 0.0138 | 0.0138 |
| Upcoming 命中 / 目标 | 0 / 145 | 1 / 145 | 2 / 145 | 2 / 145 |
| 全事件 will AP | 0.7501 | 0.7696 | 0.7443 | 0.7616 |
| Hot F1 | 0.4273 | 0.4364 | 0.4209 | 0.4334 |
| 起点 MAE (min) | 0.0409 | 0.0272 | 0.0435 | 0.0267 |
| 持续时间 MAE (min) | 2.3118 | 2.3115 | 2.1612 | 2.4142 |
| 订单剩余时间 MAE (min) | 21.6723 | 14.9998 | 12.4754 | 14.6633 |
| Upcoming 起点 MAE (min) | N/A | 1.0000 | 0.5000 | 0.5000 |
| Upcoming 持续时间 MAE (min) | N/A | 6.2591 | 3.8732 | 5.8142 |
| Upcoming MAE 支持数 | 0 | 1 | 2 | 2 |
| P>=0.80/R>=0.70 | 未通过 | 通过 | 未通过 | 未通过 |

B4 均值 P/R/F1=0.8296/0.6872/0.7515、upcoming R=0.0034；
B5 均值 P/R/F1=0.8459/0.6785/0.7530、upcoming R=0.0138。
总体 F1 相对原控制组仅小幅变化，提前召回没有实质提升。没有证据把三分类变体提升为
正式方案，更不能宣称全部指标已合理；下一步是已启动的类别学习诊断，不再增加 epoch。

### 27.7 加权模型的分组诊断完成

复用原诊断会话，以 `model_before_upclass20260912.zip/best.pt` 为来源，完成四份加权
模型各自 train/validation 诊断，输出标签 `upweight20260912`。八份数据哈希、成员哈希、
weighted profile 和 epoch 均核验通过，tmux exit=0。未覆盖当前三分类权重，也未读取 test。

| 加权候选分层诊断 | B4 seed42 | B4 seed43 | B5 seed42 | B5 seed43 |
|---|---:|---:|---:|---:|
| Train upcoming AP | 0.006877 | 0.016216 | 0.008025 | 0.012320 |
| Validation upcoming AP | 0.006881 | 0.011589 | 0.007115 | 0.009009 |
| Train upcoming recall @保存的 validation 阈值 | 0.0000 | 0.0118 | 0.0000 | 0.0118 |
| Train 概率漏报 / 时间漏报 | 595 / 0 | 588 / 0 | 595 / 0 | 588 / 0 |
| Validation 概率漏报 / 时间漏报 | 145 / 0 | 143 / 0 | 143 / 0 | 143 / 0 |

提高权重没有让所选 checkpoint 在训练集上充分学会提前类别。保留这项负结果，不扩展
倍率搜索；不能仅凭总体 AP 约 0.75 断言提前判别良好。

### 27.8 最佳权重的三分类诊断完成

八份 `*_diagnostics_upclass20260912.json` 已完成并逐份核验 checkpoint SHA-256、数据
哈希、three_class profile 和最佳 epoch，tmux exit=0。以下两种 AP 均在 upcoming 对
无事件负例的同一事后子集计算，排除 ongoing；正式报告仍使用边际事件概率。

| 三分类最佳权重诊断 | B4 seed42 | B4 seed43 | B5 seed42 | B5 seed43 |
|---|---:|---:|---:|---:|
| Train upcoming 边际事件 AP | 0.007126 | 0.009866 | 0.009075 | 0.009469 |
| Validation upcoming 边际事件 AP | 0.005921 | 0.008852 | 0.006089 | 0.007939 |
| Train upcoming 类概率 AP | 0.019425 | 0.036356 | 0.053975 | 0.033744 |
| Validation upcoming 类概率 AP | 0.014145 | 0.032186 | 0.015404 | 0.016979 |
| Train upcoming 类 argmax recall | 0 | 0 | 0 | 0 |
| Validation upcoming 类 argmax recall | 0 | 0 | 0 | 0 |
| Train 真 upcoming 上的平均 none 概率 | 0.986401 | 0.939807 | 0.974438 | 0.967346 |
| Validation 真 upcoming 上的平均 none 概率 | 0.987023 | 0.941347 | 0.975877 | 0.965279 |

类别 argmax 诊断不是 report recall，也不参与正式阈值和选模。结果显示一定类别排序信号，
但当前最佳权重对真实提前事件主要预测“无事件”，不是单纯混淆为 ongoing。
训练集也如此，不能仅用验证过拟合解释；“拆开事件类别就能解决问题”的假设未获支持。

### 27.9 已有最后一轮权重诊断完成

不额外训练、不改选模规则，复用原诊断会话读取四份已存在的 `last.pt`，分别计算
train/validation 分组与三分类诊断，输出标签 `upclasslast20260912`。
该对照用于判断提前判别是否在后期才学到；各 AP 不依赖报警阈值，召回比较需区分
保存阈值与固定相同阈值，不能把后期的某个最好单项换进最佳 checkpoint 结果表。
八份均已完成，诊断 tmux exit=0。逐份核验 last.pt SHA-256、数据 manifest 哈希、
最终 epoch、split 和样本数；只读取 train/validation，未新训练或改选模。

| 三分类最后权重诊断 | B4 seed42 | B4 seed43 | B5 seed42 | B5 seed43 |
|---|---:|---:|---:|---:|
| Last epoch | 13 | 15 | 27 | 25 |
| Train upcoming 边际事件 AP | 0.110850 | 0.142374 | 0.296762 | 0.289588 |
| Validation upcoming 边际事件 AP | 0.010861 | 0.010878 | 0.008820 | 0.007475 |
| Train upcoming 类概率 AP | 0.226719 | 0.275094 | 0.446460 | 0.467045 |
| Validation upcoming 类概率 AP | 0.035218 | 0.024133 | 0.020829 | 0.011249 |

这些 AP 是 upcoming 对无事件负例的诊断值，不是全事件 AP 或 report F1。
后期训练集类别排序明显改善，而验证集没有同步改善。因此第 27.7/27.8 节的
“训练集也弱”只描述当时按共同规则选中的 best.pt，不能推广为模型根本学不会。
新增证据指向泛化差距，不能据此认为单纯增加 epoch 会解决问题；不将 last.pt 的
训练集单项指标替换进正式 best.pt 结果表。当前 upcoming 优化目标仍未达到。

### 27.10 主实验历史输入范围待对齐

只读核查 `origin/dev_tyx@20c40e2` 的 `PDFormer/factory_bn` 代码发现：

- `remain.py` 的 `PRECURSOR_FAR_WINDOWS=30`，`pack_precursor_features` 除近窗统计外，
  还汇总编码窗口之前最多 30 个窗口的四项均值和队列最大值。
- `dataset.py` 第 496 行起，`far=feats[far_lo:hist_start]` 与近窗一起构建
  `sample["precursor"]`；推理样本也有同样的历史截取。
- `model.py::_fuse_precursor` 将这些统计和邻接队列统计投影后加入事件表示；
  第 1260 行附近的事件头路径实际调用该函数。

所以当前主模型代码不只是 30 分钟历史输入，还可能使用更早 30 分钟的历史摘要；
这些是过去信息，不因此构成未来泄漏。当前 baseline v6 没有这部分远历史摘要，
之前“基础历史输入完全相同”的表述需要收窄为“名义编码长度相同”。
这不能证明已公布主实验 checkpoint 实际使用了非零摘要权重，也不能单独解释分差。
目前尚缺对应正式数据包和权重的完整同口径核验，204/208 episode 差异也仍存在。

后续需显式固定共同可用信息范围，再做独立输入消融；不能悄悄扩展 baseline 的
历史窗口后仍沿用原数据契约，也不应机械照搬主模型的专用解码器。此轮只记录证据，
未修改数据、标签、split、模型输入或主实验仓库。

### 27.11 新对话恢复与全量历史输入审计

2026-09-12 按交接指南通过 UU 的 Leo 终端 SSH 恢复。BSTAN 起始为干净
`dev_xwt@6b2b216`；`baseline_dense_v6` 和 `baseline_dense_diag` 均 dead=1、exit=0，
未发现对应训练/诊断进程。原 v6 manifest SHA-256 与 27.10 前一致。
主仓库仍在旧 `f322dbf`；只读查询的 `PDFormer/raw_data` 和
`PDFormer/libcity/cache/model_cache` 中没有 `dense*` 目录，现有 n10 权重不能替代
文档的 dense checkpoint。主路径实际解析到 `/home/sci/work/isaac_factory`，两条路径
均作为主实验只读。本地与 BSTAN 只更新 Git 远端对象，未 checkout/merge 主分支；
最新引用仍为 `20c40e230aedee6aef2429d352413fbcf0fa571a`。

新增审计工具 `tools/audit_baseline_precursor_history.py`，代码提交 `2d6bbd0`。
它从固定主分支 Git 对象加载 `remain.py`，记录 remain/dataset/model 源码哈希，
只读取现有共同导出中 train/validation episode 的特征。首次服务器预检发现缺少该
主分支 Git 对象，补充 fetch 后各 3 项测试通过；没有重装依赖、切换分支或运行训练。

令 t 为首个未来窗口，编码输入为 `[t-30,t)`；远历史为 `[max(0,t-60),t-30)`。
主源码的 23 维 precursor 包含最近 5 窗的 12 项末值、5 项均值和 1 项队列变化，
以及远历史的 4 项均值（队列、阻塞、入站等待、缺料）和队列最大值。主事件头还拼接
1 维邻接队列统计。远历史不足 30 窗时用实际已发生部分，没有远历史时补零。
投影最后一层权重和偏置为零初始化，因此源码有该路径不证明正式 checkpoint 有效使用它。

| 现有 v6 输入审计 | Train | Validation |
|---|---:|---:|
| Episode | 138 | 30 |
| 样本 | 23859 | 5439 |
| 无编码窗之前历史 | 138 | 30 |
| 具有部分额外历史（1--29 窗） | 4002 | 870 |
| 具有完整额外 30 窗历史 | 19719 | 4539 |
| 远历史摘要至少一项非零的样本 | 23721 | 5409 |

所有被审计样本均通过 split/episode/node/锚点身份检查及分钟窗口连续性检查；
近窗-only 和近窗+远历史的前 18 维逐元素相同，近窗-only 后 5 维严格为零；
截断 t 及其后的全部未来观测不改变摘要。这里“非零”按所有 38 个节点统计，包含
不适用节点，既不是有效事件目标计数，也不证明这些特征对 upcoming 有预测价值。
审计不读取标签或 test episode 特征数组，不训练、不选阈值、不产生 AP/recall/MAE。

完整证据位于原 benchmark 的 `baseline_precursor_history_audit_20260912.json`，
SHA-256 为 `3308c7bb55856d70dff79e1d9185afde7faf8c3ebde9d2be1be966dcd71319b6`。
导出包哈希也与原 v6 manifest 的全样本对齐记录一致。原数据、split、历史 hot 定义、
权重和已有结果均保留。

下一项建议是有相同容量的“近窗摘要 / 近窗+远历史摘要”输入对照：两个分支均保留
GCN/GAT-GRU、相同 23 维摘要投影和初始化，控制分支最后 5 维归零，扩展分支填入
上述 5 项远历史摘要。只改变远历史是否可用，不能直接用新投影分支与原 history
模型的差异归因于远历史。拟使用二分类原控制配置、每模型 seed42/43、最多 60 epoch，
共 8 次；选模、阈值、事件规则和 train/validation 诊断口径保持原样。不采用主模型
专用 decoder，也不把三分类后期训练分数拼入新结果。

该对照目前仅为预案，未开训；任何新增输入需要单独命名特征契约并记录原 v6 来源哈希。
执行前仍须取得主模型正式 204-episode 导出包、实际 split、运行配置和 checkpoint，
核验摘要实际使用及共同 cohort/历史范围。若共同范围确定为纯 30 分钟，则只讨论该范围
内的表示对照，并要求主模型也遵守此范围；不会单方面改动主仓库。当前没有共同数据的
正式新成绩，不能把输入可用性结论解释为已解决泛化差距。

## 28. 固定 208 episode 的 B4/B5 upcoming 优化续接

用户已明确：先沿用 baseline 的 208 个 episode，优先查清低 upcoming recall 的原因并
改善 B4/B5，保留与主实验对比的价值。因此正式 204-episode 主包暂不再作为本轮开发
实验的开训前置；它仍是最终主模型公平重评所需材料，不能将本轮成绩冒称该对照。
本轮初始 BSTAN 为干净 `dev_xwt@7c0c0ca`，dense 训练及诊断会话均 dead=1、exit=0，
没有匹配训练/诊断进程。继续复用原目录和 v6 manifest，不重建数据，不用 test 调参。

### 28.1 第一项受控假设：近窗起始征兆是否在表示中丢失

新 `near_precursor` 只把原编码范围内最近 5 个窗口的 18 项末值/均值/队列变化摘要，
通过零输出初始化的两层投影加到事件表示。它们均来自已经审核的原 bundle 历史特征，
不提供额外历史、未来、scenario/run ID、历史 hot 或标签。为保留后续等容量远历史
对照的条件，输入固定为 23 维，后 5 维本轮严格为零。未来是否试 far arm 另据结果决定。

保留两层 GCN/GAT 和 GRU、二分类事件头、原损失权重、均匀抽样、AdamW、学习率、
dropout、早停、阈值列表和 report 选模规则。新增参数全部明确记录；与原 history 的
差异检验“显式近窗摘要路径”整体效应，不声称单独证明更远历史的收益，也不复制主模型
专用解码。零输出初始化并保留 RNG 流使首次 forward 与原控制组严格相同。

预注册首轮 B4/B5 各 seed42/43，共 4 次、每次最多 60 epoch；仍使用 B4 的
min_epochs=10/patience=10，B5 的 min_epochs=15/patience=20。输出复用原
`candidate_history/seed42,43`，唯一归档标签 `nearprec20260912`，原三分类权重先逐文件
哈希核验归档，历史 ZIP 和完整 metrics 导出保留。

输入契约单独命名 `factory_baseline_precursor_v1` / mode=near，并在 checkpoint 保存
原 v6 manifest、bundle、split、样本索引和特征构建源码哈希。只在内存给 train/validation
样本附加摘要，未构建 split 的访问会报错。原 v6 数据文件与标签契约保持原样。

本轮判读必须同时查看每 seed 的总 P/R/F1、upcoming recall、train/validation upcoming AP、
误报组成和时间 MAE；不得仅靠挑某 epoch 的 upcoming 单项、降阈值或拼接多个 checkpoint
宣布成功。若出现稳定提升，继续确认跨 seed/episode 支持；若没有，转向事件可预测性、
独立事件支持数和泛化分布诊断，不能以单个近窗试验失败宣称 GCN/GAT-GRU 已到理论上限。

### 28.2 近窗摘要开训与独立事件支持诊断准备

近窗代码提交为 `e5f6cb4`，本地 79 tests / 14 subtests、服务器 4 项候选预检通过。
四个旧权重目录均确认是三分类候选，且 `nearprec20260912` 归档标签未占用。
复用 `baseline_dense_v6` 开训，初始 pane PID=45693，首个训练 PID=45709（B4 seed42）；
这些 PID 仅是启动记录，后续必须重新核验。日志在原 benchmark 的
`nearprec20260912_train.log`。已观察到 epoch 1--11 正常推进，不能把某一中间 epoch
单项当作最终结果。训练模块在整批过程中固定，不切换分支或 pull 新训练代码。

另准备 `diagnose_dense_event_support.py`，只读原包 train/validation，按 episode、工位和
绝对起点去重 upcoming 支持；同时区分起点 0 且历史冷的正例，以及已有扰动/未来才出现
扰动的回顾性分组。全量目标必须严格重现 train 的 4191/595/297152 与 validation 的
950/145/67331，否则拒绝输出。2 项合成测试通过；零局部征兆不等于不可预测，同局事件
也不能直接视为统计独立。该独立诊断可从已测试的 Git 对象流式执行，无需更新运行中的
训练模块或建立源码副本目录，并单独记录诊断源码提交与服务器工作区提交。

### 28.3 独立事件支持诊断完成，B4 近窗结果已确认

`dense_event_support20260912.json` 正常生成，诊断代码为 `bdbac4a`，训练工作区仍固定
`e5f6cb4`。结果 SHA-256 为
`56d2dc07169d12e420ed1d2877bcf3290d2cd53d0abdd317ea6581845b32cd93`；
manifest 与原 v6 相同，重建的全部目标数严格复现冻结记录。未读取 test 特征或选择阈值。

| 支持与回顾性条件 | Train | Validation |
|---|---:|---:|
| Episode / 有 upcoming 的 episode | 138 / 113 | 30 / 27 |
| Upcoming 窗口-工位目标 | 595 | 145 |
| 按 episode、工位、绝对起点去重的 upcoming 起始事件 | 299 | 73 |
| Upcoming 锚点已有本地扰动 | 36 | 8 |
| Upcoming 锚点任一节点已有扰动 | 198 | 58 |
| Upcoming 锚点本地无扰动、未来至起点才观测到本地扰动 | 300 | 82 |
| Upcoming 最近 5 窗的 9 项选定本地征兆通道全零 | 88 | 32 |
| 正起始事件 start=0、但历史末窗 cold | 304 | 73 |

除去重起始事件一行外，条件计数均按窗口-工位目标；最后一行属于 start=0 分组。
条件可以重叠，不能相加当作互斥事件类别。同一 episode 内去重事件仍可能相关。
未来才出现本地扰动的 upcoming 目标占 train 50.4%、validation 56.6%，提示应审查
可观测提前信号与扰动生成机制。零选定本地通道不代表完整图历史没有信号，未来出现
扰动也不证明其发生时间完全不可预测；这些统计不能充当模型理论召回上限。

B4 两颗 seed 已正常完成，读取最终保存 metrics 与 completed 运行记录：

| B4 近窗摘要 | seed42 | seed43 |
|---|---:|---:|
| Best / total epoch | 2 / 12 | 5 / 15 |
| Report P | 0.8109560 | 0.8331504 |
| Report R | 0.6894977 | 0.6931507 |
| Report F1 | 0.7453110 | 0.7567298 |
| Upcoming 命中 / 目标 | 2 / 145 | 2 / 145 |
| Upcoming recall | 0.0137931 | 0.0137931 |
| Validation 选定阈值 | 0.60 | 0.60 |

F1 均值为 0.7510204，原 history 为 0.7482113；upcoming 均值从 0.0103448 到
0.0137931，仅 seed43 多命中一个目标，不能称为稳定、实质性改善。B5 seed42 已观察到
epoch3、训练会话仍存活。B4 训练集/验证集诊断复用 `baseline_dense_diag`，CPU 2 线程，
标签 `nearprec20260912`，启动后已核实实际 Python 进程；B5 尚未诊断。后续须重新查询
真实状态，不能把本段部分完成记录当作整批结束，也不能重复启动已完成训练。

### 28.4 B4 近窗最佳权重诊断完成

四份 `b4_seed{42,43}_{train,validation}_diagnostics_nearprec20260912.json` 已完成，
诊断会话退出 0；逐份核验当前 best.pt 的 SHA-256、epoch 和样本数。以下 AP 使用
二分类事件概率、只比较 upcoming 与无事件负例，不与三分类 subtype AP 混用。

| B4 近窗 best 诊断 | seed42 | seed43 |
|---|---:|---:|
| Train upcoming AP | 0.006839 | 0.010118 |
| Validation upcoming AP | 0.006336 | 0.008807 |
| Train upcoming AUC | 0.795668 | 0.859065 |
| Validation upcoming AUC | 0.750310 | 0.805106 |
| 保存阈值下 train upcoming 概率漏报 | 595 / 595 | 590 / 595 |
| 保存阈值下 validation upcoming 概率漏报 | 143 / 145 | 143 / 145 |
| 保存阈值下 validation upcoming 时间错位漏报 | 0 | 0 |
| 诊断阈值降至 0.30 时 report P | 0.701322 | 0.679012 |
| 诊断阈值降至 0.30 时 upcoming 命中 | 2 / 145 | 3 / 145 |

Validation 的 upcoming-vs-negative 正例率为 0.002149。AUC 有排序信号，但在这种
稀有度下 AP 和高精度区域的召回仍低；降低阈值到 0.30 并不能充分恢复漏报。正式阈值
仍为各自 validation 选定的 0.60。当前 near 最佳权重在 train 也没有学好 upcoming，
因此不能仅将其描述为验证集掉分。下一项检查直接使用同次训练已有的 last.pt，保持
原训练模块和评价规则，区分训练后期是否获得了排序能力；标签 `nearpreclast20260912`，
四份 CPU 诊断复用已结束的 `baseline_dense_diag`，已核实启动进程。此时 B5 seed42
训练观察到 epoch21，仍未将整批训练标为完成。

### 28.5 B4 近窗最后权重诊断完成

`nearpreclast20260912` 四份诊断已结束，tmux 退出 0；逐份验证 checkpoint 文件哈希、
epoch 与 train/validation 样本数。seed42/43 对应 last epoch12/15，二分类事件概率的
upcoming-vs-negative AP 分别为 train 0.082134/0.179934、validation 0.009541/0.009479。
相同近窗输入与 GCN-GRU 骨干在后期明显改善训练排序，验证排序没有同步改善。这支持
泛化限制，不能将最佳权重的训练弱表现等同于骨干完全没有表达能力；不替换正式 best。
另核实 B5 seed42 已完成（best6/total26），seed43 进入训练，整批尚未完成。

## 29. 主模型与 B4/B5 架构核查（2026-09-12）

用户要求检查是否存在使 baseline 无法找到 upcoming 的架构限制。重新 fetch 确认
`origin/dev_tyx=20c40e230aedee6aef2429d352413fbcf0fa571a`，对照该 Git 对象的
`PDFormer/factory_bn/{model.py,backbone.py,train.py}`、`FactoryBN_dense_f1_p80.json`
和当前 B4/B5、共用预测头及损失代码。以下为源码与配置证据，正式主 checkpoint 的实际
参数、覆盖配置和同包指标仍缺核验，不能把配方差异直接解释为已测得的因果贡献。

| 部分 | 主模型配方 | 当前 baseline 控制／近窗方案 | 对 upcoming 的含义 |
|---|---|---|---|
| 时空编码 | 5 个时空注意力块，含地理、语义和历史时间注意力，带模式键及节点位置/身份嵌入 | 每分钟两层 GCN 或 GAT，然后逐节点 GRU，末状态与历史均值融合 | 主模型反复交换已编码的时空信息；baseline 的时间表征经过汇聚压缩。可能影响弱征兆提取，尚非性能归因证据 |
| 可用历史 | 近窗摘要及编码窗之前最多 30 分钟摘要，事件表示还融合邻接队列与历史状态簇 | 本轮摘要只来自已有 30 分钟，后 5 维远历史为零 | 信息范围尚不相同，不能把潜在收益全算作 Transformer 骨干优势 |
| 事件头 | continue 与 onset 两个独立二分类头，并对历史冷状态组合两者 | 一个事件发生概率头；已试三分类变体，但三分类不是这套独立双头和监督 | onset 可获得专门的区分目标；三分类负结果不能代替双头机制的消融 |
| 事件损失 | 有 onset 辅助 BCE、含 upcoming 项的 soft F-beta、focal 和资源类型权重 | 当前为类型平衡的加权事件 BCE、起点与时长损失；已完成加权和三分类对照 | 主模型对提前事件的优化压力不同；损失差异不属于骨干表达能力差异 |
| 报警决策 | onset 阈值映射、未来占用连续段并入报警、资源类型阈值，另有 ongoing/prefix 抬升 | 独立事件头概率决定报告；hot 网格头另行评价，历史 hot 只参与共同起点解码 | 主模型可通过其他预测输出补发报告，最终 recall 不是纯事件头能力比较 |
| 训练过程 | 从 f180 续训、事件窗 4× 过采样、解冻 5 个编码块 | 当前从头训练、均匀抽样、seed42/43 | 训练经历和优化预算也是混杂因素，需披露并受控 |

两点容易混淆：该主配置 `use_stgnpp=false`，不能把当前分差归于 STGNPP；
`prefix_mlp` 的抬升由历史 hot 门控，直接服务持续事件，不能笼统视为 upcoming 高召回
的来源。主 `train.py::_epoch` 确实调用 `model.predict` 后计分，以上解码路径属于
实际评估代码路径，不是仅有未使用的辅助函数。

### 29.1 实际图覆盖核查

服务器在原 benchmark 新增 `baseline_graph_reach_audit_20260912.json`，SHA-256：
`9365d26867b7d3740e8c40d4fd297ce6f5969f37ace2d0371fdb0ab461d532c5`。
只读当前冻结包的图边表、节点目录和 train/validation split，检查对称边及自环，
未读取标签或评估 test。按每个 episode 的目标资源节点统计（不是窗口目标）：

| 图边表的两跳覆盖 | Train | Validation |
|---|---:|---:|
| Episode | 138 | 30 |
| Episode 内目标节点累计数 | 1758 | 380 |
| 两跳可覆盖全部活跃节点的目标节点数 | 1167 | 254 |
| 两跳未覆盖的目标节点→目标节点对 | 0 | 0 |
| 未覆盖来源类型 | 仅 buffer | 仅 buffer |

因此，不能用“两层 GCN/GAT 看不到其他机器”解释当前整体漏报。图上存在路径只证明
信息可能到达，不保证模型保留或利用这些信号；部分 buffer 的不可达与语义/时间建模
差异仍可单独检查。该结果基于导出图边表，不冒称完成了逐 checkpoint 的输入梯度归因。

### 29.2 当前归因与下一项诊断

已有证据更支持事件概率判别与泛化不足，尚不足以宣称 GCN/GAT-GRU 达到架构上限。
B4 近窗 last 的 train AP 明显提高但 validation 仍弱，是同一输入和同一骨干下的直接
证据；B5 三分类 last 也曾显示训练学习能力。更复杂的时空编码可能帮助泛化，但当前
主模型同数据、同输入、同事件头/解码的受控对比尚缺材料。

下一项优先做不重训的输出头诊断：在同一 B4/B5 checkpoint 上分别衡量事件概率与
已有未来 hot 预测对 upcoming 的排序信号，判断信息是否已在另一预测路径中出现。
不改变正式报警、阈值、真值或选模。之后才据证据考虑单独的 onset 监督或时间汇聚
对照，保留 GCN/GAT-GRU 身份；不直接搬入主模型的整套强制报警规则。

### 29.3 冻结预测头诊断准备

`diagnose_baseline_events.py --compare_hot_head` 在同一次 forward 中读取事件概率和
已有 hot 网格预测。Hot 分数预定义为：对起点 0、1、2 的每段连续 8 窗预测概率取最小值，
再对三个起点取最大值。该分数的构造不读取真实 hot、真实起点、历史 hot、真实有效
未来长度或预测完工长度。真值只用来按既有契约评价 upcoming 与无事件负例的 AP/AUC。
另以主参考配置的固定 hot 阈值 0.45 和 checkpoint 保存的事件阈值记录两头越阈值的
重合／独有／均未命中计数；这些是分数越阈值统计，不冒称正式 report 命中或新最优阈值。

原有 canonical report 完全保留，输出独占新文件且要求父目录已存在。流式运行可显式
登记源码 Git 提交和 SHA-256，无需更新仍在训练的模块。新增连续性、最晚起点、无标签
输入、无原数组改动、mask/ongoing 排除、空支持测试；本地诊断及 precursor 相关测试
共 18 项、11 个子测试通过。首轮使用近窗候选已有 best.pt 的 train/validation，
不新增训练、不替换 best、不运行 test。实际执行状态后续另记。

### 29.4 近窗整批与冻结预测头诊断完成

四次近窗训练正常结束，训练代码保持 `e5f6cb4`。按 B4 seed42/43、B5 seed42/43 排序，
best epoch 为 2/5/6/5，total epoch 为 12/15/26/25。B4 validation P/R/F1/upcoming R
均值为 0.82205317/0.69132420/0.75102038/0.01379310；B5 为
0.83816708/0.67579909/0.74823627/0.01034483。近窗方案没有稳定的实质 upcoming 提升。
完整四次 metrics/config/运行记录与 best/last 等文件哈希保存在
`baseline_dense_near_metrics_20260912.json`（113320 bytes），SHA-256：
`e71c84ded25a2b5fe861b45ecc12e2a0941193043a526654a9d8327a9d7a4b27`。

随后在无训练进程时部署 `00d4c6f`，以这四个冻结 best 权重完成
`b[45]_seed4[23]_{train,validation}_diagnostics_headprobe20260912.json` 共 8 份。
逐份核验 checkpoint SHA-256 与完整近窗导出、当前文件一致，诊断源码哈希与提交一致，
manifest 未变，epoch 和 train/validation 样本数 23859/5439 正确。四份 validation
canonical P/R/F1/upcoming R 与原保存 metrics 在 1e-10 容差内相同。两项 tmux 均
dead=1、exit=0，无相应训练或诊断进程，未使用 test。

| Validation frozen best | B4 s42 | B4 s43 | B5 s42 | B5 s43 |
|---|---:|---:|---:|---:|
| Event upcoming-vs-negative AP | 0.006336 | 0.008807 | 0.008201 | 0.007818 |
| Predicted hot-run upcoming-vs-negative AP | 0.008456 | 0.009430 | 0.010149 | 0.007853 |
| Hot 越阈值、event 未越阈值的 upcoming | 0 | 0 | 1 | 0 |
| Hot 越阈值、event 未越阈值的负例 | 33 | 16 | 11 | 21 |

AP 转录到六位小数，精确值保存在服务器。阈值为 checkpoint 原 event 阈值及固定 hot
0.45，越阈值数不是 canonical report 命中。现有 hot 输出没有提供大量可补回的 upcoming，
不据此移植主模型占用补报逻辑。下一项转向专门监督，而不是继续降低报警阈值。

## 30. 独立 onset 辅助监督受控实验

### 30.1 预注册与实现

`onset_aux` 以已完成的 `near_precursor` 为父对照，保留同一 208 episode、原 v6
manifest、近窗摘要及 GCN/GAT-GRU。唯一机制变化是添加独立 onset 辅助分支及其损失。
分支复制原二分类事件头的初始参数，但不共享参数；不消耗额外 RNG，所有已有参数及
原首次 forward 输出严格相同。新增分支在事件表示上训练，不参与正式报警或选模。

辅助目标严格沿用评分分组：正例为有效节点上的 event_will=1 且 start>0，负例为
event_will=0；所有 start=0 正例（包括历史 cold 的 start=0）和无效节点排除。
每个 batch 的辅助损失为 `0.5*mean(BCE_upcoming)+0.5*mean(BCE_negative)`，
某类缺失时其项为零，不重新归一化另一类；总损失系数固定 1.0，不搜索。这个定义是
batch 内按类归一化，不冒称全数据集均衡抽样，也不等同于主模型原加权 onset BCE。
它避免该辅助分支的正例梯度再按大量负例目标数稀释，检验额外监督对共享表示的影响。

保持原二分类事件头、其他所有损失、均匀抽样、优化器/学习率、dropout、早停、报告阈值
列表及 canonical 总 report F1 选模。B4/B5 各 seed42/43，共 4 次最多 60 epoch，
min_epochs/patience 仍为 B4 10/10、B5 15/20。归档标签 `onsetaux20260912`，复用原
目录，先逐文件验证归档近窗结果。near_far 尚未启动；不增加历史信息或复制主报警规则。

实现同时添加辅助头的只读 AP/AUC 诊断；分数与正式事件报告分开，不能用于选 checkpoint。
判读同时看两颗 seed 的正式 P/R/F1/upcoming recall、原事件头及辅助头的 train/validation
upcoming AP、误报与时间 MAE。若辅助头有信号而事件头没有，需要另行验证部署决策；
若仍只有训练改善，继续诊断泛化，不以本轮失败直接宣布骨干上限。

本地相关测试 29 passed / 11 subtests：涵盖正负梯度方向、ongoing/mask 排除、缺类、
无效配置、共享骨干梯度、初始权重/RNG/正式输出一致、辅助头参数不改变报警、内存
checkpoint 回读与配置差异。服务器开训前已确认干净 `dev_xwt@00d4c6f`、两项旧会话
退出 0、无相关 Python 进程；GPU 上其他 Isaac 进程保持原样。此处尚未记录开训成功。

### 30.2 部署与开训确认

实现提交 `df9ee0e` 已部署到服务器干净 `dev_xwt`，服务器 11 项预检通过（两骨干的
初始化、梯度、序列化、配置差异及辅助诊断隔离等）。开训前逐份验证四组近窗完整导出
中的文件哈希仍匹配当前 best/last/config/history/metrics/summary，manifest 未变，
新归档和运行记录标签均未占用。确认旧训练、诊断 pane 均已退出 0，无对应 Python
进程后，复用 `baseline_dense_v6` 启动四组顺序训练；日志为
`onsetaux20260912_train.log`，启动 pane PID=94376、首个 B4 seed42 Python PID=94392。

首组近窗 ZIP 已逐文件与冻结导出再次核对通过；运行记录为 started，新 config 为
`dense_onset_aux_v2`、`event_onset_aux=true`、`lambda_event_onset_aux=1.0`。
后续每组训练仍在开始前归档，不将尚未开始的组误写为已归档/完成。不在整批训练中
更新服务器源码，不根据中间 upcoming 单项改选模或临时改配方。当前尚无本轮最终成绩。

开训后又只读核验 B4 seed42 的 epoch4 last.pt：16641 个辅助头参数已与初始化时
相同的事件头发生分化，全部模型参数有限；源码为 df9ee0e，manifest 未变，
evaluate_test=false。这证明辅助分支实际参与更新，不代表已改善 validation upcoming。

### 30.3 Upcoming 的 batch 暴露率

从已冻结、已校验哈希的 `dense_event_support20260912.json`，按每个 episode 的
first_future_position 对所有 upcoming 工位目标取并集，得到含 upcoming 的样本数：
train 583/23859（2.4435%），validation 141/5439（2.5924%）。这与窗口-工位目标
595/145 不同；同一样本可包含多个正例，不能把目标数直接当作正例样本数。

对均匀无放回 batch，按超几何概率 `prod((N-K-i)/(N-i), i=0..batch_size-1)` 计算，
B4 batch24 没有 upcoming 的期望比例为 0.552104，B5 batch16 为 0.673043。
这是随机抽样下的期望，不是本次实际 batch 日志，也不代表永久看不到这些样本。
它说明辅助损失缺类分支会经常发生，按 batch 分别取均值也不等同于全数据集 50/50
均衡抽样。进一步核查 validation 没有完全缺席于 train 的 source_prefix；这只排除
整个粗粒度来源类别缺失，不能证明 episode 内分布一致或排除分布偏移。

### 30.4 B4 两次训练及八份冻结权重诊断完成

B4 两颗 seed 正常结束，未达到整体 P>=0.80 且 R>=0.70 的可行门，仍按原 fallback
report F1 规则选 best；没有为了 upcoming 替换为最后权重。实际新旧 config 对照通过：
model 只新增 event_onset_aux，loss 只新增 lambda_event_onset_aux=1，training 仅
training_profile 名称变化；完整 input_feature_contract 与各自近窗父对照一致。

| B4 onset 辅助监督 validation | seed42 | seed43 |
|---|---:|---:|
| Best / total epoch | 1 / 11 | 5 / 15 |
| Report P | 0.84429066 | 0.82190265 |
| Report R | 0.66849315 | 0.67853881 |
| Report F1 | 0.74617737 | 0.74337169 |
| Upcoming 命中 / 目标 | 0 / 145 | 3 / 145 |
| Ongoing recall | 0.77052632 | 0.77894737 |
| 保存阈值 | 0.70 | 0.55 |
| Start MAE（分钟，who TP） | 0.03825137 | 0.03230148 |
| Duration MAE（分钟，who TP） | 2.38544846 | 2.34631681 |
| Remain MAE（分钟，全样本） | 42.15194399 | 15.75402098 |

P/R/F1/upcoming R 均值为 0.8330966562/0.6735159817/0.7447745279/0.0103448276；
近窗父对照 F1/upcoming R 为 0.75102038/0.01379310，未改善。

`onsetaux20260912`（best）与 `onsetauxlast20260912`（last）两组标签各有
B4 seed42/43 × train/validation 共 8 份诊断，全部正常结束。逐份验证权重 SHA-256、
诊断源码及提交 df9ee0e、冻结 manifest、epoch 和 23859/5439 样本数；两份 best
validation 的 canonical P/R/F1/upcoming R 与保存 metrics 在 1e-10 容差内一致。

| Upcoming-vs-negative AP | B4 s42 train | B4 s42 validation | B4 s43 train | B4 s43 validation |
|---|---:|---:|---:|---:|
| Best 原事件头 | 0.005973 | 0.005918 | 0.014834 | 0.009733 |
| Best 辅助 onset 头 | 0.020208 | 0.010003 | 0.063347 | 0.025995 |
| Last 原事件头 | 0.096618 | 0.008469 | 0.348024 | 0.011584 |
| Last 辅助 onset 头 | 0.282862 | 0.031005 | 0.535091 | 0.032333 |

AP 转录到六位小数，两个头均排除 ongoing、只与无事件负例比较。辅助头在 validation
中保留部分排序信号，但与训练仍有显著差距；它不参与正式报警，不能把它的 AP 或
最后权重的单项结果当作正式提升。这也不是 GCN/GAT-GRU 理论上限的证明。

服务器 `baseline_dense_onset_aux_b4_20260912.json`（71326 bytes）保存两次完整 metrics、
config、运行记录、文件哈希及八份诊断摘要，SHA-256：
`8f4df2ad370dd9dd9870d2cae862caec8c88a4cd4f3bded3199878864f9d4a86`。
此时 B5 seed42 实际进程 PID=104896，观察到 epoch10；seed43 尚未开始。
训练 pane 存活，B4 诊断 pane 已退出 0。整批代码继续固定 df9ee0e，未新增搜索或评 test。

### 30.5 B5 seed42 训练完成，seed43 继续

B5 seed42 的 `dense_control_onsetaux20260912.json` 已为 validation_completed，best7、
total27；正式 validation P/R/F1/upcoming R 为
0.80982906/0.69223744/0.74643033/0.03448276，即 upcoming 5/145。相较近窗父对照
1/145 多 4 个命中，但 precision 从约 0.846 降至约 0.810，不能据单颗 seed 宣称稳定
判别或泛化提升。仍按原 report F1 选模；B5 seed43 已开始，最近观察到 epoch7。

在确认该组完成、诊断 pane 已退出 0、新文件标签未占用后，复用 baseline_dense_diag
启动 seed42 best/last × train/validation 四份诊断（启动 pane PID=117275）。最近已
四份均已完成：best 原事件头 AP train/validation 为 0.013069/0.007142，辅助头为
0.088299/0.021978；last 原事件头为 0.468398/0.009923，辅助头为
0.668891/0.030574。权重、诊断源码与提交 df9ee0e、manifest、epoch7/27 和样本数
23859/5439 均核验通过；best validation canonical 指标与保存 metrics 在 1e-10
容差内一致。诊断 pane 正常退出 0。不得重复启动已有输出。

B5 seed42 的 start/duration/remain MAE 分别为 0.03957784/2.18840814/13.41996141
分钟。B5 两颗 seed 的实际 config 与各自近窗父对照相比，仅新增 onset 辅助头、其
损失系数和 profile 名称，完整输入契约相同。分组审计还确认：当前 208 包所有 train/
validation upcoming 的历史末窗均为 cold；这限制了历史 hot 门控抬升的直接作用范围，
不能外推到主模型 204 包。

2026-09-13 后续核验 B5 seed43 已正常完成，best5/total25；validation
P/R/F1/upcoming R 为 0.84044944/0.68310502/0.75365239/0.01379310（2/145），
ongoing R 为 0.78526316，阈值 0.70，start/duration/remain MAE 为
0.02406417/2.34715843/15.00381604 分钟。训练 pane dead=1、exit=0，未发现训练
Python 进程。继续以固定 df9ee0e 补齐 seed43 best/last 的 train/validation 四份
诊断，文件及日志使用已登记的独占标签；当前前 12/16 份诊断已完成核验。

### 30.6 整批辅助监督训练与 16 份诊断完成

2026-09-13 核实最后诊断 pane 已退出 0，无相关训练或诊断 Python 进程。重新逐份
核验四组 best/last 权重 SHA-256、诊断源码哈希和提交 df9ee0e、冻结 manifest、
对应 best/total epoch 以及 train/validation 样本数 23859/5439。四份 best validation
canonical P/R/F1/upcoming R 均与保存 metrics 在 1e-10 容差内一致；全部未评 test。

| B5 seed43 upcoming-vs-negative AP | Train | Validation |
|---|---:|---:|
| Best 原事件头 | 0.010637 | 0.007708 |
| Best 辅助 onset 头 | 0.051083 | 0.019800 |
| Last 原事件头 | 0.416553 | 0.008778 |
| Last 辅助 onset 头 | 0.658729 | 0.018469 |

四次完整 metrics/config/运行记录、文件哈希及 16 份诊断摘要保存为
`baseline_dense_onset_aux_metrics_20260913.json`（357939 bytes），SHA-256：
`c6cad9f27a765852a4ccdd772307cf8ec1f5131f4af196f9ecd6fcdaa499e294`。
导出过程中系统 Python 缺少 hashlib.file_digest，改用等价 SHA-256 字节哈希后完整
验证通过；未修改训练模块、原结果或权重。

B4 正式 upcoming 为 0/145、3/145；B5 为 5/145、2/145。B5 的单颗 seed 增益
没有在另一颗 seed 重现，训练与验证排序仍存在明显差距。本轮不采用辅助监督方案
替换近窗父对照，也不因此宣布骨干上限。独立 onset 头始终未参与正式报警，本轮不是
双头报警部署机制的完整消融。

## 31. 远历史摘要对照

这一项独立检验已确认的主分支信息范围差异：原 30 分钟编码历史保持不变，另提供
编码窗之前最多 30 分钟的 5 项摘要。该对照以已经完成的 near_precursor 为父方案，
不在 onset_aux 上叠加新机制。当前主模型正式 204-episode checkpoint 尚未重评，
因此它只检验当前 208-episode baseline 的信息预算，不冒称主模型正式输入已全部对齐。

### 31.1 真实 baseline 构建路径核验完成

服务器以固定 df9ee0e 的既有 attach_precursor，分别构建 near 与 near_far 的
train/validation 输入；未改训练源码或当前训练。结果为
`baseline_far_precursor_contract20260912.json`（3984 bytes），SHA-256：
`1d6ae2b96d60d621027481582778fd15dc4861f22283a12b3abb908b94d41808`。

核验覆盖实际共享张量的 train 23859、validation 5439 个样本：

- 前 18 个摘要通道逐元素完全一致，near 模式的后 5 维严格为零；未请求的行没有构建摘要。
- near 的完整输入契约与冻结近窗父结果一致；两模式契约仅 mode、额外历史窗数和远历史
  字段策略不同。时间边界及前缀不变性的证据另见 27.11 的全量历史审计与已有边界测试。
- 有非零远历史摘要的 train/validation 样本为 23721/5409，全部构建值有限。
- B4/B5 各两颗 seed 的近窗/远历史模型初始权重和 RNG 状态完全一致；参数量分别为
  B4 273054、B5 285982。使用相同 23 维投影，不通过增加模型容量混入额外收益。
- manifest、split、normalization、sample index 和 episodes.npz 的前后哈希一致。
  使用现有共享张量缓存，只处理 train/validation 历史输入；不使用标签做该审计，未训练
  或评估 test。此结果证明输入构建与容量控制可用，不是模型性能结果。

### 31.2 后续受控训练注册（尚未启动）

预登记 `far_precursor` / `farprec20260913`，B4/B5 各 seed42/43，共 4 次，从头训练、
最多 60 epoch。其参数、损失、均匀抽样、优化器、学习率、早停、阈值列表和 canonical
report F1 选模沿用 near_precursor；event_onset_aux=false、lambda_event_onset_aux=0。
唯一信息变化为后 5 维从零改为更早历史的 4 个均值（queue、blocked、inbound wait、
material shortage）和 queue 最大值；不提供原始未来数据、事件标签或 run/scenario ID。

必须先等当前 onset_aux 四次训练及对应冻结诊断全部结束，保存完整结果并核验已退出
进程；随后复用原目录、用唯一标签逐文件验证归档当前 onset 权重，再运行远历史对照。
不能把旧 near 结果文件路径误当成当前 onset 权重，或在运行中的服务器 pull 代码。
正式比较仍报告每颗 seed、均值、upcoming AP/recall、误报和时间 MAE；未使用 test
调参。这个注册不是“已开训”或“已改善”，也不改变现有实验的选模与结果。

### 31.3 配置预检与启动

在服务器固定 df9ee0e 上，将 far_precursor 的四组拟用配置与已保存的 near_precursor
完整配置逐字段核对：model 仅 event_precursor 从 near 改为 near_far；training
仅 profile 名称改变；loss 完全相同、onset 辅助分支关闭。JSON 的阈值列表与 Python
默认元组先统一为相同序列表示，阈值数值和顺序均一致。四组均通过，不涉及重新调参。

开训前重新确认服务器 dev_xwt tracked clean、两项 pane 已退出 0、无相应 Python
任务，四组现有权重/config/history/metrics/summary 的哈希与完整 onset 导出一致，
manifest 未变，新日志、归档及运行标签均未占用。GPU 上其他 Isaac 进程 PID3347767
保持原样；可用内存约 39 GB。复用已退出的 baseline_dense_v6 启动四组顺序训练，
pane PID=137283，日志 `farprec20260913_train.log`，源码不 pull、固定 df9ee0e。
沿用既有 OMP_NUM_THREADS=2、OPENBLAS_NUM_THREADS=2，其他配置按上述核验结果。

此时仅确认启动命令及 pane 存活；实际 Python、首组归档、新配置和 epoch 进展须继续
核验，不能将尚未开始的后三组写成已归档或已完成。

随后已确认 B4 seed42 实际 Python PID=137299，并观察到 epoch1 日志。首组
`model_before_farprec20260913.zip` 的 11 个文件全部匹配归档 manifest，其中
best/last/config/metrics/history/summary 还与冻结 onset 完整导出的哈希一致。
实际新记录为 started，config 为 dense_far_precursor_v2、near_far、额外 30 窗，
event_onset_aux=false、lambda_event_onset_aux=0、evaluate_test=false。manifest
与父对照不变。后三组当时尚未开始；这不是整批性能结果。

### 31.4 首组训练完成、第二组继续

B4 seed42 已正常完成，best1/total11；validation P/R/F1 为
0.82812500/0.67762557/0.74535409，upcoming 0/145，ongoing recall 0.78105263，
阈值 0.65。Start/duration/remain MAE 分别为 0.03908356/2.42473698/40.51504550
分钟。近窗父对照 upcoming 为 2/145；单组远历史结果暂未改善，不能据此提前判定
整批结果，也未改选模或以中途 epoch 替换 best。

B4 seed43 已接着训练，实际 Python PID=141321、记录 started；其 onset 归档
11 个文件逐一验证通过，关键文件哈希与冻结整批 onset 导出一致。B5 两组尚未开始。
在已退出的 baseline_dense_diag 中启动 B4 seed42 best/last × train/validation 四份
诊断，pane PID=142098、实际 Python PID=142100，标签 farprec20260913 与
farpreclast20260913，日志 farprec20260913_b4s42_diagnose.log。诊断仍在运行，尚无
四份完整验证结论；不重复启动已有标签或更新服务器训练源码。

### 31.5 B4 两组训练完成，首组四份诊断已核验

B4 seed43 正常结束，best5/total15；validation P/R/F1/upcoming R 为
0.81164021/0.70045662/0.75196078/0.01379310（2/145），ongoing R 为 0.80526316，
阈值 0.55。Start/duration/remain MAE 分别为 0.02868318/2.27562284/14.82699580
分钟。B4 两组 upcoming 0/145、2/145，近窗父对照为 2/145、2/145，未看到提升。
B5 seed42 已开始，实际 Python PID=146730；整批未完成。

B4 seed42 四份 best/last train/validation 诊断全部正常结束；逐份权重与诊断源码
哈希、提交 df9ee0e、manifest、epoch1/11、样本数23859/5439 及 split 均核验通过，
best validation 原指标在 1e-10 容差内一致。Best AP train/validation 为
0.005372/0.005739，last 为 0.088649/0.010517；保存阈值下 upcoming 概率漏报
依次为 595/145、532/144，时间漏报均为零。远历史首组仍有训练与验证学习差距，
主要漏报发生在概率判断阶段。数值转录到六位小数，完整 JSON 保存在服务器。

随后复用已退出的诊断 pane 启动 B4 seed43 的四份 best/last train/validation
诊断，日志 farprec20260913_b4s43_diagnose.log，源码仍固定 df9ee0e。一次 UU
读取连接中断经重新获取窗口恢复，实查进程后继续；没有因 UI 错误重启训练。

### 31.6 B4 八份诊断全部完成并导出

B4 seed43 四份诊断正常结束；与 seed42 合计八份，重新全部核验当前权重 SHA-256、
诊断源码哈希与 df9ee0e 提交、冻结 manifest、对应 epoch、split 和样本数。两份
best validation canonical 指标在 1e-10 容差内相同，均未评 test。

| B4 seed43 upcoming-vs-negative AP | Train | Validation |
|---|---:|---:|
| Best epoch5 | 0.010004 | 0.008706 |
| Last epoch15 | 0.202656 | 0.010266 |

两颗 seed 的 P/R/F1/upcoming R 均值为
0.8198826058/0.6890410959/0.7486574389/0.0068965517。当前 B4 结果不支持这五项
远历史摘要带来改善，且两颗 seed 均有后期训练排序提升、验证排序仍弱的现象；不能
据此断言所有长历史表示无用，也不能代替 B5 的结果。

两次完整 metrics/config/运行记录、文件哈希及八份诊断摘要已保存为
`baseline_dense_far_b4_20260913.json`（172561 bytes），SHA-256：
`f24e51738a2bd51bd708350fbb1a17a5882ef1db3e40d3e446ab33021b073388`。
B5 seed42 的实际 far 配置和旧 onset 归档 11 个文件亦核验通过，最近观察到 epoch14、
Python PID146730；seed43 尚未开始。诊断 pane 已退出 0，服务器代码仍 df9ee0e。

### 31.7 B5 两组训练完成，八份诊断排队执行

B5 seed42/43 都已正常结束，best3/5、total23/25；训练 pane dead=1、exit=0，
无训练 Python 进程。两组实际 far 配置和旧 onset ZIP 各 11 个成员均已核验。

| B5 远历史 validation | seed42 | seed43 |
|---|---:|---:|
| P | 0.83747178 | 0.83054627 |
| R | 0.67762557 | 0.68036530 |
| F1 | 0.74911661 | 0.74799197 |
| Upcoming 命中/目标 | 2/145 | 2/145 |
| Ongoing R | 0.77894737 | 0.78210526 |
| 阈值 | 0.55 | 0.65 |
| Start MAE | 0.04851752 | 0.02281879 |
| Duration MAE | 2.28503227 | 2.41161227 |
| Remain MAE | 20.52421788 | 13.85537770 |

先完成第 32 节 B4 冻结检查并核验其终态与完整输出，然后复用已终止的诊断 pane，
启动 B5 两颗 seed 的 best/last × train/validation 共八份原诊断，源码 df9ee0e。
日志 farprec20260913_b5_diagnose.log，pane PID169477。新 Bash 包装器先经语法
检查，结束时在日志记录 DIAG_BATCH_EXIT_CODE，避免仅依赖 tmux 是否及时回收退出码。
一次包装命令的字符串语法错误发生在启动前，未运行训练或诊断；修正后才实际启动。
此时整批训练已完成，B5 八份诊断仍待完成及逐份核验。

### 31.8 四次训练及 16 份远历史诊断全部完成

2026-09-13 核验训练和诊断 pane 均 dead=1、exit=0，B5 诊断日志末行为
DIAG_BATCH_EXIT_CODE=0。逐份核对四组 best/last 权重、源码 df9ee0e、manifest、
epoch、split 及样本数，四份 best validation canonical 指标与原保存 metrics 在
1e-10 容差内相同。实际配置均 near_far、onset 辅助关闭、test 关闭。

| B5 远历史 upcoming-vs-negative AP | s42 train | s42 validation | s43 train | s43 validation |
|---|---:|---:|---:|---:|
| Best | 0.006051 | 0.006104 | 0.009204 | 0.007873 |
| Last | 0.184805 | 0.008044 | 0.308020 | 0.008621 |

AP 转录至六位小数。四组 upcoming 命中为 0/2/2/2（每组145个目标），没有稳定的
实质增益；B5 后期训练排序与验证排序仍有明显差距。这只是否定当前五维摘要方案的
稳定增益，不是所有远历史建模的上限，也不证明双方正式输入已经完全一致。

四次完整 metrics/config/运行记录、文件哈希及全部诊断摘要保存为
`baseline_dense_far_metrics_20260913.json`（345240 bytes），SHA-256：
`d3420cc6c4c23ccd4a3d07b514bdc35443e1f8e39e964c0b444ab7cb1994f020`。

## 32. 冻结 onset 报警路径检查

用户指出 baseline 可能缺少提前识别组件。第 30 节只检验辅助监督，不能代替 onset
输出参与报警的检查。新增可选诊断 `--compare_onset_report`，使用同一已冻结的
onset_aux best.pt，比对原事件概率与一个固定决策：历史末窗 hot>0.5 时保持原概率，
其余节点取 max(event_probability, onset_probability)。该分数函数只接收两路预测
概率及已有历史 hot；不读取真实未来、真实起点、未来有效长度或完工长度筛选。

两路使用 checkpoint 保存的训练阈值列表，并包含其已选报告阈值；起点/时长预测、
真值、匹配容差、目标 mask 和误报统计完全相同。完整阈值曲线独立保存，原 canonical
输出不变，不自动选择阈值或 checkpoint。首先比较原保存阈值下新增命中与误报；
曲线用于检查是否存在值得后续验证的信号，不把最后权重或某个单项高点作为正式结果。
本检查没有单独校准 onset 概率，也没有按双头决策重新训练和选模，因此负结果不能
排除所有 onset 解码方法，更不是主模型完整解码器的复刻或架构上限证明。

实现只修改诊断文件及其测试，B4/B5 训练模块和默认诊断行为保持原样。新增测试覆盖
history 门控与边界、无输入改动、原 canonical 不变、补回 upcoming 同时增加误报、
起点错误不能算命中、无效节点/空支持、缺分支/无效阈值及 CLI 开关。相关 26 项测试、
3 个子测试通过，实现提交 3b0e784；服务器尚未部署，远历史训练/诊断继续使用 df9ee0e。

预登记后续标签 onsetreport20260913，B4/B5 seed42/43 × train/validation 共 8 份，
仅使用第 30 节原规则选出的 best。输入保持对应近窗契约；从已验证的
model_before_farprec20260913.zip/best.pt 直接读取，不恢复覆盖当前远历史权重。
先完成当前训练和诊断、保存完整远历史结果，再核验退出状态与归档后部署执行。
不新增训练或模型目录，不改变事件真值和匹配评分规则，不使用 test。

### 32.1 执行顺序调整：先检查已冻结的 B4 归档

服务器 B5 远历史训练继续，B4 的两份 onset_aux best 已在校验过的 ZIP 中，当前
诊断 pane 空闲。为利用已有 CPU 诊断会话，将 B4 四份检查提前；B5 两颗 seed 仍等
各自权重归档完成并核验后再安排。此调整仅改变执行顺序，未观察新报告曲线前已经
固定第 32 节的权重选择、历史输入、门控、概率组合和阈值列表。

不在训练期间 pull：只 fetch Git 对象，通过 stdin 执行固定 3b0e784 的诊断源码，
模型/数据模块仍从服务器 df9ee0e 工作区导入。已核对两个提交之间的代码差异只涉及
诊断与测试，B4/B5 训练及输入构建模块相同。执行前须以同样的内存加载方式运行新增
诊断测试，验证归档成员与冻结 onset 完整导出的哈希，并确认工作区 HEAD 和 tracked
文件不变。此处为执行安排，不表示新检查已运行或已有性能结论。

### 32.2 内存执行预检与 B4 首组结果

已只 fetch Git 对象、未 pull。服务器内存加载 3b0e784 的诊断与测试源码，新增
3 项测试全部通过；HEAD 仍为 df9ee0e，tracked 工作区不变。新诊断 SHA-256 为
`2b66392909d141995a4d603950f9bacc978b8680c2b5d65fe07e8cb3db58d045`。
两个提交间 source 差异确实只有诊断和测试。B4 两颗 seed 的 ZIP/best.pt 成员哈希
与冻结 onset 完整导出一致、配置为近窗且含独立 onset 头。随后复用已退出的
baseline_dense_diag，经 git show | python stdin 启动四份诊断，pane PID156899、
首个 Python PID156903，日志 onsetreport20260913_b4_diagnose.log。

B4 seed42 的 train/validation 两份已生成并核验：ZIP 与成员哈希、诊断 Git 对象
源码及 SHA、manifest、epoch1、样本数23859/5439、split 全部正确；原 ranking/groups
与之前 onset 诊断完全一致。event_only 在保存阈值的完整报告行与顶层原报告相同，
validation 正式指标与冻结 metrics 在 1e-10 容差内一致。两路阈值列表均与 checkpoint
原配置一致，加入 onset 后每个阈值下的预测数和正确报告数均不减少。

| B4 seed42 保存阈值0.70 | 原事件头 | 历史 cold 取两头最大值 |
|---|---:|---:|
| Validation P | 0.844291 | 0.825254 |
| Validation F1 | 0.746177 | 0.738648 |
| Validation upcoming 命中 | 0/145 | 0/145 |
| Validation 误报（含起点错误） | 135 | 155 |
| Train upcoming 命中 | 0/595 | 13/595 |
| Train 误报 | 450 | 529 |

固定阈值列表中，validation 原事件头/组合头分别有 12/9 个 P>=0.8 的点，其
upcoming 命中范围均为0至0；train 分别有12/9个点，命中范围为0至0、0至13。
这是已登记离散曲线的范围统计，没有选定新阈值或替换 checkpoint，不是所有可能
校准方案的上界。对 seed42，“onset 头未参与报警”不能单独解释低召回；组合路径
补回部分训练正例却没有补回验证正例。此时整个 B4 四份批次尚未完成，不能外推。

### 32.3 B4 四份冻结报告检查全部核验

两颗 seed 的四份 JSON 全部生成、日志均有对应 Output 完成标记。诊断外壳 PID156899
处于 Z（已退出、待回收）状态，tmux pane_dead=1，但退出码和信号字段均为空。
因此不登记为“正常退出0”；确认没有活跃的该诊断 Python 后，对四份完整文件做
独立核验，不覆盖或重跑。导出中保留 terminal_state.exit_code_unavailable=true。

四份归档/成员/源码哈希、manifest、epoch1/5、样本数与 split 全部通过；原排序、
分组、辅助头统计与第30节旧诊断完全相同。两颗 seed 原保存阈值下 event_only 完整
报告行与顶层原报告相同，validation 指标与冻结 metrics 在1e-10容差内一致。
全部同阈值对照满足预测数、正确报告数和误报数不减少，未改变真值或起点预测。

Seed43 在保存阈值0.55下，原事件头与组合路径的 validation P/R/F1 分别为
0.82190265/0.67853881/0.74337169 和 0.23752553/0.74337900/0.36001769；
upcoming 从3/145增至52/145，但误报从161增至2613，不能采用该固定阈值组合。
其 P>=0.8 的已登记离散曲线点中，upcoming 命中范围由原0至3变为组合0至6。
这是曲线范围，不是选定的新阈值，也不排除后续独立校准；该 raw-max 检查没有显示
同时维持精度与显著提高提前召回的稳定收益。首组为0至0，两颗 seed 表现不能混合
成一个最佳指标。结果说明专门分支能触发部分提前报警，但大量误报仍是关键限制。

四份文件哈希、完整两路曲线及原保存阈值统计汇总为
`baseline_onset_report_b4_20260913.json`（1170612 bytes），SHA-256：
`1900fb4d778c142f4cdf001da0b428600cc56a46c50c4550a3d824fac3387cd7`。
源码通过 stdin 使用3b0e784，底层模型运行模块和服务器 HEAD 保持df9ee0e。
B5 冻结 onset 报告检查尚未执行，等当前 B5 far 原诊断完成后再复用会话；不新增训练。

### 32.4 B5 四份冻结报告检查启动

在第31.8节完整导出与退出核验后，复用 baseline_dense_diag，启动 B5 seed42/43 的
onset_aux best.pt × train/validation 四份检查。读取既有
model_before_farprec20260913.zip，逐一验证 ZIP CRC 及关键成员哈希与完整 onset 导出
相同；不解压替换当前 far 权重。使用同一3b0e784诊断 Git 对象经 stdin 执行，底层模块
及 HEAD 仍 df9ee0e，CPU2线程。新日志 onsetreport20260913_b5_diagnose.log，启动
pane PID179389；最近实际 Python PID181276，seed42 train 已输出、validation 正在运行。
包装器语法检查通过，并登记日志退出标记；四份结果尚待完整核验，不重复启动。

### 32.5 B5 四份冻结报告检查完成，整批 8/8 核验

B5 诊断 pane 正常退出0，日志末行为 DIAG_BATCH_EXIT_CODE=0。四份 ZIP及成员、
源码、manifest、epoch7/5、样本数和split均核验通过；原排序、分组及辅助头统计与
第30节完全相同，event_only保存阈值行与原canonical相同，validation指标在1e-10
容差内一致。所有已登记同阈值曲线点满足报告数、命中数和误报数不减少。

| B5 validation，原保存阈值 | s42 event | s42 cold max | s43 event | s43 cold max |
|---|---:|---:|---:|---:|
| 阈值 | 0.55 | 0.55 | 0.70 | 0.70 |
| P | 0.80982906 | 0.71244240 | 0.84044944 | 0.74853229 |
| R | 0.69223744 | 0.70593607 | 0.68310502 | 0.69863014 |
| F1 | 0.74643033 | 0.70917431 | 0.75365239 | 0.72272083 |
| Upcoming命中/145 | 5 | 12 | 2 | 9 |
| 误报 | 178 | 312 | 142 | 257 |
| 已登记P>=0.8曲线点的upcoming命中范围 | 0–5 | 0–6 | 0–2 | 0–6 |

阈值范围是描述统计，没有据此选择新阈值或新checkpoint。两颗seed在原阈值下都以
跌破P>=0.8为代价提高召回，不采用固定raw-max组合。与B4合计8份检查完成；结果说明
专门分支有部分提前信号，但单独补上这条报警路径尚不能解决误报与泛化。未进行独立
onset校准或双头报告参与选模的重训，不将本次负结果扩展为这些方法全部无效。

B5完整四份结果、两路曲线及原阈值统计存
`baseline_onset_report_b5_20260913.json`（1171996 bytes），SHA-256：
`d1f2519da3448f19262d9a3caf0199f923671786b86bea47197b933ba4111e50`。
B4退出码缺失的历史记录保持原样；不能把B5正常退出0推广为整批外壳均退出0。

## 33. 时间注意力汇聚对照准备

针对“是否缺少提前识别组件”的问题，单独检验 GRU 历史均值是否削弱了部分历史信号。
B5 既有 GAT 是每个时刻的空间注意力，不能等同于主模型的历史时间注意力；但 GRU
本身能编码顺序，因此这一差异不构成 baseline 无法学习 upcoming 的证明。

注册 `temporal_attention` / `timeattn20260913`，父对照为 near_precursor，B4/B5 各
seed42/43、最多60epoch。将 `last_mean` 改为 `last_attention`：保留 GRU 末状态，
将历史均值替换为单个可学习查询对历史 GRU 状态的 softmax 加权均值，再用相同的
Linear/GELU/LayerNorm 融合。每个模型新增128个参数，GCN/GAT层及GRU保持不变。
这是增强 baseline 的单项消融，不称为原始架构结果，也不是主模型多层时空注意力的复刻。

查询从零初始化，不消耗 RNG；采用“原均值 + 相对均匀权重的修正”计算，在初始化时
原参数、RNG与全部首次输出严格相同。近窗输入、二分类事件头、其他预测头、损失、
均匀抽样、优化器、学习率、早停、阈值和 canonical report F1 选模不变；aux关闭，
远历史关闭。没有根据任何新训练成绩修改配方。

本地13项汇聚/模型检查通过，另28项相关回归检查、11个子测试通过。覆盖初始参数及
带dropout的首次输出/RNG一致、注意力能选择不同历史状态、梯度回传、节点mask、
内存checkpoint回读、两个骨干及两颗seed、与父对照的唯一配置差异。服务器尚未部署
该实现，尚未启动训练。等第32节 B5 检查完成并保留结果后，再做服务器预检、逐文件
归档当前 far 结果并复用原目录；期间不 pull 模型模块、不评 test。

### 33.1 服务器预检与首组启动

实现提交abc671383c92135483fdd6c5e09971018015364c。先核验全部far、B4/B5冻结报告
导出哈希和四组当前far文件哈希，确认两项pane退出0、无相应Python进程，再将服务器
干净dev_xwt从df9ee0e快进到abc6713。此前诊断期间仅fetch Git对象，没有更换模型模块。

服务器对四组实际模型配置完成预检：与各自冻结near配置相比，只有temporal_readout
及profile名称不同；loss相同，旧字段缺省由原配置类规范化，JSON列表/元组规范化后
训练参数相同。所有已有初始权重、RNG及带dropout的首次输出严格相同；新增query
及GRU梯度非零，mask和内存checkpoint回读通过。参数量B4为273054→273182，B5为
285982→286110。预检文件baseline_temporal_attention_preflight20260913.json，
1873 bytes，SHA-256：
`004277d1859ca304064426d6eb1ffa797b006ab4d360b266b711d11fa6701ed9`。

预检后复用baseline_dense_v6启动四组顺序训练，日志timeattn20260913_train.log，
pane PID188079、首组B4 seed42实际Python PID188096。首组far归档
model_before_timeattn20260913.zip全部11个文件与归档manifest一致，关键文件还与
完整far导出一致；新记录started，实际config为last_attention、near、aux关闭、test关闭。
GPU可用约28.7GiB，既有Isaac进程PID3347767保持原样。后三组尚未开始，普通路径仍为
far权重；按每组开训时归档，不能写成已经全部归档。训练源码固定abc6713，尚无本轮
最终成绩，也没有改用upcoming单项选模。

### 33.2 冻结时间汇聚观察诊断

新增 `--inspect_temporal_attention`，代码4b48e2b80e08ef7045d779cb0918834cd2cb8994，
诊断源码SHA-256：
`7224e47f953b4af714b6dcf6bb285f0f27bae65edf631ece09c23fb641c72aae`。
通过一次原始forward的临时hook读取历史GRU状态及实际汇聚输出；由同一个已学习query
重建时间权重，记录query范数、权重距均匀分布的总变差、最大权重、归一化熵、实际
汇聚相对原均值的L2改变量。按ongoing/upcoming/negative事后分组，保留各时间位置的
平均权重及分位数。输入仅历史状态和query，未来标签只用于诊断分组；不改变预测、
阈值或选模。注意力权重不是因果重要性，表示改变量也不能直接等同于召回贡献。

本地19项检查及5个子测试通过，新增的3项也在服务器内存执行通过；验证单次forward、
预测/参数/RNG不变、hook在成功和异常后均移除、零query均匀权重、分组和mask、
空组/单时间步及非法输入。仅fetch新Git对象，模型相关6个模块的blob与abc6713相同，
服务器HEAD及训练模块保持abc6713；诊断代码经stdin运行，不在训练期间pull。

### 33.3 B4 seed42完成，首组四份诊断启动

B4 seed42正常完成，best2/total12；validation P/R/F1/upcoming R为
0.80876068/0.69132420/0.74544559/0.01379310（2/145），ongoing R为0.79473684，
保存阈值0.60。Start/duration/remain MAE为0.04095112/2.27932024/28.95911663分钟。
按相同seed的near父对照比较，F1由0.74531096微升至0.74544559，upcoming仍2/145；
不能与父对照两seed均值混比，也不能从这一组确定整批效果。

B4 seed43已开始，实际Python PID192940，最近观察到epoch11。其far归档11文件、
关键文件与完整far导出及实际last_attention/near/aux关闭/test关闭配置均核验通过。
B5两组尚未开始。复用已退出的baseline_dense_diag，启动seed42 best/last ×
train/validation四份冻结诊断，并加入33.2节观察；标签timeattn20260913及
timeattnlast20260913，日志timeattn20260913_b4s42_diagnose.log，pane PID196237、
实际Python PID196242。当前诊断仍运行，尚无完成核验结论，不重复启动。

### 33.4 B4两组训练完成，首组汇聚统计与泛化差距

B4 seed43已正常完成，best5/total15；validation P/R/F1/upcoming R为
0.81925134/0.69954338/0.75467980/0.01379310（2/145），ongoing R为0.80421053，
阈值0.55。Start/duration/remain MAE为0.02872063/2.29687619/15.09694913分钟。
两颗seed的upcoming均为2/145，与各自near父对照相同；F1均值约0.7500627，父对照
约0.7510204，尚未显示稳定改善。数值转录八位小数，正式指标保留原CUDA评估值。

首组四份诊断已全部输出、外壳正常退出0。四份权重/源码/manifest/epoch/样本数及
分组计数核验通过；best validation出现一个负例跨阈值的CPU/CUDA差异，完整复现与
处理见33.5，不能写成两设备报告完全相同。

| B4 s42 upcoming组 | Best train | Best validation | Last train | Last validation |
|---|---:|---:|---:|---:|
| Upcoming-vs-negative AP | 0.00687252 | 0.00644431 | 0.07918116 | 0.00968419 |
| Query L2 | 0.21593310 | 0.21593310 | 0.81003129 | 0.81003129 |
| 时间权重距均匀分布TV中位数 | 0.00195198 | 0.00197301 | 0.01282728 | 0.01561396 |
| 汇聚相对均值改变量中位数 | 0.00135524 | 0.00144304 | 0.01385991 | 0.01708787 |

最佳权重的汇聚仍很接近均值；最后权重的表示改变量中位数约1.4%/1.7%，训练排序
提高而验证未跟上。该观察只说明当前轻量查询的实际行为，不证明所有时间注意力无效，
更不能根据权重大小直接宣称因果重要性或骨干上限。

B5 seed42已开始，实际Python PID198500，最近epoch23；其far归档全部11成员及实际
配置核验通过，seed43尚未开始。完成数值核对后复用诊断pane启动B4 seed43的四份
best/last train/validation诊断，日志timeattn20260913_b4s43_diagnose.log，pane
PID207920、实际Python PID207925。该四份尚未完成，不重复启动。

### 33.5 冻结验证报告的设备数值差异已定位

B4 seed42 best的CPU batch32诊断报告935条、命中757条；原CUDA batch24正式评估
报告936条、命中同为757条，upcoming均2/145。因此原1e-10完全复现检查未通过；
没有静默放宽容差、修改阈值或覆盖原结果。为定位原因，固定同一权重、5439个validation
样本及0.60阈值，逐batch比较CPU观察诊断开/关，再复现原CUDA设置，并在差异样本
上交叉核验CPU batch24、CUDA batch32及重复CUDA batch24。

数值审计正常退出0，权重/manifest前后哈希不变，完整检查通过：CPU观察开关对所有
样本的全部预测输出严格相同；CPU32复现原CPU诊断，CUDA24完整复现保存的正式指标。
唯一概率跨阈值发生在sample_index26422、node32（num08_workbench_ws0），真实负例：
CPU24/32均为0.5999934673309326，CUDA24/32及重复CUDA24均为0.6000308990478516。
全有效目标概率最大绝对差为0.0003158450126647949；另有一个预测起点argmax不同，
未改变最终命中数。结果支持设备推理数值差异，而不是观察hook改变预测；不将之概括为
所有设备/所有checkpoint都已经审计。正式成绩继续使用原CUDA值，CPU诊断值单独保留。

审计保存为baseline_timeattn_b4s42_numerical_audit20260913.json（22132 bytes），含
完整审计脚本、脚本哈希、两路报告、交叉设置概率和来源，SHA-256：
`9172284d23cf648a7079c52faa45f8ab58bdce1c1c66f4ed72fb238ce7185759`。
最初审计启动因tmux缺少DENSE_DIR环境变量退出1，未开始推理/生成结果；补齐参数并
将复用检查改为确认该已失败终态后，仅重试审计。原失败日志保留，成功日志为
timeattn20260913_b4s42_numerical_audit_retry1.log；未重启训练、未覆盖四份原诊断。

### 33.6 四次时间汇聚训练完成，冻结诊断继续收尾

训练pane已正常退出0，timeattn20260913_train.log末行为TRAIN_BATCH_EXIT_CODE=0。
B5 seed42/43分别为best3/5、total23/25；实际配置均为last_attention、near、aux关闭、
test关闭，四组far归档各11个成员均已核验。下表沿用各次正式选定checkpoint的原CUDA
验证指标，不根据upcoming单项另选epoch；四组upcoming均为2/145。

| B5 validation | Seed42 | Seed43 |
|---|---:|---:|
| P | 0.83652762 | 0.81936888 |
| R | 0.67762557 | 0.68767123 |
| F1 | 0.74873865 | 0.74776564 |
| Upcoming R | 0.01379310 | 0.01379310 |
| Ongoing R | 0.77894737 | 0.79052632 |
| 阈值 | 0.55 | 0.60 |
| Start MAE (min) | 0.04851752 | 0.02523240 |
| Duration MAE (min) | 2.28531456 | 2.41945028 |
| Remain MAE (min) | 20.49183636 | 13.84202161 |

B4八份冻结诊断已全部核验，seed43的best validation与保存正式指标在1e-10容差内
一致；seed42保留33.5的设备差异说明。完整B4导出baseline_dense_timeattn_b4_20260913.json，
257125 bytes，SHA-256：
`7030c7505836dbc935f5e8120c0f02cd1a028255077a2ac1ebb2e34956235abf`。
导出包含原CUDA指标、CPU诊断报告及完整数值审计，不混淆两路数值。

| Upcoming诊断 | Best train | Best validation | Last train | Last validation |
|---|---:|---:|---:|---:|
| B4 seed43 AP | 0.01011920 | 0.00879545 | 0.19218590 | 0.01024910 |
| B4 seed43汇聚相对均值改变量中位数 | 0.00598663 | 0.00594889 | 0.02001793 | 0.02281547 |
| B5 seed42 AP | 0.00650444 | 0.00611549 | 0.18110063 | 0.00801939 |
| B5 seed42汇聚相对均值改变量中位数 | 0.00278260 | 0.00296816 | 0.02326306 | 0.02498073 |

B5 seed42四份诊断正常退出0，来源、权重、manifest、epoch、样本数和观察分组计数
核验通过；best validation与正式指标在1e-10容差内一致。随后复用baseline_dense_diag
启动seed43最后四份检查，日志timeattn20260913_b5s43_diagnose.log，pane PID220198。
诊断仍使用4b48e2b的Git对象经stdin执行，服务器HEAD/模型模块保持abc6713。
此时12/16份诊断已核验，最后四份尚在运行，不重新启动训练或已有诊断。

### 33.7 已完成搜索复核与架构归因边界

重新查第21节和baseline_event_sampling_20260907.json，确认旧134-episode/v5包已有
B4/B5各uniform_control、event4、upcoming4、seed42/43共12次完整抽样对照。其manifest
为66c6554d0ae1c7a293e482a201829a4c324f1d9d0b88934c1c2a6c1e27333183，train 13813窗，
与当前208包不同。旧实验中event4/upcoming4提高部分召回，但precision和F1下降，
两模型均保留uniform_control。不能将其写成当前208包结论，也不能把过采样说成从未
尝试的新方向直接重跑。本次只复核历史，未登记或启动新抽样搜索。

原B4/B5已有事件概率、未来起点和时长预测头，低upcoming不能解释为没有未来输出。
主模型参考配方具有独立onset监督和冷状态补报、多层时空注意力等差异；其中监督、
输入范围、报警决策和训练经历必须与骨干结构分开讨论。已有onset辅助头明显改善
最后权重的训练排序，但验证排序仍弱；固定raw-max补报伴随大量误报。当前时间汇聚
四次训练也未获得实质upcoming改善。这些证据支持“提前信号判别与泛化不足”，尚不能
证明GCN/GAT-GRU在架构上无法预测upcoming，也不能把一个轻量汇聚查询的负结果外推
为全部时间注意力无效。主模型同包、同输入、同解码的受控结果仍未获得，正式净结构
优势不能由主204包和baseline208包的分数差直接推出。

### 33.8 整批16份冻结诊断完成并保留完整结果

B5 seed43最后四份诊断正常退出0，日志末行为DIAG_BATCH_EXIT_CODE=0；权重、源码、
manifest、best5/last25、train23859/validation5439和观察分组计数均通过。Best validation
与原CUDA正式指标在1e-10容差内一致。Upcoming AP依best train/validation、last
train/validation顺序为0.00918577/0.00784095/0.31128931/0.00858045，汇聚相对均值
改变量中位数为0.00592527/0.00727255/0.03036844/0.03325982。训练排序提高而验证仍弱。

四组完整记录、配置、原CUDA指标、文件哈希及16份CPU诊断汇总为
baseline_dense_timeattn_metrics_20260913.json，489113 bytes，SHA-256：
`f3f66b75ccae759b36c5e52d9d56867b33d473d20e9f26b29e5f3be137abfe16`。
导出包含B4 seed42完整数值审计；该组显式保留CPU/CUDA差异，其余三组best validation
正式指标复现通过。四组input_feature_contract与各自near父对照完全一致。

| 两seed原CUDA指标均值 | B4 | B5 |
|---|---:|---:|
| P | 0.8140060103 | 0.8279482502 |
| R | 0.6954337900 | 0.6826484018 |
| F1 | 0.7500626981 | 0.7482521442 |
| Upcoming R | 0.0137931034 | 0.0137931034 |

B4两颗seed命中均与near父对照相同；B5由1/2变为2/2，尚无实质改善。保留本次
负结果，不采用该时间汇聚候选，也不根据中间epoch或CPU复算替换正式成绩。该结论
只限这项单查询汇聚对照，不代表所有时空注意力无效或模型骨干已被证明无法提前识别。

最终核实训练/诊断pane均dead=1、exit=0，均有对应整批退出0标志，未发现使用该
benchmark的相应训练/诊断Python；服务器HEAD仍为abc6713，tracked clean。未评test、
未启动下一轮训练、未改主仓库。当前upcoming优化目标仍未达到。

## 34. 冻结双头独立阈值可行范围诊断

第32节只检验事件头与onset共用阈值的raw-max组合，不能排除两路分数尺度不同的影响。
本次登记onsetfrontier20260913，仅使用第30节原选定的四份onset_aux best.pt，读取
model_before_farprec20260913.zip内已核验成员；B4/B5各seed42/43、train/validation，
共8份新诊断。现有日志没有逐样本分数缓存，需重新前向以计算新统计；不重训、不重跑
原raw-max曲线、不改变checkpoint或原正式指标，不读取test，不新建目录。

报警集合限定为event>=阈值E，或历史末窗cold且onset>=阈值O；两路均可关闭。
输入仍只有两个预测概率及已观察的hist_hot，起点/时长预测、标签和计分分母保持不变。
遍历全部不同事件分数对应的报告集合，onset按同分组维护精确前缀计数，以整数
5×正确报告数−4×报告数检查P>=0.8。对每个事件阈值取最靠后的可行onset前缀：
upcoming命中数与总命中数随前缀长度非减，因此可求该双标量阈值族的最大upcoming命中。
额外记录P>=0.8且总R>=0.7的可行范围，以及固定原事件阈值时单调节onset的范围。
同分样本不可拆开；实际返回的见证组合另调用canonical report逐项复核。

这是**利用该split真值计算的事后经验上界**，仅针对这一冻结权重和这一报警族。
见证阈值不是实际校准或部署选择，不是无偏验证成绩，也不能推广为新权重、任意融合
函数或骨干架构的上限。如果连这种乐观范围仍很低，才可排除当前权重的简单独立阈值
组合能解决问题；若范围明显改善，还需独立的校准与验证设计，不能直接认领该分数。

实现位于diagnose_baseline_events.py，开关--inspect_onset_frontier，仅增加诊断。
本地25 tests及5 subtests通过，其中80组随机小样本与双阈值穷举比较最大命中数，
覆盖同分、零/一概率、float32阈值边界、历史hot门控、起点错误、无有效节点、无可行点、
不连续的可行前缀和删除后的树状态；原诊断报告及输入数组保持不变。
以上为执行前注册，尚未取得本轮服务器结果；服务器模型模块继续固定abc6713。

### 34.1 部署方式与实际启动

诊断及测试提交f9cc9aa083bc8e19b939dcf885ddf014800b3e2b，源码SHA-256为
`dd06cddd14813c22187ad02611319f936fa0134804b45d97e59af224a169f960`。
服务器仅fetch Git对象，HEAD和模型模块保持abc6713；六个模型/训练相关文件与该
诊断提交相同。服务器内存加载新诊断和测试，四项新测试通过，包括80组穷举核对。
四份onset归档全部成员及原完整导出关键文件哈希、新旧配置和当前manifest均通过。

首次tmux调用因命令过长在启动前被拒绝；确认pane仍为原退出0状态且没有新日志/结果，
捕获同一拒绝原因后，将公共环境和调用参数提取为shell函数，bash语法检查通过。
复用baseline_dense_diag启动成功，pane PID240076、首个实际Python PID240081；
日志onsetfrontier20260913_diagnose.log。失败调用未开始诊断，也没有重跑已完成结果。

### 34.2 八份诊断完成及经验范围

八份诊断正常完成，日志末行为DIAG_BATCH_EXIT_CODE=0。逐份核验归档及best成员哈希、
诊断源码、manifest、epoch1/5/7/5及train23859/validation5439样本数；原分组、排序、
onset辅助头统计和完整保存阈值canonical报告与第30节逐项相同，四份validation原指标
在1e-10容差内一致。所有见证组合另经canonical函数核验，未改正式报警或选择。

| 冻结权重的最大upcoming命中 | B4 s42 | B4 s43 | B5 s42 | B5 s43 |
|---|---:|---:|---:|---:|
| Train，固定原event阈值，P>=0.8（/595） | 23 | 66 | 79 | 53 |
| Train，两个阈值自由，P>=0.8（/595） | 31 | 94 | 97 | 70 |
| Train，两个阈值自由，P>=0.8/R>=0.7（/595） | 无可行点 | 89 | 93 | 63 |
| Validation，固定原event阈值，P>=0.8（/145） | 1 | 7 | 7 | 6 |
| Validation，两个阈值自由，P>=0.8（/145） | 3 | 10 | 10 | 7 |
| Validation，两个阈值自由，P>=0.8/R>=0.7（/145） | 无可行点 | 无可行点 | 无可行点 | 无可行点 |

以下仅是实现返回的最大upcoming命中的一组见证，未优化相同upcoming命中数之间的F1：

| Validation见证指标 | B4 s42 | B4 s43 | B5 s42 | B5 s43 |
|---|---:|---:|---:|---:|
| P | 0.80000000 | 0.80000000 | 0.80021954 | 0.80000000 |
| 总R | 0.55159817 | 0.36164384 | 0.66575342 | 0.53333333 |
| 总F1 | 0.65297297 | 0.49811321 | 0.72681954 | 0.64000000 |
| 误报数 | 151 | 99 | 182 | 146 |

最大validation upcoming召回仅约2.07%/6.90%/6.90%/4.83%。因此对于这些原选定权重，
即使事后使用validation真值挑选双标量阈值，仅靠该阈值族也不能使upcoming达到合理
水平。这比第32节共阈值raw-max的结论更强，但仍不证明新权重、其他融合或骨干无效，
也不将这些乐观见证当作实际校准收益或正式validation成绩。

完整八份结果、原权重引用、模型运行文件哈希及核验状态存
baseline_onset_frontier_metrics_20260913.json，1092127 bytes，SHA-256：
`22c883374b9bc1d2423ddc87a871890f2d417fc28cdfeaef866901a40591d478`。
最终训练/诊断pane均dead=1、exit=0，无对应baseline Python；服务器仍abc6713且
tracked clean。未启动新训练、未评test；当前优化目标仍未达到。

### 34.3 下一步归因范围

补查主参考源码：_cluster_fused只融合hist_cluster，即t-1操作特征按固定规则生成的
状态类别；_fuse_precursor另外加入邻接平均末窗队列。baseline原27维历史X已含生成
这些特征所需的队列、等待、活跃、阻塞、饥饿等字段。因此它们是历史特征的直接编码
路径差异，不能一概称为baseline完全没有观测到这些信号。主dataset中的未来cluster_id
与hist_cluster用途不同，不能将未来cluster_id作为待对齐的推理输入。

下一步应区分完整历史信息的不可辨识性和模型未学会利用信息。历史对称性审计只覆盖
旧13723个train样本内的工位交换；尚未检查当前208包跨窗口、跨episode的完整模型
输入是否相同而未来目标冲突。若开展该检查，哈希应覆盖实际_model_inputs及报警所用
hist_last_hot，不能只比较单个目标工位的局部序列。该后续检查尚未实现或运行，不能
预先声称发现不可预测样本，也不能因没有精确重复而排除近似模糊性或分布差异。

## 35. 当前208包完整观测输入身份审计

登记baseline_input_identity20260913，只检查train/validation。新增工具
audit_baseline_input_identity.py，直接使用实际_model_inputs适配器的全部字段，再加
报警解码所用hist_last_hot：整张图历史X、邻接矩阵、node/target_node掩码、全局历史、
jobs_remaining/jobs_total，以及存在时的event_precursor。未来标签、remain_mask、
occ_node_mask和sample/episode身份不参与输入哈希；occ_node_mask只决定目标统计分母。
标签由现有FactoryBaselineTensorDataset生成，哈希完成后才统计正负冲突，保持节点身份。

哈希包含字段名、dtype、shape和完整数值，正负零视为同一观测值；每个相同哈希组
还重新读取代表样本，逐字段核验张量实际相等。分别记录train内、validation内及
train/validation交集；跨split监督冲突不能当作validation性能上限。输出窗口目标数
和少量可追溯示例，不把相关窗口目标当成独立物理事件。

先检查没有可选摘要的基础输入。若基础输入在train+validation并集中全部唯一，
则附加near/near_far字段不可能把不同基础输入合并，直接记录这项排除证据，跳过
无必要的摘要重建。若发现重复，再调用实际attach_precursor构建near和near_far，
核验两种完整输入，避免把缺少可选字段的粗分组误称为实际模型输入冲突。

审阅发现，当前near/onset/far/timeattention的event_context=false；全局特征和订单
数量只进入其他任务头，target_node_mask在共用头中被忽略。因此同时审计两种口径：
full_adapter保留全部传入字段；event_context_disabled只保留完整图历史X、邻接、
node_mask、可选摘要和解码所用hist_last_hot。分别检查重复，不能用全部字段唯一替代
事件分支口径的结论。该口径仍未归并掩码外数值、图对称或学习表示等价。B4/B5实际
前向检查在none/near两种配置下改变四个被排除字段，四类事件输出逐元素完全相同。

本地7项测试及4个前向子测试通过：逐个实际输入变化都会改变哈希，未来/身份字段变化不改变哈希；
覆盖完整字段、无效目标、跨split与split内冲突的区分、正负零、附加摘要拆分重复组、
强制哈希碰撞后的张量复核、样本覆盖，以及在读取样本前拒绝test。不修改任何模型、
阈值、权重、数据文件或目录；服务器源代码保持abc6713，新审计将由固定Git对象stdin
执行。此处为执行前注册，尚未取得真实数据结果。

### 35.1 双口径真实数据审计完成

审计源码d85212f经Git对象stdin执行，服务器模型模块及HEAD保持abc6713。本地7项
测试、4个实际前向子测试通过；服务器在实际模型模块上也通过全部7项测试。执行前
核验208 manifest、上一项onset frontier完整导出、tracked clean及两项pane退出0；
复用baseline_dense_diag，外壳PID264887，日志input_identity20260913_audit.log。
实际审计已完成，pane退出0，日志AUDIT_EXIT_CODE=0；没有启动训练或重跑旧诊断。

| 口径 | Train样本/唯一输入 | Validation样本/唯一输入 | 并集样本/唯一输入 | 冲突输入组 |
|---|---:|---:|---:|---:|
| 全部模型适配器字段+hist_last_hot | 23859/23859 | 5439/5439 | 29298/29298 | 0 |
| event_context=false事件分支字段 | 23859/23859 | 5439/5439 | 29298/29298 | 0 |

两口径跨split共享输入组均为0；train/validation的ongoing、upcoming、negative分别
严格复现4191/595/297152和950/145/67331。事件口径确认为x、adjacency、node_mask、
hist_last_hot，未把不进入事件头的全局/订单/target_node_mask字段用于区分样本。
两口径基础输入都唯一，因此near/near_far追加摘要不可能产生精确重复，按注册逻辑
跳过摘要重建。该结论仅排除精确完整输入重复，不证明未来可预测，不排除近似模糊、
隐藏外生信息或模型表征压缩，也不是任何模型的性能上限。

完整产物baseline_input_identity20260913.json，10334 bytes，SHA-256：
63b2c5e46cdee806030165fa00c54ab0e6d79e0f2689445f365462454f27288b。
源码SHA为006607ffbf165c89ec7efd6b665a59a32b860283c42bc274b008071f8acb4e2e。
完成后独立重算五个数据文件的全量SHA并与审计前一致；六个运行模块也与abc6713的
Git对象及记录哈希一致。共享tensor包被加载，但未遍历/评价test样本。

## 36. GRU历史表示后的图交互对照

登记history_graph_refine / postgru20260913，以near_precursor为父对照，B4/B5各
seed42/43，从头训练、最多60 epoch。动机是原模型在每分钟先做两层空间图处理，
再逐节点GRU，已编码的工位历史之间没有后续图交换。新增一层同类型GCN/GAT，
作用于last_mean汇聚后的GRU历史表示，检查这条时空交互路径能否改善upcoming。
它不是完整BNPDFormer的多层时空注意力复刻；失败不能证明所有时空交互无效。

新增路径为128维历史→同类型图层64维→LayerNorm/GELU→无bias线性128维→残差。
最后投影初始化为零，额外模块在fork_rng内创建，保留原参数和RNG；不新增随机
dropout，B5新增图层attention dropout为0。初始全部任务输出与父对照逐元素相同，
初次反传更新输出投影，之后梯度可进入图层。B4新增16576参数，B5新增16704参数；
实际总参数需服务器真实配置复核。输出表示仍进入原预测头及原多任务损失。

保留原两层前置GCN/GAT、GRU和last_mean、30分钟历史及near摘要、二分类头、aux
关闭、event_context关闭、均匀抽样、学习率/优化器、早停、阈值、P/R门和选模规则。
这是增加一个图层的明确架构变体，应与原两层baseline分开报告；不能将容量变化
隐去或把该候选称为已验证标准baseline。正式比较仍保留共同204包缺口。

本地验证包括新增5项测试（16子测试），与输入审计/precursor/B5模型共30项（28
子测试）通过；B3/B4、图表示、归档和dense评分相关58项（4子测试）通过。归档测试
中旧temporal_attention的两个断言仍固定last_mean，与原注册不符，修正为其注册
last_attention后26项归档检查通过；未改变该旧候选模型。检查覆盖控制配置/损失、
初始化及训练时RNG、原输出、新增梯度、历史邻居依赖、图边/掩码、节点置换、checkpoint
恢复，以及新增参数。所有旧模型默认关闭该层，无新增状态参数。

启动前先核验timeattention完整导出及四组现有模型关键文件哈希、两项pane退出、
manifest和现有目录。用已有逐成员验证ZIP逻辑归档model_before_postgru20260913.zip，
复用candidate_history/seed42或43目录；无新目录、无test。日志postgru20260913_train.log。
判读两颗seed正式P/R/F1/upcoming、best/last的train/validation排序与误报；不按
upcoming单项挑epoch或拼接预测头。此处仅为注册，尚未启动。

### 36.1 服务器预检与启动

执行前两项pane均dead=1/exit=0，无baseline Python；208 manifest、完整timeattention
导出、input identity产物及四组现有模型的6项关键文件SHA均通过。随后只在BSTAN的
dev_xwt快进到371afb69d29118a3c42b3274a65e5b07509c1032。服务器新增5项测试全部
通过，并在实际train/validation各2个样本上检查四组配置、所有原参数、eval/train
首次输出及RNG一致。只构建near，参数总数B4为273054→289630，B5为285982→302686。

预检产物baseline_postgru_preflight20260913.json，12455 bytes，SHA-256：
609484c93f0809efc7e6a8c73cfcbd5603f50391752c2eb26ece3d78fb84cb53。其中training配置
为CPU预检、profile字段归一到near父对照以比较；说明字段明确实际训练profile为
dense_history_graph_refine_v2、device为cuda:0，不误认为运行配置采用near profile。

复用baseline_dense_v6启动四组串行训练，pane PID272151，首个实际Python272155
（B4 seed42）。日志postgru20260913_train.log；没有重复启动旧任务。首组旧timeattention
ZIP的11个成员逐项核验，6项关键成员与完整旧导出一致。实际config已确认
history_graph_refine=true、near、last_mean、onset_aux=false、evaluate_test=false。
首次观察epoch1：validation F1=0.745137、P=0.820879、upcoming=1/145；只是训练进度，
不是最终或选定结果，也不据此判断新路径有效。当前仅首组已开始，其他组按顺序执行，
各组开始前独立归档；普通模型路径内容会随组次切换，读权重时必须核验来源。

训练期间不得pull后续诊断或文档提交。待每组权重终态后按原口径检查best/last的
train/validation，继续核验归档、模型实际配置、来源及最终指标。

### 36.2 新图路径只读诊断

新增diagnose_baseline_events.py --inspect_history_graph：在一次原始eval forward
中hook实际history_graph输入/输出，记录掩码后原历史表示、更新后表示和实际残差
的L2范数，以及初始为零的输出投影当前L2范数。按既有ongoing/upcoming/negative
分组记录q10/q50/q90与相对更新量；相对分母下限1e-8并记录低于下限的数量。未来
标签只用于事后分组，不影响forward或诊断量。非零更新只能证明路径活跃，不能
当作有益信息或因果贡献，更不能代替train/validation泛化比较。

新增3项检查通过，覆盖实际B4/B5单次forward输出/参数/RNG完全不变、初始零更新、
非零更新、掩码、hook异常清理、分组统计/空支持/非法值和CLI；与原诊断及新模型
合计33项测试、23个子测试通过。新观察与旧temporal_attention观察互斥；当前新
候选使用last_mean，原时间汇聚候选仍按原开关诊断，不改变其预测。

待每组训练完成并核验record/config/权重后执行best/last × train/validation，
保持原报告和排序口径、CPU2线程，无test。用Git对象stdin执行新增诊断，不在当前
训练中pull源码；底层模型模块保持371afb6。输出登记为
b4_seed42_train_diagnostics_postgru20260913.json等，last权重在diagnostics前加last_。
本节为准备状态，真实权重诊断尚未运行。

### 36.3 B4 seed42训练与四份冻结诊断完成

首组正常完成，best epoch1、total11，正式validation P/R/F1为
0.8208791208791208/0.6821917808219178/0.7451371571072319，阈值0.55，
upcoming为1/145（0.006896551724137931）。相同seed的near父对照为2/145，
F1=0.74531095755，本seed没有upcoming改善；不将单seed结果外推整批。

验证record/config/summary、7项现有文件SHA和旧timeattention归档11成员后，
冻结输入快照baseline_postgru_b4s42_diagnostic_inputs20260913.json（29175 bytes）。
新增观察源码5cfce2d65074860d17e454dc7c32f7375d294312，SHA为
704a4fc94bf6951bb12e39148cd3a9b9f3dd51126d6f259f6bdfe4e1c41f4bf1；服务器3项
定向检查通过。源码经Git对象stdin执行，底层模型模块及HEAD始终为371afb6。
复用baseline_dense_diag，pane PID278553，首个诊断Python278558，日志
postgru20260913_b4s42_diagnose.log。四份best/last × train/validation均正常完成，
pane退出0，日志DIAG_BATCH_EXIT_CODE=0。

| 观察项 | Best train | Best validation | Last train | Last validation |
|---|---:|---:|---:|---:|
| upcoming-vs-negative AP | 0.005445 | 0.005887 | 0.054607 | 0.010036 |
| upcoming图残差/原历史表示L2，中位数 | 1.026171 | 1.071347 | 0.475858 | 0.488301 |
| 输出投影权重L2 | 2.849632 | 2.849632 | 3.598016 | 3.598016 |

表中诊断值保留六位小数，精确值在完整导出。权重/诊断源码/manifest/epoch和样本数
均核验；train/validation目标分母严格复现，图观察分组与原事件分组一致。Best
validation的P/R/F1、ongoing/upcoming recall、who指标及事件计数与原保存正式
结果在1e-10内相同。该核验没有声称CPU/GPU的连续时长误差逐位相同。六个实际
运行模块逐文件与371afb6 Git对象相同，近窗父对照导出哈希也再次通过。

新路径不是零初始化后未参与计算，但非零或较大更新不等于有效提前信号。后期训练
排序有所改善，validation改善有限，所选best仍只命中1/145；不采用首seed为替代
方案，也不据此宣称骨干上限。B4 seed43已接续且实际Python276207仍活跃，B5两组
尚未开始；不得重复启动批次或把普通目录中的不同阶段权重混淆。

完整首组导出baseline_dense_postgru_b4s42_metrics_20260913.json，548096 bytes，
SHA-256：034447da244ab18f376471a3c2dd18960dd284a8db76cdf640d91f14607e68b9。
含完整训练快照、四份诊断及保存阈值报告、原文件与运行模块哈希、近窗父对照来源。
本轮无test，未更改阈值或checkpoint选择规则。

### 36.4 B4两颗seed与八份诊断全部完成

B4 seed43正常结束，best7/total17，正式validation P/R/F1为
0.8263736263736263/0.6867579908675799/0.7501246882793018，阈值0.70，
upcoming=1/145。旧timeattention归档11成员及其关键文件与旧完整导出逐项相符；
实际配置仍near/last_mean/refine开启/aux关闭/test关闭。冻结训练输入快照
baseline_postgru_b4s43_diagnostic_inputs20260913.json，29166 bytes。

复用已退出诊断pane，PID283768，用相同5cfce2d源码及371afb6模型模块完成四份
best/last train/validation诊断，pane退出0，日志postgru20260913_b4s43_diagnose.log
以DIAG_BATCH_EXIT_CODE=0结束。首次核验时只完成3/4，等待同一会话终态后再核验，
没有重复运行诊断。四份权重/源码/manifest/epoch/样本数/分组分母均通过，best
validation的离散计数及P/R/F1与原保存指标在1e-10内相同；六个运行模块与371afb6
Git对象一致，近窗父对照完整导出哈希也通过。

| Seed43观察项 | Best train | Best validation | Last train | Last validation |
|---|---:|---:|---:|---:|
| upcoming-vs-negative AP | 0.018874 | 0.009939 | 0.198485 | 0.008768 |
| upcoming图残差/原历史表示L2，中位数 | 0.560887 | 0.568791 | 0.465529 | 0.466489 |
| 输出投影权重L2 | 3.549455 | 3.549455 | 4.365580 | 4.365580 |

诊断数值保留六位小数。完整第二颗seed导出
baseline_dense_postgru_b4s43_metrics_20260913.json，587794 bytes，SHA-256：
37e95077af853d69d223a20a7768522ca142aad02da99f97e49bd73157ca7bc7。

核验两颗seed完整导出及其8份原诊断文件哈希后，按同seed匹配near父对照，汇总为
baseline_dense_postgru_b4_metrics_20260913.json，1334176 bytes，SHA-256：
e8c5f415bc16b823275077166cc6f933b3bd1997b20e354dd7369b145817118b。
保存完整两颗seed结果、父对照来源、逐seed差值及均值，不混入其他权重或epoch。

| B4两seed均值 | Near父对照 | GRU后图交互 |
|---|---:|---:|
| Precision | 0.8220531728 | 0.8236263736 |
| Recall | 0.6913242009 | 0.6844748858 |
| F1 | 0.7510203841 | 0.7476309227 |
| Upcoming recall | 0.0137931034 | 0.0068965517 |

两颗seed的upcoming都从2/145降至1/145，F1均值下降0.00338946，因此不采用B4
本次新增图层候选。图残差实际非零，不能将负结果归于路径没有启动；两颗seed仍有
训练/验证排序差距。它只说明这一受控路径在当前配方下没有帮助，不证明全部时空
交互或GCN-GRU无效。B5 seed42实际Python282818在运行，配置和旧归档11成员已
核验，首次复核观察epoch5；seed43尚未启动。训练pane272151保持live，模型HEAD
371afb6不变，未使用test，待B5完成后才能判断整批。

### 36.5 B5 seed42训练及四份诊断完成

B5 seed42已正常完成，best7/total27，总参数302686，validation正式P/R/F1为
0.8359375/0.6840182648401827/0.7523857358111502，阈值0.60，upcoming为0/145。
旧timeattention归档11成员及其中关键文件与旧完整导出相符；实际配置保持
near/last_mean/refine开启/aux关闭/test关闭。训练快照
baseline_postgru_b5s42_diagnostic_inputs20260913.json为29041 bytes。

复用诊断pane，PID295800，四份best/last train/validation诊断正常退出0；日志
postgru20260913_b5s42_diagnose.log末行为DIAG_BATCH_EXIT_CODE=0。核验四份权重
哈希、诊断源码5cfce2d、manifest、epoch7/27、样本数23859/5439及事件分组分母；
best validation离散计数及P/R/F1与原保存正式指标在1e-10内相同。六个实际运行
模块仍与371afb6 Git对象一致，服务器tracked clean，近窗父对照导出哈希也通过。

| B5 seed42观察项 | Best train | Best validation | Last train | Last validation |
|---|---:|---:|---:|---:|
| upcoming-vs-negative AP | 0.010409 | 0.006376 | 0.422239 | 0.008441 |
| upcoming图残差/原历史表示L2，中位数 | 0.698670 | 0.839081 | 0.439568 | 0.530445 |
| 输出投影权重L2 | 3.661362 | 3.661362 | 4.945151 | 4.945151 |

观察值保留六位小数。最后权重的训练排序明显提高，验证排序仍低；图交互路径实际
参与计算，但此seed没有upcoming收益，不能据此外推所有图时空编码的上限。
完整导出baseline_dense_postgru_b5s42_metrics_20260913.json，548987 bytes，SHA-256：
bddea7fa14a9e665fa96529155c89addf158a1c3cc9d19116ea897c996479334。

B5 seed43已接续，实际Python294501和训练pane272151仍live；四组训练和16份诊断
尚未全部结束。不要重复seed42已完成的四份检查，不在训练中更新371afb6模型模块。

### 36.6 B5 seed43完成，四次训练及16份诊断全部核验

最后一组B5 seed43正常完成，best10/total30，总参数302686；正式validation
P/R/F1为0.8372615039281706/0.6812785388127854/0.7512588116817726，阈值0.55，
upcoming为2/145。训练pane272151退出0，日志末行为TRAIN_BATCH_EXIT_CODE=0；
实际Python294501已结束。四次训练均正常结束，没有重启、换epoch或使用test。

原timeattention归档11成员逐一核验并与旧完整导出匹配，配置near/last_mean/refine
开启/aux关闭/test关闭。冻结快照baseline_postgru_b5s43_diagnostic_inputs20260913.json
为29275 bytes。诊断pane308378（首个Python308383）串行完成best/last的train与
validation四份诊断；观察到last train/validation进程310127/311542，等待同一会话
正常退出0后核验，未重复执行。日志postgru20260913_b5s43_diagnose.log以
DIAG_BATCH_EXIT_CODE=0结束。

| B5 seed43观察项 | Best train | Best validation | Last train | Last validation |
|---|---:|---:|---:|---:|
| upcoming-vs-negative AP | 0.031092 | 0.010234 | 0.458472 | 0.008134 |
| upcoming图残差/原历史表示L2，中位数 | 0.538978 | 0.554987 | 0.433804 | 0.489656 |
| 输出投影权重L2 | 4.012980 | 4.012980 | 5.271747 | 5.271747 |

观察值保留六位小数，精确值在完整导出。四份来源/权重/manifest/epoch/样本数和
分组分母核验通过，best validation离散计数及P/R/F1与原正式指标在1e-10内一致。
六个运行模块仍为371afb6，诊断源码仍5cfce2d；未将连续时长误差的设备差异纳入
逐位相同的声明。完整最后一组导出baseline_dense_postgru_b5s43_metrics_20260913.json，
568743 bytes，SHA-256：
47f465d9ed12656853320fd6142ca134a6b0b4b0e8d7ea093f60c8de128e1507。

### 36.7 整批汇总与不采用决定

训练全部结束后重算dataset_manifest.json、dataset.pt、model_sample_index.csv、
split_manifest.json及normalization.json的全量SHA，五项均与已固定的208包输入审计
一致，哈希期间文件size/mtime未变化；该操作只读文件字节做校验，没有评价test样本。
核验产物baseline_postgru_data_integrity20260913.json，1423 bytes，SHA-256：
d697c36016147a70b874a1cd56e3aa38ecfe5346d25ff081e5c6df746ed33a78。

逐组重验四个完整导出及其16份原诊断、冻结训练快照、当前七个权重/配置/结果文件，
四份旧timeattention归档各11成员、近窗父对照、运行源码，以及上项数据完整性记录。
两项pane均dead=1/exit=0，未发现该benchmark的训练或图观察诊断Python。最终汇总
baseline_dense_postgru_metrics_20260913.json为2654014 bytes，SHA-256：
db6a3e6a004426404820a774c428da3f862805bd94d4312c1db457f2a6271312。
其中保留四组完整训练/诊断、逐seed配对差值、模型内两seed均值及全部核验来源。

| 模型／两seed均值 | Near父对照 | GRU后图交互 | 差值 |
|---|---:|---:|---:|
| B4 Precision | 0.8220531728 | 0.8236263736 | +0.0015732009 |
| B4 Recall | 0.6913242009 | 0.6844748858 | -0.0068493151 |
| B4 F1 | 0.7510203841 | 0.7476309227 | -0.0033894614 |
| B4 Upcoming recall | 0.0137931034 | 0.0068965517 | -0.0068965517 |
| B5 Precision | 0.8381670772 | 0.8365995020 | -0.0015675752 |
| B5 Recall | 0.6757990868 | 0.6826484018 | +0.0068493151 |
| B5 F1 | 0.7482362744 | 0.7518222737 | +0.0035859993 |
| B5 Upcoming recall | 0.0103448276 | 0.0068965517 | -0.0034482759 |

四组upcoming命中由near的2/2/1/2变为1/1/0/2（各145个目标）。B5平均F1虽略升，
但当前目标的提前召回没有改善；B4平均F1也下降，因此不采用本次新增图层。最后权重
train/validation AP差距在两模型两seed均保留，图残差活跃不等于可靠提前征兆。这只
否定当前固定的图交互配方，不证明所有时空交互或GCN/GAT-GRU都无法预测upcoming。
整批已结束并保留完整结果，下一项event_fbeta仍以near为父对照，不能叠加本轮图层。

## 37. Upcoming soft F-beta损失单项对照准备

登记event_fbeta / fbeta20260913，父对照near_precursor，B4/B5各seed42/43，最多
60 epoch。它检查主参考配方中的事件损失形式，而非增加新骨干：原两层GCN/GAT、
GRU、last_mean汇聚、二分类事件头、近窗摘要及原参数量273054/285982均保持。
history_graph_refine和onset_aux均关闭，远历史5维仍为零。该项不能与上一轮新增图层
混在一起归因，也不称为主模型完整训练配方的复刻。

固定主参考20c40e230aedee6aef2429d352413fbcf0fa571a，采用其all-event与
upcoming-vs-negative soft F-beta的等权组合，beta=1.5，事件损失内系数0.8：

`L_new = L_near + lambda_event_will * 0.8 * (1 - 0.5 * (F_beta_all + F_beta_upcoming))`

保留baseline现有BCE权重并detach用于该项，upcoming子项排除ongoing。batch中
没有upcoming时只用all-event项；没有任何有效加权节点时返回可微零，避免把padding
变成观测。后一边界刻意不同于主源码的全零权重转全一兜底，正常有效节点下的公式及
梯度已与主源码块核对。原lambda_event_will=2.5，因此新增项有效系数为2.0；主配置
相应为2.4*0.8=1.92。其他正负/资源类型权重、BCE、focal=0、辅助损失及抽样均不变，
不能把这一项称为主损失全部对齐。

训练保持从头初始化和均匀抽样，B4 batch24/lr3e-4/min10/patience10，B5
batch16/lr1.5e-4/min15/patience20，weight_decay=0.01。阈值、事件定义和checkpoint
选择沿用near父对照，不根据upcoming单项改选epoch，不使用test。soft F-beta只是
可微训练目标，并不保证硬阈值下precision>=0.8或提高validation recall。

实现涉及torch_losses.py和train_dense_baseline_control.py，默认系数0时旧损失
计算路径不变。新增test_baseline_event_fbeta.py的六项测试覆盖：从固定主Git对象
提取实际公式块，对16组随机输入逐项比较数值与梯度；upcoming梯度方向和beta权衡；
padding/无正例/无有效节点及极端logit；B4/B5真实forward中的梯度通路；四组实际
控制配置只改变损失系数；非法配置和目标网格拒绝。与onset、归档及B4/B5模型相关
检查合计62 tests、10 subtests通过。F-beta项的梯度到达GRU和事件头，不直接改变
起点或hot头；参数、RNG以及原损失分项保持不变。

当前仅完成本地实现及登记，尚未部署或训练。服务器继续保持371afb6。只有当前图
交互四次训练、16份诊断和完整汇总全部终态并核验后，才进行新源码部署及四组真实
数据预检；复用原目录，逐成员验证model_before_fbeta20260913.zip后再使用正常权重
路径。不得覆盖未核验的当前结果，不重新执行已完成的near/onset/far/timeattention
或阈值搜索。评判同时看两颗seed正式P/R/F1/upcoming、误报及best/last训练与验证
排序；失败只否定这个固定单项配方，不宣称GCN/GAT-GRU普遍无法提前预测。

### 37.1 固定源码的服务器独立进程测试

实现和登记已提交07398bb308b2ef9df8d15b0afca8bf7fd146c7e9，并push至dev_xwt。
服务器只fetch对象，在独立CPU进程中通过Git对象执行新loss/control/test源码，
将该进程的package与trainer函数引用同步至新损失；未改训练进程或工作区文件。
新增六项unittest全部通过，包含实际主参考公式与梯度核对。执行前后服务器HEAD
保持371afb6，六个loss/head/model/trainer/control文件逐一与371afb6对象及原哈希
一致，tracked clean。未加载真实数据或启动新的训练；这不是部署后的真实数据预检。

完整核验产物baseline_fbeta_server_tests20260913.json，2056 bytes，SHA-256：
8bf745a586d5495429b94a16e20e19f90a41b3d573aa2f27cad908cfcca5b88c。
其中保留三个新Git源码文件及六个未改运行文件的哈希、测试结果和执行方式。当前B5
seed43实际Python294501继续训练，最近观察epoch21；仍按37节的完整批次收尾门槛
决定何时部署，不重复已通过的测试。

### 37.2 完整前序结果核验后部署固定版本

通过UU的Leo独立终端SSH，再次确认BSTAN/dev_xwt、旧运行HEAD371afb6、两项pane
退出0且无该benchmark Python。逐组复核四份完整图交互导出、16份原诊断、当前七个
文件、旧timeattention归档，并重新计算五个冻结数据文件SHA；全部通过。
新版本1728ca69bd85e695f6004522ff774a20eaaab8e4中的三个F-beta源码/测试文件与
服务器独立进程测试的07398bb对象逐项相同，之后才fast-forward至1728ca6。
部署后六个骨干/训练器/数据/摘要模块与图交互批次相同，tracked clean。

部署核验产物baseline_fbeta_deployment20260913.json，1283 bytes，SHA-256：
25af6c82eb3f61cf16b67d365c78e99464ea5f88b957c18a72d7ef108e0f95ac。
新训练的运行源码固定1728ca6，不能将之后仅更新文档的commit混作训练源码。

### 37.3 真实数据预检与序列化比较修正

在现有benchmark保存可复现预检脚本baseline_fbeta_preflight20260913.py，复用
baseline_dense_diag。首次完整near构建已通过，但在比较旧JSON训练配置与新dataclass
时停止：report_threshold_sweep的数值完全相同，旧JSON是list，内存默认是tuple。
这不是阈值、训练配方或模型差异。只修正预检：先逐项确认阈值序列相同，再把容器
规范为list，同时仅归一化实验profile名称；实际训练配置仍完整原样保存在预检输出。

首次pane320936退出1，脚本与日志经既有archive_files逐成员哈希验证后归档为
baseline_fbeta_preflight_serialization_guard20260913.zip，两个原文件，4166 bytes，
SHA-256：57854d70425d71010105421d46f0bb7f8b5614e681209a6fe32d20792e024c96。
没有覆盖失败记录或模型权重。修正后的同名脚本SHA-256为
e0ec3c399beeede5a47e6e4b205b6dd62d45e296f25c996890cb414c1bc20189；
复用已退出pane，PID324031，正常退出0，日志以PREFLIGHT_EXIT_CODE=0结束。

完整调用实际attach_precursor，为train/validation全部29298个样本构建near摘要，
与四组near父对照的输入契约逐项相同，远历史5维为零。随后从train和validation各取
首个含upcoming与首个不含upcoming的样本：train全局索引18/0、validation索引
2012/1979；两个upcoming样本各有一个目标。样本按边界类别确定，不按预测结果挑选。

四组实际配置B4/B5×42/43全部通过：模型参数量273054/285982，参数及初始化RNG
相同；train/eval模式全部输出逐元素相同；原损失分项相同，仅新增登记的F-beta项。
新增项的梯度到达GRU、事件头和摘要投影，未直接给起点/hot头梯度，backward后模型
参数未改变。CPU用于预检，登记训练设备仍为cuda:0；没有optimizer step或test样本
评价。原始共享张量包被加载，但只构建和选取train/validation样本。运行HEAD及五个
数据文件的size/mtime在预检期间保持不变；全量SHA已在部署前重新核验。

完整预检baseline_fbeta_preflight20260913.json，16640 bytes，SHA-256：
eea6a7bd545063b22d7d34daa6e60e06985c72ea4e40cb83ca558d3c5c272235。
保留四组完整model/training/loss配置、输入契约、样本身份、梯度及失败归档引用。

### 37.4 四组串行训练已启动

确认预检及其脚本哈希、两项pane退出0、无相应Python和新tag文件后，复用
baseline_dense_v6启动B4 s42/s43、B5 s42/s43四组串行训练。启动前GPU0总32607 MiB，
空闲28696 MiB；没有停止其他任务。运行固定1728ca6，driver每组开始前核验branch、
HEAD和旧图交互结果文件，完成后核验实际配置/分母及归档成员。

串行脚本baseline_fbeta_driver20260913.py，SHA-256：
ef9ac53dced60f4beebcf4b21a0f8f0dcef2b126a010d2f5127cafe7be929068。
日志fbeta20260913_train.log，pane PID326665，实际driver Python326668，首组模型
Python326721（B4 seed42）。首组旧model_before_fbeta20260913.zip的11个成员已
核验并与图交互完整导出匹配；实际config与预检逐项相同，F-beta0.8/beta1.5、near、
last_mean、refine关闭、aux关闭、test关闭。

首次配置核验观察到epoch2，validation日志P/F1为0.817391/0.746402，upcoming为0；
这只是中间进度，不是最终成绩。其余三组尚未开始，普通权重路径会依次转为F-beta，
不能据目录名误认来源。训练中不pull后续文档提交；保持原阈值/选模，待每组终态后
做best/last × train/validation冻结诊断，不重复已完成的图交互或其他搜索。


### 37.5 B4两次训练与八份诊断完成

通过UU/Leo终端重新读取完整B4汇总的成功输出，两个seed均VERIFIED，未重跑训练或
诊断。B4两组普通权重、冻结快照、八份原诊断、八个运行源码文件与1728ca6、预检配置、
近窗父对照导出以及旧图交互归档各11成员均已核验。Best validation的14项事件计数、
P/R/F1、ongoing/upcoming及who指标与原正式保存值在1e-10绝对容差内相同；不将CPU与
CUDA的连续时长输出宣称为逐位相同。所有诊断仅用train/validation。

| B4 F-beta正式validation | seed42 | seed43 |
|---|---:|---:|
| Best / total epoch | 3 / 13 | 6 / 16 |
| 参数量 | 273054 | 273054 |
| P | 0.8353591160220994 | 0.828693790149893 |
| R | 0.6904109589041096 | 0.7068493150684931 |
| F1 | 0.756 | 0.7629374075899458 |
| Upcoming命中（各145） | 0 | 2 |
| 原保存阈值 | 0.88 | 0.98 |
| 相同seed近窗F1 | 0.7453109575518262 | 0.7567298105682951 |
| 相同seed近窗upcoming命中 | 2 | 2 |

两颗seed均值P/R/F1/upcoming R为0.8320264531/0.6986301370/0.7594687038/
0.0068965517，近窗父对照为0.8220531728/0.6913242009/0.7510203841/0.0137931034。
平均F1增加0.0084483197，但upcoming命中下降，因此保留总体指标收益，不能将其
认定为upcoming解决方案，也不换用last权重或诊断阈值作为正式结果。

| Upcoming排序AP及保存阈值诊断 | s42 best train | s42 best val | s42 last train | s42 last val | s43 best train | s43 best val | s43 last train | s43 last val |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AP | 0.007035 | 0.006204 | 0.176375 | 0.012447 | 0.014783 | 0.008983 | 0.235693 | 0.008248 |
| 概率漏报 | 595 | 145 | 462 | 137 | 582 | 143 | 425 | 140 |
| 时间漏报 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 误报 | 488 | 149 | 378 | 176 | 538 | 160 | 308 | 152 |

AP转录至六位小数，完整原值保存在服务器导出。训练/验证目标分别595/145；last训练
排序明显提高、验证没有同步提高，仍支持泛化差距。已计算的有限诊断阈值网格在P>=0.8
下最大观察命中依次为1/0/166/8、29/2/203/5。该统计不是精确阈值上界、部署策略或
正式成绩，不能与第34节的冻结双头精确范围混淆。

Seed42诊断pane333459、seed43诊断pane340516均退出0，对应日志均以
DIAG_BATCH_EXIT_CODE=0结束。两个输入快照baseline_fbeta_b4s42_diagnostic_inputs20260913.json
和baseline_fbeta_b4s43_diagnostic_inputs20260913.json分别30866/31083 bytes。
完整单seed导出及SHA-256：

- baseline_dense_fbeta_b4s42_metrics_20260913.json，579386 bytes，
  9396dc112dea1d7cb2dcc30ca14058e1016e5502734f58f0bf6c8916f531bea9。
- baseline_dense_fbeta_b4s43_metrics_20260913.json，562239 bytes，
  078534f21146f169e4e069ee33135efdb607ed381375addcedefff9411c6fcc6。
- baseline_dense_fbeta_b4_metrics_20260913.json，1343096 bytes，
  5b3fbc9ce9df2a0fbc08e9c9eab3cd6fc955734c2bc4ba734ceda89d86495f1f。

本次实时检查训练pane326665和driver326668仍live，B5 seed42原Python337927运行，
日志最新epoch25；B5 seed43尚无新tag记录。诊断pane340516为dead1/exit0，服务器
HEAD1728ca6。不中断原进程、不在训练中pull；B5最终结果尚待收尾。

### 37.6 B5 seed42训练完成及冻结诊断启动

继续读取同一训练会话：B5 seed42已validation_completed，best7/total27，参数285982；
P/R/F1为0.8477043673012318/0.691324200913242/0.7615694164989939，阈值0.80，
upcoming为0/145。B5 seed43实际Python350531已进入训练，driver326668和pane326665
仍live；未重新启动训练。

仅对完成的seed42核验实际配置与四组真实预检、near输入契约、manifest、1728ca6的
八个运行文件、诊断源码704a4fc9及旧图交互归档11成员，通过后冻结七个当前文件的
SHA和完整训练记录。快照baseline_fbeta_b5s42_diagnostic_inputs20260913.json，
30956 bytes。随后复用已确认dead1/exit0的诊断pane，新pane351899、dead0；标准
best/last × train/validation四份诊断已启动，CPU2线程、batch32，不启用图或时间
观察器。日志fbeta20260913_b5s42_diagnose.log。当前尚未核验四份终态输出，不能
把诊断启动当作完成；服务器继续保持1728ca6，训练中不pull。

### 37.7 B5 seed42四份诊断完整核验

B5 seed42诊断pane351899已dead1/exit0，无对应诊断Python。四份best/last ×
train/validation已逐份核验权重、训练快照、1728ca6源码、manifest、样本数和分组；
best validation的14项canonical计数及P/R/F1/who指标与原保存正式值在1e-10内一致。
没有启用图/时间观察器或读取test。旧图归档和真实预检匹配记录均保留在完整快照。

| B5 seed42 upcoming诊断 | Best train | Best validation | Last train | Last validation |
|---|---:|---:|---:|---:|
| 排序AP | 0.013628 | 0.008034 | 0.359266 | 0.009133 |
| 保存阈值下概率漏报 | 586 | 145 | 373 | 139 |
| 时间漏报 | 0 | 0 | 0 | 0 |
| 误报 | 438 | 136 | 250 | 154 |
| 既有有限网格P>=0.8最大观察upcoming命中 | 30 | 2 | 258 | 6 |

最后一行只是现有离散诊断网格统计，不是精确上界、部署策略或正式成绩。AP转录至
六位小数，训练/验证目标595/145。训练后期排序改善而验证仍弱，不能用last替代
原选定best。相同seed近窗P/R/F1为0.8459770114942529/0.6721461187214612/
0.7491094147582696、upcoming1/145；F-beta正式F1增0.0124600017，但upcoming降至0。

完整导出baseline_dense_fbeta_b5s42_metrics_20260913.json，542226 bytes，SHA-256：
ca85e5b0014532648b5ee4a7b2777f71014fc89c36c02f1fd3a3d47b0e192676。
最新实际检查B5 seed43原Python350531、训练pane326665及driver326668仍live，日志
epoch26；服务器保持1728ca6，不能重复已完成诊断或在训练中pull。

### 37.8 四次F-beta训练全部完成，最后四份诊断启动

原训练pane326665现已dead1/exit0，日志末行TRAIN_BATCH_EXIT_CODE=0；driver和
模型Python均已退出。B5 seed43正式记录validation_completed，best7/total27，
P/R/F1为0.8388157894736842/0.6986301369863014/0.7623318385650224，upcoming
0/145，阈值0.80，参数285982。整批正式upcoming为0/2/0/0，近窗对照2/2/1/2；
四组均未获得upcoming提升，不以总体F1增加宣称任务解决。

在服务器仍为1728ca6且tracked clean时，核验最后一组实际配置与预检、near契约、
八个运行源码、manifest、旧图归档11成员，再冻结七个当前文件。快照
baseline_fbeta_b5s43_diagnostic_inputs20260913.json，31008 bytes。复用已确认
退出0的诊断pane，最后best/last × train/validation四份标准诊断启动成功，新pane
365017、dead0；日志fbeta20260913_b5s43_diagnose.log。此前12份已完成核验，最后
四份尚未终态，整批完整导出仍待完成。服务器不pull本地联合路径草稿。

### 37.9 F-beta整批16份诊断及完整汇总完成

B5 seed43最后四份诊断已正常结束，pane365017退出0；最后实际观察的train/last
Python366727已退出。每份原JSON、冻结输入快照、权重和源码哈希、样本数/分母、
保存阈值及best validation的14项canonical指标均通过核验，源码仍1728ca6。

| B5 seed43 upcoming诊断 | Best train | Best validation | Last train | Last validation |
|---|---:|---:|---:|---:|
| 排序AP | 0.016703 | 0.008830 | 0.354932 | 0.009689 |
| 保存阈值下概率漏报 | 574 | 145 | 392 | 142 |
| 时间漏报 | 0 | 0 | 0 | 0 |
| 误报 | 464 | 147 | 234 | 130 |
| 既有有限网格P>=0.8最大观察upcoming命中 | 42 | 0 | 245 | 3 |

AP转录至六位小数，最后一行不是精确上界、部署策略或正式分数。训练/验证排序差距
依旧，未把last权重的诊断替换为正式best。单seed完整导出
baseline_dense_fbeta_b5s43_metrics_20260913.json，561403 bytes，SHA-256：
a04cb6069018855621d955c6a1c5cca61f799eb6eb89a8419cf6973fe5ce9bc4。

整批汇总前再次确认两项pane均dead1/exit0，无该benchmark训练/诊断/driver Python，
工作区tracked clean、HEAD1728ca6。四份完整单seed导出、16份原诊断、四个冻结快照、
28个当前训练文件及各自八个运行源码均复核；四份旧图交互ZIP各11个成员的CRC和
SHA与原图交互完整导出一致。五个冻结数据文件（包括约4.58GB的dataset.pt）重新
全量计算SHA并与之前完整性基准逐项相同；另全量核验episodes.npz与四组近窗输入
契约的bundle SHA相同。整文件哈希不属于test样本评价，没有使用test调参。

| 两seed均值 | B4 near | B4 F-beta | B5 near | B5 F-beta |
|---|---:|---:|---:|---:|
| P | 0.8220531728 | 0.8320264531 | 0.8381670772 | 0.8432600784 |
| R | 0.6913242009 | 0.6986301370 | 0.6757990868 | 0.6949771689 |
| F1 | 0.7510203841 | 0.7594687038 | 0.7482362744 | 0.7619506275 |
| Upcoming R | 0.0137931034 | 0.0068965517 | 0.0103448276 | 0.0000000000 |

完整导出baseline_dense_fbeta_metrics_20260913.json，2642311 bytes，SHA-256：
3d375d44317934c1e6e0f3670bc435299e6216699ec419989d39a834d10abf28。
它包含四份完整运行/诊断、同seed比较、两seed均值、旧归档成员和六个数据文件的
SHA及size/mtime。F1均值增加约0.00845/0.01371，但upcoming由2/2/1/2降为0/2/0/0。
保留总体指标收益作为独立结果，不采用F-beta作为upcoming解决方案，不宣称骨干上限。
当前目标仍未完成；此批训练、诊断和收尾已全部结束，不再重复。

## 38. 主模型联合onset报警训练路径的源码差异

再次只读检查固定主参考20c40e230aedee6aef2429d352413fbcf0fa571a的model.py和
FactoryBN_dense_f1_p80.json。配置split_will_heads=true、event_cold_will_max=true、
event_will_use_continue_main=false。model.py的_combine_will_logit（965–985行）在
历史末窗hot时使用continue分数，cold时取continue与onset的maximum；forward
（1260–1268行）将组合值写入event_will_logit，事件损失（1430–1434行）直接读取
该组合值。因此该参考方案在训练时已经让主事件损失约束组合后的报警分数；对cold
负例，若onset分数较高，它会收到主事件损失的误报梯度。两分数相等时的梯度分配
由torch.maximum定义，不能笼统说任何时刻两头都收到主损失梯度。

Baseline的onset_aux实现则将event_onset_logit作为独立辅助输出，event_will_logit
仍由原事件头生成。第32节raw-max和第34节独立双阈值均在已冻结权重上进行，未按
组合输出重新训练。故既有负结果不能排除联合训练该报警路径的收益；同时源码差异
本身不是收益证据，实际主204-episode权重与覆盖配置仍未核验。

此处只记录机制核查，未新增训练候选、未叠加F-beta或修改服务器运行代码。先完成
当前F-beta四组及16份诊断，再依B5结果决定是否登记保持GCN/GAT-GRU、共同208数据/
输入/阈值/选模的联合onset单项对照。原骨干已有事件发生、未来起点、时长和hot预测
头，不能解释成没有未来输出；训练排序改善而验证弱也不能证明骨干已经达到上限。


### 38.1 本地联合路径准备及历史状态输入边界修正

在B5 seed43继续原F-beta训练期间，本地准备可选event_onset_joint路径和定向检查，
尚未注册新的dense训练variant、未部署服务器。原有onset_aux权重/参数数量及初始化
RNG完全保留；新开关要求binary和独立onset辅助监督，训练与评估都对冷状态组合值
使用原主事件损失。辅助监督、起点/时长/hot损失、原阈值/选模未改变。

实现审阅重新查到第16/17节已有边界：legacy hist_last_hot由整局ops_hot_mask平滑
生成，可能依赖未来短段是否持续满门槛。因此不能因字段名为“历史”或原解码已经使用它，
就把它新增为网络输入。最初仅本地测试的草稿接了该字段，现已移除；没有部署或训练。
对话中“可直接复用过去信息”的表述在核查后已明确纠正。

新的event_history_hot与旧字段分开：onset_history.py仅将已有30×N×27编码X的
前21个连续通道按冻结训练归一化反变换，在这30个已观测窗口内调用共同ops_hot_mask
（60秒、min8、gap1）并取末帧，最后按node_mask屏蔽。保留编码后的缺失观测值，
不另读raw补全；不读取更早历史、future、y_hot或旧hist_last_hot，也不修改冻结包、
目标、原解码或评分。因此它不是主参考整局历史标志的完全复刻，正式在线因果性仍
受共同旧解码协议限制，不能宣称已修复整个benchmark。

派生字段只在需要的split内存构建，配套valid掩码阻止使用未构建样本；checkpoint
保存单独onset_history_contract，包括输入范围、归一化、manifest、构建器和operational
规则源码哈希，重评/诊断缺失契约或不匹配即拒绝。原near摘要构建源码未变，旧权重
默认joint=false，不添加新输入或改变原输出。

本地57 tests、17 subtests通过；3个需要临时目录的既有归档/目录测试未运行，本次
没有改动其逻辑。新测试覆盖固定主_combine_will_logit源码的12组数值/BCE梯度与同分
边界、两种骨干参数/RNG/初始输出一致、cold负例的主事件损失确实更新onset、hot/
无效节点门控、实际一轮训练与评估共同使用组合分数、checkpoint/旧权重往返、未来
标签及旧hist_last_hot改变不能影响输入、30窗截断的前视反例、只构建train/validation
和缺失观测保持。真实208数据构建与四组配置预检尚待批次收尾后完成，未声称提升。


### 38.2 联合onset候选注册与启动前归档边界

本地登记joint_onset，标签jointonset20260913，以第30节onset_aux为父对照，B4/B5
各seed42/43，最多60 epoch，仍复用现有模型目录。相对父对照只启用event_onset_joint，
沿用独立onset头及辅助系数1.0、近窗摘要、last_mean、原GCN/GAT-GRU和原BCE等损失。
F-beta为0，远历史/时间注意力/图refine关闭。没有额外可训练参数；基于现有128维
事件MLP，完整参数量应为B4 289695、B5 302623，待真实四组配置再次核验。

新分数在event_history_hot>0.5时取原event分数，其余取两头logit的maximum；主事件
损失和正式评估都使用该组合。门控输入由38.1定义的原30窗X单独计算，不用legacy
hist_last_hot选择分支。原未来监督、decoder使用的历史标志、P>=0.8阈值选择和
P>=0.8/R>=0.7可行优先的F1选模仍保持共同协议。因此该项只检查因果历史门控下的
联合训练机制，不冒称完整主配方对齐，也不修复整个共同解码协议的前视问题。

训练从头开始，B4 batch24/min10/patience10/lr3e-4，B5 batch16/min15/patience20/
lr1.5e-4，AdamW weight_decay0.01、uniform、seed42/43。保持训练预算和评分，不因
upcoming单项更换epoch、阈值或拼接输出。比较对象同时保留同seed onset_aux和near，
分别解释联合路径相对辅助监督的效果及相对原近窗baseline的实际收益。

run_control先构建并检查near和新的onset历史输入，成功后才归档原权重；失败不得
归档或替换当前结果。新运行记录保存onset_history_contract及联合分数是否参与主
损失/报警；旧variant不新增该记录字段。配置/记录新增测试覆盖四组唯一配置差异、
正确父对照和输入构建失败时的归档隔离，连同联合路径/F-beta共16 tests、6 subtests
通过；另两项所有dense候选的配置回归通过。既有目录型归档测试补齐新输入构建mock，
本轮未创建临时目录运行它们；实际旧文件归档机制未改。

该候选仅登记，尚未部署或开训。F-beta最后四份已正常结束，正在逐文件汇总核验；
整批完整导出完成并复核后，才允许部署、执行真实208数据/四组配置预检，再串行训练。

### 38.3 联合路径部署、真实预检失败及无效节点门控修正

F-beta整批完整核验后，服务器由1728ca6快进至9c741c5，四组原权重/结果共28文件
哈希不变。部署前守卫最初因git默认引用中文文件名而拒绝路径检查，未发生merge；
改为git diff --name-only -z的NUL分隔后保持原目录范围限制，通过并部署。
部署记录baseline_joint_onset_deployment20260913.json，2426 bytes，SHA
bc3f2b94ae08040da1dbad3d048fef406ef9b1101deb1d04696009c71b78d501。
服务器61 tests、17 subtests通过，3项目录型测试排除；记录
baseline_joint_onset_server_tests20260913.json，1476 bytes，SHA
25163ed4581a793f4c12fa5b84aa4f0c58ee0e80945116fdce04f340eba569be。

首次真实预检在baseline_dense_diag复用pane382787执行，现dead=1、exit=1；训练
pane326665仍dead=1、exit=0。脚本baseline_joint_onset_preflight20260913.py，
12426 bytes，SHA 50b75db4daff3c604b6177a1400ac07bb36ed7d2c0af1f68b8824fc7e10aafc4。
其第43行比较父onset_aux保存的occupancy_type_masks失败：父metadata为节点索引
列表（例如machine=[27,28,29,30,31]），_occupancy_type_masks返回的是38维0/1
向量。两者须严格统一为索引列表再比较，不改变实际类型分配，不移除这一守卫。
失败日志jointonset20260913_preflight.log，619 bytes，SHA
712bd2383377d0b2bab9cd71def89cb70da8152b181f46de564b898550250311。
没有完成四组预检、没有优化器更新或联合训练；脚本与日志原地保留，重用名称前须
通过已有archive_files机制验证归档。不能把61项合成测试写成真实208预检通过。

另在本地源码审查中复现独立的门控边界缺陷：一个有效machine连续blocked=60、
queue=0，另一个node_mask=0的human，其unavailable从0变1时，旧observed_history_hot
会让有效machine的gate从0变1。共同ops_occupancy_raw对操作员缺席使用跨节点any，
因此只对最终gate乘node_mask不够。此反例不证明当前208样本实际受到影响。

本地修正observed_history_hot显式接收既有node_mask，验证为逐节点binary；在冻结
训练归一化反变换之后，将无效节点全部27通道（包括类型）置零，再调用原30窗ops规则，
最后仍屏蔽输出。归一化之前置零不等价，因为归一化零值可能还原为非零均值。
独立onset_history_contract升级v2并记录mask顺序；旧variant默认关闭，不改变旧权重
或原标签/解码。新增回归同时验证无效human不影响有效machine、有效human仍影响规则、
非零归一化均值、全无效节点、输入不被修改以及无效mask形状/数值被拒绝。

修正后本地90 tests、17 subtests通过，3项显式排除。执行范围有一次违规：误把
test_baseline_artifact_reuse.py整文件纳入，旧tmp_path测试自动在系统临时目录创建
pytest-49及27个fixture目录，与用户不新建目录/仅指定仓库写入的约束冲突。已在对话
告知用户，不删除这些目录，不将本次描述为完全符合执行边界。路径为
/var/folders/_f/gg6rlmn16b760vw5j5tkhgs40000gn/T/pytest-of-xuwantong/pytest-49。
后续该文件仅允许显式node id test_dense_candidates_are_single_variable_and_leave_scoring_unchanged，
不得沿用整文件命令。其他测试也先查tmp_path/TemporaryDirectory等fixture再选择。

服务器仍9c741c5，v2修正尚未部署。下一步在无活动任务时部署修正，运行不创建目录的
定向检查，保留旧失败证据后严格修复预检索引表示，并量化v1/v2 gate在真实train/
validation的差异；全部四组配置、输入契约和损失路径通过之后才允许joint_onset开训。
这项修正不解释既有低召回：新门控尚未用于任何已完成训练。架构结论仍是联合报警
训练路径值得单独检验，已有训练/验证差距不足以宣称GCN/GAT-GRU无法预测upcoming。

### 38.4 门控修正部署与完整真实208预检通过

本地门控修正cbec4a9已推送。首次修正部署守卫发现上一节日志SHA从终端转录时少了
两个字符，停止于fetch/merge之前；重新分四段读取原619-byte日志，确认实际SHA为
712bd2383377d0b2bab9cd71def89cb70da8152b181f46de564b898550250311，日志原字节未变。
本地文档/JSON修正提交7ff759d后，服务器在两项pane终态、无本仓库活动Python、
完整F-beta结果及28个当前训练文件哈希通过后，从9c741c5快进至
7ff759dd722c8b12b730be5a3714672796655ddf。部署后上述28文件及六份数据size/mtime
与最近全量SHA核验记录一致，运行Python源码逐份与Git对象核验。

部署记录baseline_joint_onset_mask_deployment20260913.json，1221 bytes，SHA
febe6fd63fa358449afde4da16210d162857622d46040ea5ba39a073debef5c3。
服务器定向检查62 tests、17 subtests通过，3项会创建目录的诊断测试排除，
artifact_reuse只运行明确的配置检查node id，未整文件执行。测试记录
baseline_joint_onset_mask_server_tests20260913.json，1481 bytes，SHA
079b2b5213af0dcc83458896e05bd7fd635153ca84e14e677751fed1de57eafe。

旧预检脚本/日志通过既有archive_files逐字节验证后归档并释放原文件名，归档
baseline_joint_onset_preflight_guard20260913.zip含两个原文件及manifest，5389 bytes，
SHA a1e2601764c422b7ec91e1769b62383a0a031f6234baca0e2ff8e35130aafa99。
归档首次调用因未拼接tmux既有PYTHONPATH，导入时缺torch，尚未写归档；补齐原有
依赖路径后完成，不重装依赖、不新建目录。旧9c741c5部署/测试报告仍原地保留。

修正版预检复用baseline_joint_onset_preflight20260913.py，15108 bytes，SHA
ab7bbd0b29ed0848f1a266c113e73a9084b1a6d2cbbec9b2cbfe0386924234cc。
只把实际类型掩码严格转换为节点索引列表，与四份onset_aux父metadata完全比较；
不改变类型分配。增加v2契约断言，并从固定9c741c5 Git对象提取旧gate函数，对全部
train/validation样本逐一比较旧、新构建结果。未选择test样本，也没有优化器更新。

| 全量预检 | Train | Validation |
|---|---:|---:|
| 样本数 | 23859 | 5439 |
| ongoing / upcoming / negative目标 | 4191 / 595 / 297152 | 950 / 145 / 67331 |
| v1→v2发生gate变化的样本/活跃节点 | 0 / 0 | 0 / 0 |
| observed hot：ongoing / upcoming / negative | 1635 / 0 / 395 | 334 / 0 / 97 |
| 旧hist hot、observed cold的有效目标 | 2260 | 543 |
| 旧hist cold、observed hot的有效目标 | 0 | 0 |

掩码修复在当前包没有改变有效gate，不能用它解释既有低召回；它修复的是输入边界。
新门控与legacy hist_last_hot并不完全相同，不能将2260/543个差异全部归于同一个原因，
也不能声称完全复刻主模型的旧门控。所有595/145 upcoming均进入cold组合分支。
near输入契约在29298样本与四个父对照一致，后五维远历史保持零，原payload字段对象
保留，near/gate valid掩码一致。近窗/新门控只用注册的原历史范围。

四组真实配置全部通过：B4参数289695、B5参数302623；原参数、RNG、初始train/eval
输出和全部损失与onset_aux父对照相同，新增开关没有额外可训练参数。实际类型权重下
主事件损失对GRU、原事件头、onset头、precursor投影均有非零梯度，起点/hot头不受
该项直接梯度；改变未来标签及legacy hist不能改变模型输出。各父归档11成员及关键
权重/config/results与旧完整导出逐项匹配。此预检在CPU执行，注册训练设备cuda:0。

预检pane396832、实际Python396835均结束，pane dead=1/exit=0，日志
PREFLIGHT_EXIT_CODE=0。完整报告baseline_joint_onset_preflight20260913.json，
27178 bytes，SHA 8ba2f21b149e66962bf2421d54faad3d4af7135096ed4875fca678ae8b7562f0。
以上是执行正确性和输入/参数控制证据，不是召回收益证据。

### 38.5 联合onset四组串行训练启动

核验完整预检SHA、源码、旧F-beta文件、两项pane退出0、无相关活动Python后，在原
baseline_dense_v6复用会话启动；GPU为RTX5090，启动前空余28696 MiB。驱动
baseline_joint_onset_driver20260913.py，6380 bytes，SHA
306bbebc45db7edf65e99acdfd776b086bf7b4e29e3f43e0aabbde84fea32955。
日志jointonset20260913_train.log，训练pane400178、driver Python400181，首个实际
训练命令Python400222为B4 seed42。首次观察到ALL_FOUR_OLD_F_BETA_RESULTS_VERIFIED_BEFORE_TRAINING
及START_JOINT_ONSET B4 42，仍在输入准备，尚无epoch/config完成核验或新指标。

顺序B4 seed42/43、B5 seed42/43，复用四个candidate_history/seed目录，标签
jointonset20260913；每组原F-beta文件先逐成员验证归档model_before_jointonset20260913.zip。
每组最多60epoch，参数、损失、抽样及选模按38.2；F-beta不叠加、远历史/attention/
graph refine关闭。驱动逐组运行前后核验固定源码和冻结数据状态；完成后核对实际
模型/损失/训练配置、近窗和历史门控契约、旧归档全部11成员及新文件SHA，再允许下一组。
逐组training快照与整批训练记录写入原benchmark；不把训练快照当作16份冻结诊断完成。

运行中服务器保持7ff759d，不pull后续文档。恢复时检查具体进程和checkpoint来源，
不能因观察超时重启。下一步确认首组新config/旧归档并等待权重终态，再做best/last
的train/validation诊断；未开test，尚未宣称upcoming提升或架构上限。

随后首组进至epoch6，400222仍为实际B4 seed42进程、400178 pane仍live。实际config
的model/training/loss与预检逐项一致，metadata的near/observed_history/类型掩码契约
均一致；joint=true、aux=true、near、F-beta=0、evaluate_test=false。旧F-beta归档
11成员CRC及SHA通过，六个关键权重/config/result与F-beta完整导出一致；其余三组
原文件哈希不变，六份数据stat不变、运行源码仍7ff759d。运行核验记录
baseline_joint_onset_launch20260913.json，1024 bytes，SHA
b5322d4d9b6836033203813cc97f8d2e11cbdf80651b8e34875d51af73e9c9b9。
早期epoch1–3的upcoming均为0，不能据此提前停止或判断最终结果。继续原训练预算和
选模规则；下一步等待首组权重终态并收集冻结诊断，不重复当前训练。

### 38.6 首组训练完成与联合报警分支只读观察准备

B4 seed42实际Python400222已消失，驱动记录该组正常结束并接续B4 seed43，
后者实际Python405436已观察至epoch6，训练pane400178和driver400181仍live。
首组best1/total11、参数289695，正式P/R/F1为
0.8512880562060889/0.6639269406392694/0.7460236018471011，upcoming=0/145，
保存阈值0.65。后期epoch6–10出现少量upcoming，但整体F1下降；仍按原规则保存
best，不能挑选后期epoch代替。该结果与父onset_aux同seed同为0/145，尚无提升。

独立核验首组训练快照status、训练提交7ff759d、当前七文件SHA、旧F-beta归档CRC及
11成员SHA均通过；实际joint=true、F-beta=0、evaluate_test=false。快照
baseline_dense_joint_onset_b4s42_training20260913.json，32785 bytes，SHA
b37da8eab23d468685f16c554020eaae5e0a6bb5b0173a963cfa8e697b8a7187。
四份best/last × train/validation冻结诊断尚未开始，不将训练快照作为诊断完成。

本地diagnose_baseline_events.py新增仅对joint checkpoint生效的只读观察。
从同一次forward的原continue/onset/combined logits核验注册组合规则，再收集原头
概率、已有observed gate、onset实际胜出和cold同分标志。分支选择从logit比较获得，
避免sigmoid饱和把不同logit误判为同分；不再调用forward、不改参数、RNG或原输出。

保存阈值下，统计各组的gate状态、onset胜出、概率抬升、新增报警/命中/误报，同时
保留原头/onset/组合的upcoming-vs-negative AP。对同一联合训练权重临时只用原头
重新计算一份canonical报告，起点/时长/阈值及legacy解码不变。它只是冻结输出消融，
不能当作独立训练baseline、不选新阈值/epoch、不替换正式分数，也不等于联合训练的
因果贡献。旧独立辅助头路径和旧诊断不变；joint报告中的辅助头scope改为明确参与
组合报警，修正原通用文案把它写成“排除在正式报警外”的不准确描述。

本地39 tests、9 subtests通过，3项临时目录测试显式排除；新增检查覆盖两种真实骨干
观察前后参数/RNG/输出不变、不发生第二次forward、冷/热/同分/饱和边界、额外命中与
误报及起点错误分离、canonical原报告/输入保持不变、无效节点和空支持定义。只改
诊断脚本/测试及文档，服务器训练模块保持7ff759d；新诊断通过Git对象stdin执行，
先做服务器独立内存测试，再启动首组四份冻结诊断，不在运行期间pull源码。

### 38.7 两颗B4完成，首份联合分支诊断核验及接续

B4 seed43也已正常结束并逐文件核验，实际Python405436已消失；best3/total13，
参数289695，P/R/F1为0.8401400233372228/0.6575342465753424/0.7377049180327869，
upcoming=1/145，阈值0.70。当前七文件与快照、旧F-beta归档CRC/11成员均通过。
baseline_dense_joint_onset_b4s43_training20260913.json，32931 bytes，SHA
5a91b5118fdb25eb556b5b4ae2c3d4c109807f329e1cc1f24a94642a3d655a4b。
两seed平均P/R/F1/upcoming recall为
0.8457140397716558/0.6607305936073059/0.7418642599399441/0.0034482758620689655。
对应onset_aux命中0/3、near命中2/2；B4本候选未改善，不能因此推广为骨干上限。
B5 seed42已接续，实际Python411056；服务器训练pane400178保持live、HEAD7ff759d。

观察源码d5670bcfe40cbfe186dd8fc757f1b624c60c4948已push，服务器仅fetch Git对象。
独立CPU进程将固定诊断/测试源码载入内存，新增3项测试通过，实际运行模块逐份哈希
仍匹配7ff759d。源码SHA 73a86dfff21fa49cd180e19bdfd5f4a87854192ea79aaf5878db4da2ad013893。
测试记录baseline_joint_onset_observation_tests20260913.json，1374 bytes，SHA
453e22a458379c2e4b603c12ffb7304232fc3561d6421137b9bdbd3653a04793。
未创建测试目录、未更改训练模块。首组诊断驱动在原baseline_dense_diag启动，
初始pane413059；源码5547 bytes，SHA
c6df3a1b0c11f5da8e1c05c557425b2586145ad7153753db6d32d20b8bd3f3d4。

首份best validation预测和诊断正常生成后，驱动后处理因KeyError:
report_threshold_used退出1。该字段由torch_trainer附加，canonical报告中没有；
不是预测失败。保留首份完整JSON，修正后处理为比较canonical非MAE分数/计数，并
单独严格核对保存阈值、训练report_threshold_used和诊断阈值相同。训练报告其他
附加字段差异只允许MAE相关项，连续CPU/GPU MAE不作逐位一致性宣称；没有放宽
P/R/F1或计数的1e-10核验。未重复执行已生成的best validation。

旧驱动/日志两文件通过archive_files逐字节验证后归档
baseline_joint_onset_b4s42_diag_guard20260913.zip，4677 bytes，SHA
bffd1fd12436a87cb655f1d9db21417998911ecee81c28fc0cdfbda24f85d78d。
旧日志5054 bytes，SHA 6fb00fe44b24e594c6e6734c6e907571475f9812b5312c473916f57f3e723169。
接续驱动复用原文件名baseline_joint_onset_b4s42_diagnose20260913.py，6495 bytes，
SHA 07fae8c7379f0317d9083eea7d87047b361dd32ce8edefdfb1b3bade39dd03bf；
diag pane416164 live，实际Python416194在运行best train。它先按固定SHA复用并
核验首份结果，然后只运行best train、last validation、last train三个缺项。

首份baseline_joint_onset_b4s42_best_validation20260913.json，129118 bytes，SHA
4dadbee8c79cb33e4b6c252d09458b2500baf2609e520fe283f21ff1f9c026e7。
已通过来源、checkpoint、epoch1、样本5439、分组950/145/67331、正式canonical分数/
计数及阈值0.65复现。组合upcoming-vs-negative AP=0.00788754369669698；观察结果：

| Best validation组 | ongoing | upcoming | negative |
|---|---:|---:|---:|
| 目标数 | 950 | 145 | 67331 |
| observed hot | 334 | 0 | 97 |
| cold时onset logit胜出 | 107 | 142 | 64794 |
| 保存阈值下新增报警/命中/误报 | 0/0/0 | 0/0/0 | 0/0/0 |

分支已抬高多数upcoming分数，但在该best的固定阈值下没有改变报告，不能解释成onset
路径没有接上。该观察只针对同一联合训练权重，不能视为与独立训练原头的因果消融，
也不证明其他epoch/seed相同。剩余三份诊断和B5训练继续，未宣称目标已达成。

### 38.8 两颗B4全部冻结诊断完成，B5接续

B4 seed42的接续诊断正常结束，pane416164退出0、最后实际Python417961已消失；
四份结果均完成，首份best validation按原SHA复用而未重算。完整汇总
baseline_joint_onset_b4s42_diagnostics20260913.json，604578 bytes，SHA
7638c88b88e68377781f051cb8a058727e8115bb64e56ea1959a01347fd89408。
B4 seed43随后在pane424055完成四份诊断并退出0；汇总
baseline_joint_onset_b4s43_diagnostics20260913.json，605207 bytes，SHA
ed6de57ee2d7f87fe83579b322f4bd7f082462e9652b27f45977d7c7c85fd1a4。
两组均独立复核原JSON与汇总内嵌结果、四份SHA、当前七文件、checkpoint/source、
best validation canonical计数与非MAE分数；没有替换best或搜索新阈值。

| Joint frozen诊断 | B4 s42 best train | best validation | last train | last validation |
|---|---:|---:|---:|---:|
| Upcoming AP | 0.009566 | 0.007888 | 0.234417 | 0.014917 |
| Upcoming命中 | 0/595 | 0/145 | 145/595 | 5/145 |
| Onset较同权重continue-only新增upcoming命中 | 0 | 0 | 137 | 5 |
| Onset新增误报 | 0 | 0 | 169 | 62 |

| Joint frozen诊断 | B4 s43 best train | best validation | last train | last validation |
|---|---:|---:|---:|---:|
| Upcoming AP | 0.020260 | 0.013752 | 0.383150 | 0.019378 |
| Upcoming命中 | 4/595 | 1/145 | 243/595 | 6/145 |
| Onset较同权重continue-only新增upcoming命中 | 4 | 1 | 226 | 5 |
| Onset新增误报 | 12 | 0 | 212 | 65 |

AP表为六位小数；两seed的best/last保存阈值分别0.65/0.90、0.70/0.90。
以上八份诊断的upcoming timing misses均为0，剩余漏报均先被概率阈值挡住。
这不是“时间头完美”的证据，因为概率未过线的目标不进入后续timing miss分支。
额外贡献是同一联合训练权重的冻结输出消融，不是独立训练的因果效果。模型能拟合
训练集部分upcoming，验证收益小且伴随误报，排除分支完全未接入，尚不能归因于某个
骨干组件或证明GCN/GAT-GRU上限。

B5 seed42训练已正常结束、原Python411056消失；best7/total27、参数302623，阈值0.65，
P/R/F1为0.8299776286353467/0.6776255707762557/0.7461035696329815，upcoming=3/145。
当前七文件和旧F-beta归档11成员独立复核通过，joint=true、F-beta=0、test=false。
训练快照baseline_dense_joint_onset_b5s42_training20260913.json，33059 bytes，SHA
fdc0a8b9d9804ec39d96de75b0847eb522345d55b0ee9c235a0b90f18c7107d3。
对应onset_aux父对照为5/145，单seed尚无改善。B5 seed43 Python424214接续训练。

确认B4 seed43诊断pane退出0及汇总SHA后，复用baseline_dense_diag启动B5 seed42的
best/last × train/validation四份缺项，pane430931。驱动
baseline_joint_onset_b5s42_diagnose20260913.py，6015 bytes，SHA
12b167760cd66032c5c82b7be468073fb27d383c1e348dac321e76d59a720931；日志
jointonset20260913_b5s42_diagnose.log。诊断源码固定d5670bc，经Git对象stdin执行，
模型模块和训练HEAD保持7ff759d，不pull。该段记录的是启动状态，不能当作B5诊断完成。

### 38.9 主参考组件路径复核：哪些结论成立

按用户最新要求再次只读核对20c40e2的FactoryBN_dense_f1_p80.json、model.py的
_combine_will_logit、_event_span_loss、_apply_occupancy_union、predict及backbone.py。
这仍是固定参考配方，缺正式主checkpoint的实际覆盖配置与同208包结果，不能把主模型
公布的upcoming分数当作纯架构差异的受控证据。

1. 主配方的onset阈值0.50低于report阈值0.70，cold概率达到onset阈值时会被抬到
   report阈值。此路径影响报警，不能只比较原始事件头概率。旧onset冻结双阈值边界已
   完整检查过；不要再把同一阈值搜索当新实验，也不能把旧权重的结果外推到所有架构。
2. 未来occupancy以概率0.45判热、连续8窗构造候选，event_union_upcoming=true允许
   cold目标补报；随后仍受最晚起点2窗约束。29.4的同权重hot探针已覆盖对应0/1/2起点
   的连续8窗信号，新增upcoming越阈值只有0/0/1/0，负例越阈值33/16/11/21。
   不重复该检查，现有输出不足以支持直接搬入补报。
3. prefix解码在hist_last_hot存在时要求其>0.5，因而不能直接挽救cold upcoming；
   它对ongoing报警和整体阈值选择的间接作用未单独识别。STGNPP在参考配方关闭。
4. 主模型5层中反复进行时间/地理/语义交互；B4/B5先做逐时刻两层图编码，再逐节点
   GRU及末状态/均值汇聚。已测的末端时间汇聚、GRU后单层图残差只是有限近似，负结果
   不能排除反复时空交互、位置/模式编码的潜在作用。
5. 主配方还有续训、focal、事件窗采样及不同损失归一化；当前joint仅隔离组合训练机制，
   不等于完整复制主模型配方。组件间相互作用仍未识别。现有最直接证据是概率排序与
   泛化不足；把它进一步解释成某个架构组件缺失，需要新的受控证据，不能由架构图推定。

### 38.10 复用现有B4诊断：验证误报主要不是短热事件

只读两份已核验SHA的完整B4 joint汇总，提取已有概率分位数与误报分解；没有重新
forward或增加阈值搜索。下表为validation，各数均为窗口-工位统计，不能当作独立事件数。

| B4 joint validation | s42 best | s43 best | s42 last | s43 last |
|---|---:|---:|---:|---:|
| 保存阈值 | 0.65 | 0.70 | 0.90 | 0.90 |
| Upcoming事件概率中位数 | 0.0394 | 0.1047 | 0.0417 | 0.0033 |
| 全部误报 | 127 | 137 | 140 | 138 |
| 可观测未来没有hot的负例误报 | 117 | 127 | 136 | 132 |
| 有hot但不满足事件规则的负例误报 | 10 | 10 | 4 | 6 |
| 短观测区间误报／真实事件起点错报 | 0/0 | 0/0 | 0/0 | 0/0 |

“没有hot”组要求可观测未来至少8窗，且这些有效未来窗全部无hot；不是把未观测未来
当作负例。四份验证的起点容差内比例都是1.0，八份train/validation也全部为1.0。
这里所有upcoming真值起点只在1、2窗，而评分容差为3窗，因此该结果不能被解释成
准确预报起点的强证据，但足以说明当前评分下的漏报不由缺少起点组件直接造成。

Best验证误报的117/127、127/137（约92%）来自上述完全无hot的负例。不能据此继续
把本轮主要问题解释成“没有持续时长组件，所以把短热误报成长热”；现有误报构成不支持
这个解释。末轮train的upcoming概率中位数0.5604/0.8014，validation降至0.0417/0.0033，
与AP差距一致，表明当前分数对新episode的提前事件缺乏稳定区分能力。
这是对直接失败环节的定位，仍不是某个骨干组件导致泛化差距的因果证明。

已有events_vs_short_hot_negative AUC混合ongoing/upcoming正例，不能当作upcoming专属
判别指标；此处不由它推导upcoming已能区分短热负例，也不重跑旧134包相关搜索。

### 38.11 四次训练完成，B5首组诊断完成、末组接续

B5 seed43也已正常结束：best5/total25，参数302623，保存阈值0.65，P/R/F1为
0.8205405405405405/0.6931506849315069/0.7514851485148515，upcoming=2/145。
原Python424214消失，快照baseline_dense_joint_onset_b5s43_training20260913.json，
33010 bytes，SHA 91360e2e5d93bf2e814f19b8e2c3bf21d0f6de4bda474bf23718cf3a4816a1c5。
其当前七文件和旧F-beta归档11成员独立核验通过。

训练pane400178退出0，driver400181及四个实际训练Python均已消失。整批训练汇总
baseline_dense_joint_onset_training20260913.json，8598 bytes，SHA
083daffcbd73ed25a0683a15504303075c9616e35c8bca04bb3cbf0707795480。
四快照、全部28个当前文件、四旧归档CRC及44成员、固定运行源码7ff759d独立重验通过。
B5两seed平均P/R/F1/upcoming R为0.8252590845879436/0.6853881278538814/
0.7487943590739166/0.017241379310344827。四组命中0/1/3/2，onset_aux父对照0/3/5/2；
联合训练本身没有稳定改善。训练汇总仍标diagnostics_completed=false，不把它当整批收尾。

B5 seed42四份冻结诊断正常结束，pane430931退出0，最后实际Python433547消失，
完整汇总baseline_joint_onset_b5s42_diagnostics20260913.json，605413 bytes，SHA
75c3c3992660e53cdecef7df18057e8d04304a10d5849d59f0b4c66e0c371a44。
独立核验原JSON/内嵌结果/四份SHA、权重及source、正式best validation canonical分数
计数均通过。按best train、best validation、last train、last validation顺序：

| B5 s42 joint诊断 | Best train | Best validation | Last train | Last validation |
|---|---:|---:|---:|---:|
| 保存阈值 | 0.65 | 0.65 | 0.80 | 0.80 |
| Upcoming AP | 0.028816 | 0.010862 | 0.639215 | 0.010652 |
| Upcoming命中 | 24/595 | 3/145 | 380/595 | 8/145 |
| 同权重onset新增upcoming命中 | 16 | 1 | 325 | 7 |
| 同权重onset新增误报 | 18 | 5 | 186 | 61 |

四份upcoming timing misses均0；AP为六位小数。训练可拟合而验证弱的现象在B5再次出现，
不按last挑权重、不降阈值挽救指标，不由单轮负结果认定整个GAT-GRU家族到顶。

确认该组diag退出0及汇总SHA后，接续B5 seed43四份冻结诊断至pane439500，驱动
baseline_joint_onset_b5s43_diagnose20260913.py，6015 bytes，SHA
7d7fdbe3ad32038a82cd2ba9ee12266cd66b857ac48a46d031ffaa202846d2f8；日志
jointonset20260913_b5s43_diagnose.log。最近已完成last validation（AP
0.011509053184269421，onset新增upcoming命中3、新增误报45），正在last train。
末组完整汇总和整批16份诊断收尾尚未核验，不能重复前三份推理。

期间UU独立终端窗口从窗口列表消失，通过UU主窗口Leo设备卡的“终端”入口恢复，
同一SSH会话与工作目录仍在。先前整批只读核验命令未送入终端，恢复后已实际执行并
确认通过；没有重启训练/诊断，也没有改用本地直连SSH。

### 38.12 Joint整批训练与16份冻结诊断收尾完成

末组B5 seed43四份诊断已正常结束，pane439500退出0。完整汇总
baseline_joint_onset_b5s43_diagnostics20260913.json，605467 bytes，SHA
463293f2cef300f6e1705b6994246e403260f004f3e8df590782169dfa0cefa9。
按best train、best validation、last train、last validation顺序：

| B5 s43 joint诊断 | Best train | Best validation | Last train | Last validation |
|---|---:|---:|---:|---:|
| 保存阈值 | 0.65 | 0.65 | 0.90 | 0.90 |
| Upcoming AP | 0.021170 | 0.011654 | 0.598038 | 0.011509 |
| Upcoming命中 | 19/595 | 2/145 | 313/595 | 4/145 |
| 同权重onset新增upcoming命中 | 17 | 0 | 243 | 3 |
| 同权重onset新增误报 | 26 | 7 | 71 | 45 |

最终整批校验确认训练pane400178、诊断pane439500均退出0，未发现带本批variant/
diagnostic source标记的Python；服务器HEAD保持7ff759d且tracked clean。
四份训练快照、28个当前文件、四旧归档CRC及44成员、四诊断汇总/16原文件与其内嵌
JSON、诊断源码/权重/epoch/分组/样本数逐项一致。四份best validation canonical
非MAE分数和计数复现通过、保存阈值相同。五个冻结数据文件及episodes.npz共六文件
全量SHA与F-beta完整导出相同，执行前后stat相同，运行源码前后不变。

完整导出baseline_dense_joint_onset_metrics_20260913.json，2836774 bytes，SHA
c35d4d46a2a45d40f48381662278ea1655c0a1e0a02b8ec053a7018767dcae19。
写入后JSON回读等值，并独立核对该SHA转录、4次训练/16份诊断/6数据哈希项计数。
该完整导出包含四组训练快照和四组原诊断，并标diagnostics_completed=true；早期训练
快照的false保持其历史状态，不修改旧产物。不评test、不动主仓库、不新建目录。

正式upcoming命中0/1/3/2，相对onset_aux父对照0/3/5/2没有稳定收益，不采用本候选
作为upcoming提升方案。四颗last的train AP为0.234417/0.383150/0.639215/0.598038，
validation为0.014917/0.019378/0.010652/0.011509。由此可以定位当前失败在事件分数的
泛化，而不能进一步认定某一缺失组件或整个骨干上限。主参考的反复时空交互、位置/
模式编码及训练配方相互作用仍未受控隔离，正式主checkpoint和同包对照仍有材料缺口。
当前全部进程已结束，下一训练尚未预注册；不重复本轮或旧搜索，原目标仍未完成。

## 39. 邻居优先级是否能随目标工位变化：B5单项结构对照

### 39.1 源码限制与最小验证

上一目标轮完成joint四次训练/16份诊断及整批核验，属于实质进展。继续时重新核验
服务器7ff759d、tracked clean、两项旧pane均退出0、无相关Python，完整joint SHA未变。
尚不能把泛化差距归因于架构，下一步转向可明确隔离的注意力公式差异。

当前B5 DenseGraphAttention每个头的分数为
`LeakyReLU(sum(a_src * W x_i) + sum(a_dst * W x_j))`。对于同一时刻、同一头、共同
可见的邻居j/k，目标i只增加相同标量，单调激活和softmax不改变j/k的排序。因此每头
不能根据目标工位改变共同邻居的相对优先级。注意力的具体权重仍可变化；不能说B5
“没有注意力”或所有输出不依赖目标。不同头、层、残差及GRU仍可组合表示。

主参考20c40e2的地理/语义注意力采用QK点积，允许目标查询改变邻居排序。这一标量
加法限制与[Brody等人的GATv2论文](https://arxiv.org/abs/2105.14491)所分析的static
attention一致；论文并不证明本208工厂任务的低召回由此造成。B4的归一化图权重也
不按目标状态改变，但本轮不把GCN改造成GAT，以保留B4原有身份。

实际DenseGraphAttention最小验证使用两个查询(-2,0)/(0,-3)、共同邻居(1,0)/(0,2)，
投影为单位矩阵、两个注意力向量均为1。由层输出可还原邻居权重，结果如下：

| 打分方式 | 查询1优先邻居 | 查询2优先邻居 |
|---|---|---|
| 原标量加法后激活 | 邻居2 | 邻居2 |
| 向量相加后激活、最后求和 | 邻居2 | 邻居1 |

这证明新增模式可以消除这个具体限制，不代表已经提高真实召回。此前末端时间汇聚和
GRU后同类型图层的实验保留原GAT公式，不能当作这条机制已经测试过。

### 39.2 vector_gat预注册与本地实现

只登记B5 seed42/43，标签vectorgat20260913，父对照near_precursor。模型新增配置
`gat_score_mode=vector_additive`，两层都将LeakyReLU移到通道求和之前：
`sum(LeakyReLU(a_src * W x_i + a_dst * W x_j))`。W及两个注意力向量全部沿用，参数
数量、参数名、初始参数值及RNG完全一致；初始预测预期改变，因为直接替换打分公式。
这是因式分解的动态打分对照，不冒称复现具有独立任意源/目标矩阵的完整GATv2。

旧配置默认scalar_additive，保留原分支原始计算顺序，旧checkpoint可加载且输出一致。
新增模式不改变图边表、输入信息、近窗摘要、损失、抽样、GRU/读出/预测头或评分。
Onset、F-beta、远历史和GRU后图残差均关闭；B5仍batch16、lr1.5e-4、weight_decay0.01、
max60/min15/patience20、均匀抽样，保存阈值与report F1选模完全沿用。正式父对照
upcoming为1/145、2/145；不挑其他候选当父对照。B4显式拒绝该variant，维持当前文件。

只复用现有B5两个seed目录，各自在开训前归档并核验当前joint文件；保留原近窗与
joint成果，不新建目录、不改主仓库、不用test。服务器真实预检须验证近窗输入契约、
两seed初始参数/RNG和参数量、实际208 batch的有限前后向、GPU内存以及旧文件不变。
目前尚未部署或启动新训练，不能把本地测试当作真实208性能证据。

本地15 tests、22 subtests通过：包含实际层的查询换序例子、默认分支原公式精确复现、
同权重/参数量/RNG/dropout抽样、有效边/无效节点/排列等变、两层和GRU的事件损失梯度、
内存checkpoint往返与旧配置默认、两seed单变量配置及B4拒绝。只运行不创建目录的
定向测试；artifact_reuse仅选test_dense_candidates_are_single_variable_and_leave_scoring_unchanged。

### 39.3 部署、服务器测试与预检配置表示修正

19a4a0ce1c5d7ab24d7e7a40cbb59bdbcb0692d3已部署到空闲BSTAN dev_xwt，旧28模型文件、
六数据stat和joint完整SHA前后不变。部署记录baseline_vector_gat_deployment20260913.json，
5206 bytes，SHA 07b1768f9504b38ad20b17bb38fb8cfc5ac121386d7ee5a80443b35d8c481541。
服务器RTX5090为32607 MiB，当时空闲28696 MiB；两种实际B5均为285982参数。

服务器独立进程8项测试通过，baseline_vector_gat_server_tests20260913.json，452 bytes，
SHA f5dc5c428f37d14d9d21e0b6aa353b33ecf882fda7799c95979a77ff06fc9fcb。
真实预检在pane462149执行，完成近窗输入构建后，在旧/当前training dataclass直接相等
比较处退出1，未开训、未归档模型。实际读回两seed旧配置确认，唯一差异是
report_threshold_sweep的JSON列表与代码元组表示；JSON序列化后的所有字段完全一致。

本地仅修正预检的反序列化：该字段按原顺序转为tuple，其余值继续严格比较。
同时修正预检末端将不存在的TorchTrainConfig.to_dict改为dataclasses.asdict；该处尚
未执行到，不是已发生的第二次预检失败。新增检查证明阈值值或batch大小变化仍会使
配置比较不等。本地16 tests/22 subtests通过，模型打分代码没有再次改变。
旧预检日志仍保留；重新使用日志前须核验归档。修正尚待部署、服务器9项测试和真实
预检接续，不把语法/配置修正当作方法收益，不重跑已完成的训练搜索。

### 39.4 修正版真实预检通过，B5两seed批次已启动

修正部署979c680fb4d1919ae760cdfb0038f69fb7cc6708，旧28模型文件、六数据stat仍一致。
baseline_vector_gat_preflight_config_fix_deployment20260913.json，5206 bytes，SHA
2b1fefdce2372505269d6ab25d1c1cc7744a6c06f78b5b341a56e571460de710。
服务器独立进程9项测试通过，baseline_vector_gat_config_server_tests20260913.json，
452 bytes，SHA 8b632bc37cba1467fde34d2c9d3dfdf5449998d86f91d278f805db212f032e37。
首次失败日志902 bytes，SHA
1ec87c2e9249af327d0e36fcb16680b6c8a6116ee73d986e27dc301ef1fa0ecc，已通过archive_files
校验归档为baseline_vector_gat_preflight_guard20260913.zip，791 bytes，SHA
d6cca4845a4351a84f7283260e6cf93a3ca5edfd203c7f7f95629d32104e9d82；只含该旧日志及manifest。
旧8项测试报告保持原文件，不覆盖历史产物。未发生模型归档或训练重跑。

修正预检pane468293退出0，真实近窗构建、父对照配置和旧近窗归档六个关键文件核验
通过。两seed参数均285982，与父对照参数值/名称/RNG相同；初始validation事件logit
最大改变量为0.255810/0.252112，符合更换打分公式的预期。实际train batch两层source
注意力参数及GRU、事件头梯度非零且有限，GPU峰值1655.0/1663.18359375 MiB。
选择在预测前固定：train split前15样本加首个upcoming（offset18），validation前15
加首个upcoming（offset33、全局sample2012），各16样本，不据分数挑样本。

完整预检baseline_vector_gat_preflight20260913.json，17170 bytes，SHA
e5f86076c68359f6887f8875507ca256db9199d09ebe12fe1882602aa27de811。六冻结数据全量SHA
与joint整批导出相同，前后stat一致；原28模型文件前后不变，旧近窗输入契约相同。
独立回读预检/source/28原文件和真实梯度检查通过，未用test、未宣称upcoming收益。

确认预检与旧训练pane均终态后，在原baseline_dense_v6启动B5 seed42/43串行批次。
训练源码固定979c680，pane472305、driver Python472308，首组实际Python472349。
日志vectorgat20260913_train.log；驱动baseline_vector_gat_driver20260913.py，6021 bytes，
SHA e37bae80c0b5ad34ddaabc3e00ac17a9be652852cae3b045b486040efac88ad7。
两组各最多60epoch，B4不运行此变体；每组当前joint训练文件在开训前逐成员校验归档
model_before_vectorgat20260913.zip，原joint control记录保持原文件。完成后写每seed
training快照及两次训练汇总，冻结诊断另行核验，不能把训练快照当整批诊断完成。

首组已独立观察至epoch4，实际model/training/loss逐项等于预检，near契约相同，
参数285982、onset/F-beta关闭、test=false。旧joint归档CRC/11成员校验通过，六关键
文件与joint完整导出相同，旧joint control记录未改；B4两组与B5 seed43原文件不变。
运行核验baseline_vector_gat_launch20260913.json，1152 bytes，SHA
a9c0fd383169c53e3e933e3f8554efae136481646aa7555340bfbfa981242e5f。
训练继续，暂无最终结果；禁止运行中pull、按中间upcoming挑epoch或重复启动现有批次。

### 39.5 首颗训练完成，冻结共同邻居排序诊断启动

接续先恢复UU独立Leo终端，同一SSH仍在。两次进度核验显示seed42正常推进至17/24轮，
随后训练快照确认best8/total28，参数仍285982、保存阈值0.55，正式P/R/F1为
0.8355555555555555/0.6867579908675799/0.7538847117794486，upcoming命中3/145；
父near seed42为1/145。seed43已自动接续，最新观察至epoch3；未根据中间指标选模，
当前不能认定两seed稳定改善。训练快照baseline_dense_vector_gat_b5s42_training20260913.json，
31521 bytes，SHA f60b81bdb85fd27c4613e769b81ac49b5348c68cf5b579cb54b89af8ac600931。

新增冻结观察源码0953fb3f10a53a1e998e3686770c5591bfc2eb97，diagnose_baseline_events.py
SHA 5ac68a4105bc908843da17d1cff812b6d6c06e8158a56c33a4dbf97158fef43d。
对同一原始forward注册临时hook，读取两层实际dropout前softmax；覆盖所有输入历史帧、
所有头和无序查询节点对。只比较共同有效邻居且至少有两个；两查询各自首选必须在另一
查询下严格落后，注意力绝对差大于1e-7，才计为换序。按query pair/time/head计数，并
仅在forward后按ongoing/upcoming/negative分组。不同边掩码或数值并列不算换序。
非零只证明使用了这种动态排序能力，不证明对召回有益；零也不能排除低排名邻居变化
或小于容差的差异。未改预测、模型、阈值或选模；不将注意力当因果重要性。

本地13 tests/6 subtests通过，包括真实标量/向量层差异、边掩码/并列/容差、节点排列、
空支持、计数恒等式、单次原forward输出/参数/RNG相同，以及异常时hook移除。
服务器仅fetch Git对象，HEAD仍979c680；独立进程4项测试通过。完整合成CUDA形状
16×30×38×27、全连接边作为内存检查，两层实际输出与无hook前向逐值相同，CUDA RNG
不变，峰值1441.306640625 MiB。这不是数据性能实验。服务器记录
baseline_vector_gat_ranking_server_tests20260913.json，688 bytes，SHA
9c11de797260942a82a9734362a39bb1cdc27e4a43a269ed15d8a89295b0caea。

冻结驱动cbcec6b4233cd78d318e01d991d0cba90172cf0d已提交，服务器既有D目录文件
baseline_vector_gat_frozen_driver20260913.py，8677 bytes，SHA
9e8fd5c37e51476de8d16851d01c1522d0e6ce4ff735d5db76f54fd091833d37。
确认原diag pane终态0后，seed42四份诊断在原baseline_dense_diag pane489190启动，
日志vectorgat20260913_b5s42_diag.log；顺序best validation/train、last validation/train。
诊断通过0953fb3 Git对象stdin执行，模型runtime仍979c680。驱动前后检查源码、六数据
stat、该seed七文件、两颗B4及旧joint归档；best validation须复现保存阈值和正式
分数/计数（连续MAE不要求CPU/GPU逐位一致）。已有结果只能严格核验后复用，不重跑。
目前尚无完成诊断结论；B5 seed43训练继续，B4和主仓库未修改，未用test。

### 39.6 首颗四诊断完成：动态排序已使用，泛化差距依然明显

seed42冻结诊断pane489190正常退出0，四份结果均完成；独立回读结果文件与汇总内容、
权重/epoch/源码、正式best validation分数和保存阈值全部一致。服务器仍979c680，
八源码文件、六数据stat、该seed七训练文件、两颗B4和旧joint归档11成员核验通过。
本次不重复六数据全量读取；整批结束后仍须完整SHA复核。

| 权重与样本 | 保存阈值 | upcoming AP | 命中/目标 | 真例分数中位数 |
|---|---:|---:|---:|---:|
| best train | 0.55 | 0.01553820 | 13/595 | 0.01429456 |
| best validation | 0.55 | 0.00791756 | 3/145 | 0.00818510 |
| last train | 0.65 | 0.42392727 | 260/595 | 0.52487528 |
| last validation | 0.65 | 0.00816133 | 3/145 | 0.00154505 |

两层GAT在四份结果中均对全部595个train或145个validation upcoming目标观察到至少
一次严格共同邻居首选换序。各层upcoming查询参与的pair/time/head比较中，换序占比：
best train 0.20731/0.18093，best validation 0.20068/0.18690，last train
0.21464/0.22441，last validation 0.20715/0.21778。这里的“全部目标”指每个目标
至少有一次换序，不能解释为每一帧/头都换序或对预测具有因果贡献。

新机制明确在真实输入上使用，但训练后期的upcoming可分性主要留在训练集；
validation的分数中位数反而很低，AP未随train升高。这不支持把原GAT静态排序限制
当作唯一瓶颈，也不证明完整GATv2、主模型所有组件交互或任何架构都无效。
目前仅完成一颗seed，正式命中3对父near的1属于小变化，暂不采用为稳定提升。

完整四诊断汇总baseline_vector_gat_b5s42_diagnostics20260913.json，536937 bytes，SHA
02c92f54bf9361efdf0a59ce1277c075fdbc1704f44c9d50658f5c7f4385b256。
独立核验baseline_vector_gat_b5s42_verification20260913.json，4675 bytes，SHA
ee18d021d94092f5d3c9c15de8f9f1d7e7b023a3d152d6d58023f0f144c5dc6f。
seed43仍按原预算训练，禁止重新运行seed42；待其结束后用同一冻结驱动执行四份
诊断，再完成两seed/八诊断/六数据全量SHA整批核验。未改主仓库，未用test调参。

### 39.7 第二颗训练与自动诊断接续已核验

独立回读seed42漏报拆分：best train/validation、last train/validation的upcoming
概率未过阈值分别为582/142/335/142，时间误差漏报全部为0。当前低召回主要在事件
存在性分数上，不能用放宽起始时间容差解释。seed43实际model/training/loss逐项等于
预检，旧joint归档CRC/11成员及原七文件的归档或保留位置均核验通过。

继续脚本finish_baseline_vector_gat_batch.py已提交于f41d3e983c822c36e334ae1305daad9f195f345e，
服务器D既有目录中的baseline_vector_gat_finish20260913.py为7505 bytes，SHA
bb6592c809cd35e72bc89c195376ce0622f6f52755712462edb4359fd3c24f08。
只在原训练pane终态0且两训练汇总存在后，调用已冻结cbcec6b驱动执行seed43四诊断。
训练失败或两小时等待上限会退出，绝不重启训练或重复seed42。每15秒检查固定runtime
和数据stat；输出只在阶段变化时写日志。完成后核验两seed文件、两旧joint归档、
八诊断及六数据全量SHA，并写baseline_dense_vector_gat_metrics_20260913.json。
脚本内不能证明自身pane已经退出，因此最终汇总保留该退出状态待独立外部核验。

原diag pane489190退出0后已接续到pane496576，实际Python496579。最新读取日志仍为
WAITING_FOR_EXISTING_SEED43_TRAINING_NO_RESTART；训练pane472305、seed43 Python486258
正常运行至epoch21。HEAD仍979c680；只fetch新Git对象，运行中没有pull/checkout。
启动记录baseline_vector_gat_finish_launch20260913.json，581 bytes，SHA
6e8f9e0c348684bebf3def7ca34fb92d43e00bcd40c150c06332835ba31bf9b6。
一次只读查询将末段SHA抄错导致断言失败，未改文件/进程；已分16字符重新读回正确值。
后续只查看训练和vectorgat20260913_finish.log，不手动再启动第二颗诊断。

### 39.8 两次训练与八诊断整批完成并独立核验

接续实查训练pane472305和诊断pane496576均dead=1、exit=0，无本批Python，HEAD仍
979c680且tracked clean。两次训练与八份诊断都已完成，不重复启动。B5 seed43为
best5/total25、参数285982，正式P/R/F1=0.8309232480533927/0.6821917808219178/
0.749247743229689，upcoming=2/145，等于父near seed43。其训练快照31365 bytes，SHA
fad1a06efc8461bcd208331b85a666bae5252d80fdb17203ca32c0450feffc0e。
两seed P/R/F1/upcoming recall均值分别为0.8332394018044742/0.6844748858447489/
0.7515662275045688/0.017241379310344827；父near命中1/2，新候选3/2，没有稳定改善。

| B5 seed43 | 保存阈值 | upcoming AP | 命中/145或595 | 概率漏报 | 时间漏报 |
|---|---:|---:|---:|---:|---:|
| best validation | 0.65 | 0.00723073 | 2/145 | 143 | 0 |
| best train | 0.65 | 0.00941253 | 1/595 | 594 | 0 |
| last validation | 0.55 | 0.00865408 | 3/145 | 142 | 0 |
| last train | 0.55 | 0.28651585 | 155/595 | 440 | 0 |

seed43两层在四份诊断的全部145或595个upcoming目标上均有实际严格换序；last train/
validation真例分数中位数0.16032501/0.00104852。与seed42同样表现为后期训练判别能力
提高而验证提升有限，不能把动态排序缺失当作唯一原因，也不能据此断言所有图架构上限。

完整整批baseline_dense_vector_gat_metrics_20260913.json为1262573 bytes，SHA
e90919aeaae88f5bb924324bfc7d36451957f028f7dec0c554ba2e9b7f3285ae。
该文件生成时保留自身pane退出状态待核验，现通过独立记录补全：
baseline_vector_gat_final_verification20260913.json，13191 bytes，SHA
4f5679cb13afedbb8885fa70b1d45ebdaee91fba3fafdc60ee0cc927097a879b。
独立核验覆盖两seed实际配置、best阈值及正式分数/计数、八个逐文件结果和权重/epoch/
源码、28当前训练文件、两份旧joint归档22成员、六冻结数据全量SHA与stat、B4不变。
没有替换正式best、没有评test、没有改主仓库；原目标仍未完成。

## 40. 真实故障计划、起点与训练支持的只读审计

39节结束后暂不增加训练候选。先核对过往覆盖：低学习率全模型续训已有v5/134实验，
不是当前208的预训练因果证据；本轮没有据主模型的f180续训差异直接重复该搜索。
新检查从输入可见性入手：本地cfg_disturbance的私有RNG按seed/env/episode生成故障
目标、时刻和时长，injector在到计划时刻后才尝试激活；计划与seed不是当前事件头输入。
源码本身仍不能证明真实208均由该版本生成或事件不可预测，故须核验真实日志。

已只读检查各一份train/validation原始episode_config：文件哈希与冻结cohort audit
相同，保存模式为resample_per_episode，存在完整计划与实际runtime事件。计划中的
gantry序号可与物理resource ID不同，因此只用实际日志ID做起点匹配，不直接猜映射。
准备diagnose_upcoming_schedule_support.py：复用已冻结dense_event_support的299/73
去重起点及595/145窗口目标，验证168个train/validation配置和disturbance日志哈希，
重新解析runtime区间并对照旧审计；用当前生成函数尝试还原保存计划，差异如实记录。

每个upcoming anchor的匹配范围固定为首个未来窗口开始至起点窗口末端（半开区间），
只匹配相同物理resource ID的实际runtime start。另统计之前是否有任何已记录runtime
启动，但此条件不排除QC或其他历史信号。按节点、scenario和节点×scenario统计训练
独立起点支持，不把重叠窗口当独立事件；scenario定义不含run/episode/seed。
这些都是事后解释，不能变成报警过滤器、未来输入或信息论召回上限。

本地3项测试通过：实际目标ID、时间区间边界、重叠anchor的不同边界、先前其他工位
启动的处理、去重起点与窗口目标计数区别、联合场景支持和空集合。未新建目录，未改
模型或训练文件，尚待服务器运行；输出只允许写既有D目录新JSON。

### 40.1 真实运行与历史计划兼容性审计完成

源码02bc0a7d58216ea36a937b7fc4b670e6cf9e2919通过Git对象在服务器内存执行，
未pull运行checkout；HEAD仍979c680、dev_xwt、tracked clean。服务器3项测试通过，
原diag pane513374退出0，无该审计Python。primary输出548216 bytes：
`baseline_upcoming_schedule_support20260913.json`，
`ebeead2dff053e3b8041ad563d3e95519c5a58b3b91c4006b0586cc6e3d86d02`。

| 实际runtime起点对齐 | Train | Validation |
|---|---:|---:|
| Episode | 138 | 30 |
| 去重upcoming起点 | 299 | 73 |
| Upcoming窗口目标 | 595 | 145 |
| 同工位runtime在首未来窗开始至起点窗末启动 | 299 | 80 |
| 上述匹配且此前无任何已记录runtime启动 | 75 | 12 |
| 有匹配的去重upcoming起点 | 150 | 40 |

匹配用实际物理resource ID及半开时间区间，不使用计划target直接猜映射。
“此前没有runtime启动”不排除QC、其他历史信号或seed/配置相关性；时间伴随也不证明
故障造成了该标签。未来计划只用于事后审计，不进入任何网络或报警过滤器。
节点和scenario各自的验证起点均有训练支持；联合节点×scenario有11个去重起点、
22个窗口目标缺少训练upcoming支持，其中10个窗口也匹配未来runtime启动。
联合训练支持≤1/2/3个独立起点覆盖验证窗口56/72/88个，不能按窗口数当独立事件。

### 40.2 生成规则差异，不是数据格式不兼容

168个episode中153个保存resample_per_episode计划，另15个没有重采样计划。
现版函数还原84个、不能还原69个。69个差异中47个只有最短duration变为600秒，
另外22个有其他差异（物流target改动）；按事件字段统计duration变化67处（涉及55个episode），target变化22处（涉及22个episode）。
旧源码c1154978b8b04b9b0db9a042820c177ba6da7fb4的cfg_disturbance.py为31932 bytes，
SHA eabedf6b49a104dc1cafe6f9ba26431a7d1913e3e15d0261dcec3da86058ddd1，
最短时长360秒，且没有现版ensure AGV freeze步骤。只通过Git对象在内存比较，未checkout。

| 保存计划与生成函数的兼容性 | Train episode | Validation episode | Train upcoming目标 | Validation upcoming目标 |
|---|---:|---:|---:|---:|
| 仅旧函数还原 | 52 | 17 | 319 | 105 |
| 新旧均可还原 | 56 | 10 | 190 | 34 |
| 仅现版还原 | 17 | 1 | 74 | 4 |
| 没有重采样计划 | 13 | 2 | 12 | 2 |

所有153个重采样计划至少被其中一个函数准确还原；私有生成器未改变全局random状态。
输出相等仅证明函数兼容，不唯一确定历史采集Git版本。验证“仅旧函数还原”占比更高，
但尚未按固定模型预测进行分组误差分析，不能说这解释了泛化差距。
冻结208数据没有重新生成，所有已完成B4/B5变体之间仍使用同一数据；混合生成规则
可能影响任务难度，却不等于既有模型结果失效或格式损坏。主比较仍需要共同episode和split。

差异记录baseline_upcoming_schedule_plan_differences20260913.json，27247 bytes，SHA
26b4116dc888a1fcb52c7a313f489f9dbb6645ecee8e368e44a65adc251759a6。
历史比对baseline_upcoming_schedule_historical_match20260913.json，26859 bytes，SHA
4890663f63dbb570f4e5591cea441f6380ac39911ed20e785cfcbaa3587cc193。

### 40.3 独立复核与转录修正

独立重验740个目标的唯一身份/实际起点匹配、336个原始配置和扰动文件SHA、168个
train/validation episode集合、60秒窗/步长、新旧生成源码SHA、28当前模型文件及6数据
stat，全部通过。本次不重复上一项已完成的6数据全量哈希；冻结vector整批和独立报告也复核。
终态baseline_upcoming_schedule_final_verification20260913.json，7743 bytes，SHA
17bb7349324fe5747736a89c0d9bc84eac542206808b7cae53bba1643c0c1612。
服务器3测试记录334 bytes，SHA
6b9af0fa3bd128c1b6a7aba5f33cd8a014f363692e8fb7f6f1b190dad3608338。
驱动2088 bytes，SHA 89b4a514a79dcb0cecde5c3c1300ffee9b91f4fcf8cf27288b3bf93cd2c234c1。

核验初稿因两处人工转录SHA不符，在写出前失败；重读每16位哈希纠正后全部通过。
其中旧vector独立复核SHA开头应为4f5679cb，而非先前文档的4f5659cb；三份本地文档
已纠正全文值。产物、权重、训练和模型指标没有改动，不重新执行已完成实验。

## 41. 主0.633与baseline约0.01的差距：先核验正式结果来源

用户要求不要把数量级差距轻率归因于baseline架构。重新读取主引用20c40e2的
modelnote.md第86–90行及F1提升实验历程.md第92–141行：正式prefix8 validation
up_r=0.633、n_up=98（按三位小数可对应62/98，但62是推算，尚非原始字段）；
本轮baseline validation有145个目标。两者名称都是report_recall_upcoming；当前
train.py::_epoch_loop将station_report_metrics结果写入日志，up_r不是hot AP。
公开在仓库里的摘要不是当前208的重评结果，也不是多个seed的均值。

更具体的来源限制：主历程将近窗+远历史precursor列在后续未采用的uphist阶段，而
现版model.py已经包含这条路径。历史源码/配置是否与正式prefix8权重匹配，不能靠
当前分支名、默认开关或文件名判断。旧n10参考权重曾严格加载缺cluster_emb.weight，
也说明“现版有这个组件”不等于“旧结果由这个组件取得”，但n10不是dense的证据替身。
主模型评估指标.md内部还同时出现event_min=5/8和有/无start≤2的相互冲突说明；
实际config和当前代码使用8及start≤2，正式历史权重仍需原始配置/评估源码确认。

通过UU独立Leo SSH重新只读遍历主仓库（不跟随外部符号链接、跳过.git），未找到
文件名dense或匹配范围内位于dense路径的模型/metrics/manifest产物，现有BNPDFormer
权重均为旧n10系列，主HEAD f322dbf9854e06598d0461af2296ab7c095d7f4b。
服务器没有rg，首个只读命令未执行搜索即失败，随后用os.walk完成；未安装工具或改主仓库。
不能用n10的validation0.2698替代正式dense0.633；也不能宣称已复现后者。

当前正式baseline尚无稳定upcoming提升。下一项可在固定208上继续做历史状态分类的
因果输入/支持审计或冻结预测分组分析，但尚未实现/启动；不再仅凭现版主组件列表立即
登记新的训练。主正式数据/权重的缺口限制跨模型因果结论，不阻止已授权208开发工作。

## 42. 固定权重的故障与支持分层诊断

在41节限制下，先检验当前208上漏报是否集中于具体数据条件，而非再登记架构训练。
现有predictions_validation.csv只保存cause/remain预测，occupancy_events是hot头的二值
事件，均不含正式事件头逐工位will概率，不能用它们伪造新的概率分层。

新增diagnose_baseline_events.py --inspect_schedule_strata：在原forward后，将每个
真实upcoming按sample_index/group_id/物理resource ID/起点window/start offset逐项
连接40节冻结审计。按真实未来本地runtime启动、训练节点×scenario独立起点支持、
历史计划兼容性分组，记录保存阈值下命中、概率漏报、时间漏报和概率分位数；兼容性
类别另计算本组upcoming对负例AP，排除ongoing。保存740个目标逐行结果，不以未来
分组过滤推理/报警，也不选阈值。各分组的去重起点可能因anchor边界跨组，不能加和。

本地6 tests/4 subtests通过（新分层3项、既有时间/支持3项）：乱序sample身份、阈值
等号/时间漏报/历史强制start、重复或错位目标拒绝、排除ongoing、训练支持不含验证
起点、诊断标签改变不改变原概率/总命中、禁止test。新driver只读当前B4 joint_onset和
B5 vector_gat的best/last，两seed各自train/validation，共16份新诊断；这不是B4/B5
骨干单因素比较。B4沿用原诊断CPU batch32，B5沿用CUDA batch16；逐份须复现旧诊断
保存阈值、权重/epoch/样本身份、canonical分数/计数和upcoming AP。
服务器保持979c680；新代码拟以固定Git对象stdin执行，不pull。新结果只写既有D目录，
严格校验后才能复用，绝不重启训练。当前本地实现完成，尚待服务器3测试与实际运行。

### 42.1 服务器首两份输出已生成，修复结构化计数校验

诊断源码878082e73e17b86951fa86196a9b27fe17a07bd6通过服务器3项测试；新代码只
fetch Git对象，运行checkout仍979c680。既有diag/v6两pane分别545202/545207启动
B5/B4固定权重诊断，均已生成seed42 best validation的新分层JSON。随后driver对
report_false_alarm_breakdown字典做减法，两个pane均退出1；没有模型训练，也未改权重。
这两份已完成forward的输出必须保留并校验复用，不能重复推理。

本地仅修复driver：scalar计数/分数按原容差比较，结构化误报分类按字典比较，仍排除
连续MAE并单独核对保存阈值。加入driver自身Git身份和额外服务器测试门；原诊断源码
保持878082e，旧3测试记录不改。新增1项回归检查（含2子测试）通过，连同前述检查
本地7 tests/6 subtests。待服务器单项测试后以新resume驱动/日志接续，旧失败驱动/日志
原地保留，不创建目录，不修改已有输出或模型；尚无整批分层结论。

### 42.2 全部16份完成，严重泛化差距不限于未来故障或零训练支持

修复驱动6a6f5b0通过服务器新增1项结构化计数回归检查；原诊断878082e的3项服务器测试已通过。接续时逐项验证并复用两份已完成seed42 best validation输出，旧失败驱动/日志/首两份JSON原地保留，未重复forward或训练。B5 pane549093、B4 pane549099现均退出0，两个resume日志均以SCHEDULE_STRATA_EXIT_CODE=0结束，无相关训练/诊断Python。服务器checkout固定dev_xwt@979c680且tracked clean。

| 无匹配未来同工位runtime启动的upcoming组 | B4 seed42 | B4 seed43 | B5 seed42 | B5 seed43 |
|---|---:|---:|---:|---:|
| last train 命中 / 目标 | 103/296 | 147/296 | 189/296 | 135/296 |
| last validation 命中 / 目标 | 1/65 | 2/65 | 1/65 | 1/65 |

这里的“无匹配”仅指40节固定时间区间内未匹配同工位真实runtime start，不表示完全没有外部影响或历史信号。两个B5 last在匹配未来runtime组train为71/299、20/299，validation均2/80；B4为42/299、96/299对4/80、4/80。两类内均存在差距，不能再将所有漏报解释成预测后突然发生局部故障。

B5 last validation按训练节点×scenario独立起点支持0/1/2–3/4–10/>10分组，目标数为22/34/32/43/14，两个seed命中均为0/0/2/1/0。因此具有非零支持的123个目标也仅命中3个，零支持不是完整解释。训练支持计数含本episode，不能将训练组支持视为跨episode泛化；下一诊断须显式留下整个episode。

B5 last按计划兼容性分组的upcoming-vs-negative AP（屏幕输出保留5位小数）：

| 兼容组 | seed42 train / validation | seed43 train / validation |
|---|---:|---:|
| 新旧均可还原 | .53685 / .01028 | .37843 / .00829 |
| 仅现版 | .30729 / .03760 | .15250 / .03402 |
| 仅旧版 | .38765 / .00789 | .26484 / .00946 |
| 无重采样计划 | .18030 / .00042 | .23918 / .00061 |

精确值见JSON。验证仅现版/无计划分别只有4/2个正目标，不作强推断；新旧均兼容34和仅旧版105目标内仍有差距，不能将混合生成规则直接宣布为根因。未来元数据只用于预测后的分层，不参与输入、选模、阈值或报警筛选。

独立核验重新比对16份新/旧逐文件SHA、保存权重/epoch/样本数/manifest/阈值、canonical分数与结构化计数及AP，重算740目标的命中/概率漏报/时间漏报和分组总数；核对28当前模型SHA、6数据stat、4个保留的原始文件、两份复用输出、两pane终态与进程消失。没有再全量计算上一项已确认且stat不变的数据大文件哈希。

- B4汇总 `baseline_schedule_strata_b4_complete20260913.json`：52487 bytes，SHA `9e2c61aa07590f238b749c8abba229e2a81f55a14127b9928a58e39fb9be5a3a`。
- B5汇总 `baseline_schedule_strata_b5_complete20260913.json`：52666 bytes，SHA `205dd05cd6e63c177c7a124dadd1eb9dc909f201d2b478cd58224c6493ec1aca`。
- 整批独立复核 `baseline_schedule_strata_final_verification20260913.json`：22114 bytes，SHA `c1837f569cb50023ec8647a473440ca406b96be61b360463f9a174db9c2ce340`。

当前正式best不变，B4命中0/1、B5命中3/2（各145）；last未被选作替代。当前负消融之间并非骨干单因素对照。证据确认泛化差距，支持继续查过拟合与可迁移前兆，但尚未区分记忆episode、输入/支持分布差异和表示学习限制，也没有证明架构上限。新增《Baseline_upcoming排查台账.md》把11类当前208训练、已完成诊断、旧134搜索和未解假设分开，作为后续启动前的必查去重索引。原目标继续有效。

## 43. 固定历史状态与跨episode迁移：新诊断登记

按用户要求先查去重台账；本项不重复任何架构训练或已完成forward。要区分两种解释：训练效果依赖同episode统计，还是主模型的粗粒度历史状态本身提供可跨episode迁移的前兆。新增diagnose_upcoming_state_transfer.py，仅读取固定208的train/validation；不更新B4/B5、不选checkpoint或报警阈值。

状态使用主引用20c40e2 cause_cluster.py的点态规则，输入限最后两帧已观测X。分别报告baseline归一化X反变换/缺测mask后的状态和原始共同bundle参考状态，逐样本核验实际差异；不将原始参考自动当作baseline真实输入。四个固定视图为节点、当前状态、节点×当前状态、节点×前一状态×当前状态。统计探针固定用训练格子的经验概率，加1个训练总体正例率的伪观测；不搜索平滑系数。

分别报告训练原样本、整个训练episode留一、原验证集三种AP及没有其他训练正例支持的目标数。留一时同episode所有窗口均排除，连回退先验也不含该episode；验证标签仅计分，不参与任何统计。排除ongoing，仍对照595/145 upcoming及297152/67331负目标和冻结740目标身份。粗状态探针失败不能证明所有输入或架构的上限；训练归一化仍用既有全训练统计，未按折重拟合。窗口AP也不是独立事件置信区间。

本地4项无目录测试通过：随机/阈值边界与固定主函数完全一致；整episode和先验排除、改变被排除标签不影响输出；未见格子只用训练先验；缺测状态与有序状态转移身份。当前准备完成，待服务器同4项内存检查后运行一次，输出只写既有D的新JSON。此项是诊断统计拟合，不是新的正式baseline训练候选。

### 43.1 真实208检查完成：状态输入一致，粗状态统计没有复现神经模型的巨大泛化差距

源码78be1c33c7da3965698e129125755a8bedaf6a0f（SHA 7cf7f10568759a7c11296065ae161c3040d9f9b2d8dfe62951f67e5f4ac6844a）通过服务器同4项内存测试；没有pull运行checkout。固定诊断在原baseline_dense_diag会话pane567897执行一次，168个train/validation episode全部完成，pane退出0，日志STATE_TRANSFER_EXIT_CODE=0，无相关Python。仅读train/validation记录，不更新任何B4/B5权重。

29,298个样本及740个upcoming身份全部核验；在upcoming和负例目标上，baseline X反变换后的最后两帧状态与原始bundle参考的差异为零，末帧缺测也为零。12个差异/缺测计数全为0；两套状态的12项统计结果逐项相同。使用主引用20c40e2的cause_cluster规则，源码SHA b04ad5545bdcc3a5327f1da3094432356e82043fb833c9b7a4bb34a21e700acf；全episode实际状态均与固定主函数精确一致。这仅核验本项离散状态，不代表所有连续通道逐值完全相等或正式主历史权重使用了该组件。

| 固定统计视图 | train原样本 AP | 整train episode留一 AP | validation仅用train统计 AP |
|---|---:|---:|---:|
| 节点 | .00362270 | .00296486 | .00373203 |
| 当前状态 | .00265878 | .00201404 | .00275298 |
| 节点×当前状态 | .00674128 | .00486116 | .00533373 |
| 节点×前一状态×当前状态 | .00883585 | .00555857 | .00634899 |

以上AP是upcoming对负例的排序指标，不是正式upcoming recall；train/validation正例比例约.001998/.002149。状态转移视图在训练留episode后，595个正目标中42个在剩余训练中没有同格子的正例；原验证145个中5个没有。使用当前状态的节点条件视图对应13/1。平滑固定为1个训练总体率伪观测，未调参。

**新约束：**这里没有发现因格式、归一化/缺测处理导致该历史状态信息不一致；粗状态单独的判别力弱。训练留episode与原验证AP相近，没有复现B5神经末轮train约.424/.287、validation约.0082/.0087的数量级差距。因此下一步应定位神经表示学习所利用的、不能跨episode迁移的关联，而不是仅因主源码存在hist_cluster就立即登记一轮加组件训练。粗状态规则不涵盖完整30帧连续信号，不能由此宣布数据不可预测、任何状态嵌入无用或架构上限成立。过拟合迹象更明确，但具体输入/表示成因未定，正式召回未改进。

产物baseline_upcoming_state_transfer20260913.json：16094 bytes，SHA c94d9eef888e0520de75d46d70b1e1f718eb1e2479dbd916f8279cbbede74f5d。独立终态baseline_upcoming_state_transfer_verification20260913.json：1125 bytes，SHA 577a428471369a3af14e64aec2184865eb07186627447b7829d552ff5f1121cf。独立检查24项结果/原与反变换一致性、源码/驱动/服务器测试/日志来源、28当前模型SHA、6数据stat、pane终态与相关进程消失全部通过。服务器仍979c680、dev_xwt、tracked clean，主仓库未改、test未评，原目标未达成。

本轮终态核验前UU前景切到远控屏，窗口检查在粘贴前拒绝操作；通过UU主窗口重新打开Leo独立终端，同一SSH仍在，随后执行原本未发送的核验。没有因此重跑任何实验。

## 44. 冻结编码器与报警头表示：新诊断登记

43节粗状态探针没有解释神经模型的大幅泛化差距。先区分“原输入的可迁移信号在编码器中丢失”与“事件头未能利用已有表示”，不登记新的架构/超参数训练。首轮只取当前B4 joint_onset与B5 vector_gat各seed42 last，分别train/validation，共4次新的表示观察；若没有明确机制信号，不自动扩到其他权重。两者为不同负消融，不能解释为骨干单因素比较。

此前分层JSON没有保存中间向量，无法从现有概率逆推出表示。新增export_frozen_event_representations.py通过只读pre-hook取得实际event_will_head输入，同时保存模型已有node_hidden输出；并从实际归一化X在CPU计算21通道的末帧、30帧均值、总体标准差、末减首，形成固定84维原输入摘要。神经表示各128维，保留所有有效目标节点及sample_index/node_index，包含ongoing但后续upcoming-vs-negative分析排除ongoing。标签仅标注行，不进入表示。

保留原B4 CPU batch32、B5 CUDA batch16、2线程、保存阈值、epoch和权重。每次必须复现上一项冻结诊断的canonical分数/结构化计数及AP，核对28模型文件/6数据stat；原报告与模型不修改。中间向量、逐行预测及完整来源/canonical记录一并写入既有D的新NPZ；另写小JSON索引。若摘要写出中断，可以从NPZ内嵌记录恢复，不能重复已完成forward。运行checkout仍979c680，不pull。

后续分析固定为三个视图（原输入摘要、骨干表示、事件头输入）：用训练拟合的标准化类中心差得到诊断分数，比较train原样本/排除整个episode的类中心/原validation；另在相同物理工位上观察upcoming的最近训练正例/负例，训练查询排除整个本episode。该排除仅作用于诊断参考样本，神经权重本身曾见过训练episode，不能称作神经网络留episode重训或无泄漏交叉验证。验证标签仅用于计分，不选择阈值、模型或表示。

解释规则：若原输入统计能迁移而骨干不能，优先查编码阶段；若骨干可迁移而事件头输入/正式概率不能，优先查后续投影/报警头；若各简单视图都弱，只能缩小具体探针解释范围，不能证明完整30帧信号不可预测或架构上限。诊断不自动转成正式报警政策。

本地4项无目录测试通过：真实B4 joint/B5 vector模型的全部输出及RNG逐值不变，hook移除；行身份/有效mask/三种标签准确且改变标签不改变三个表示或原预测；canonical校验覆盖结构化计数并拒绝改变AP。待服务器同4项内存测试后在既有两会话各运行两份表示导出，不新建目录，不用test。

### 44.1 四份表示缓存已完成并独立核验

导出源码d79a09357fa14fc277d904a1b969ef4409dc7101，SHA 9ce7bf672414716f4eb92664240f419dc85c8af046886e3370a20c0170b1235b，通过服务器同4项内存测试。B4 pane579375、B5 pane579371均正常退出0，无对应导出/训练Python。每个模型train 301938行、validation 68426行，标签顺序negative/ongoing/upcoming为297152/4191/595及67331/950/145。

独立核验逐个检查四份NPZ及内嵌来源/外部JSON，全部sample/node唯一身份和完整split样本覆盖、三个表示维数/有限值、canonical计数和AP；740个upcoming逐行概率/start复现上一项分层结果。B4/B5的原输入84维摘要、行身份和标签逐值完全相等。28当前模型SHA、6数据stat、驱动/测试来源、两pane终态和相关进程消失全部通过，服务器仍dev_xwt@979c680、tracked clean。新观察未改变预测，没有重新训练。

| 既有D中的缓存 | Bytes | SHA256 |
|---|---:|---|
| baseline_repr_b4s42_last_train20260913.npz | 303372444 | 41093930256264f6f1f9a2bf65d4aa19e8180b5f947708827403f0e239234a98 |
| baseline_repr_b4s42_last_validation20260913.npz | 68669464 | bc77427f0c29283e4884645539f7104e8f287496f994d8393dbe437033b98ae7 |
| baseline_repr_b5s42_last_train20260913.npz | 302695986 | cbc1b746b519e3f66e7ec979d660d5e175be9077aa1b48744ca17b04577cf66f |
| baseline_repr_b5s42_last_validation20260913.npz | 68521990 | 66e4b6ad4514000de1a51cae7da8dc7422cac35c711a6dd6b70696686ff43e07 |

独立终态baseline_repr_exports_verification20260913.json：3320 bytes，SHA d9971652a72fc24eb6388482e9c28382fcab226b2d0e76f8f7bd96f269e35f3d。后续直接复用以上缓存，禁止重复四次forward；尚没有表示迁移分析结果。

### 44.2 固定类中心与同工位近邻分析准备

新增diagnose_frozen_representation_transfer.py读取44.1缓存，不加载模型、不做forward。三个视图均用训练标准化后的正/负类中心差作为诊断分数，类先验固定相等；方差小于1e-12的列使用单位缩放，不搜索正则。训练参考episode排除时，该episode也从特征标准化和两个类中心中排除；权重曾见过这些训练episode的限制仍成立。

距离使用训练的无标签标准化尺度、相同物理工位参考池，分别查每个upcoming最近的训练正例和负例；训练查询排除本episode所有参考窗口，同时记录仅排除自身时最近邻是否来自本episode。标签只划分诊断参考类别；validation不参与拟合/参考池，不把近邻规则作为报警政策。原输入摘要不能代表完整时序信号，简单类中心失败也不能证明非线性头无法利用表示。

本地4项无目录测试通过：固定类中心的尺度/常量列；整个episode的标签及标准化统计排除；同episode近邻与跨episode参考身份；缺正例参考明确为None。待服务器同4项内存测试后两个模型各运行一次统计分析；必须通过固定44.1独立核验SHA后才能读缓存，不能重复模型导出。

### 44.3 两模型缓存分析完成：泛化差距在骨干表示阶段已经存在

分析源码f0090c797d71281c945726e103e39a64537ca6df，SHA 479d43109bbd15a6737a6b639cf101082fa87c1f93a3d7a3182318bc790593bd，通过服务器同4项内存测试。直接复用四份已核验NPZ，没有新forward或训练。B4 pane585575、B5 pane585571均退出0，无相关Python。

固定类中心的upcoming-vs-negative AP如下（不是正式召回）：

| 模型 / 视图 | train原样本 | train参考统计排除本episode | validation仅用train统计 |
|---|---:|---:|---:|
| 共同原输入84维摘要 | .00648659 | .00560633 | .01856326 |
| B4骨干输出 | .15969866 | .15597322 | .01212895 |
| B4事件头输入 | .14797274 | .14400098 | .01220315 |
| B5骨干输出 | .20860964 | .20288730 | .00811197 |
| B5事件头输入 | .19487247 | .18953134 | .00832665 |

B4/B5共同原输入统计及近邻汇总完全一致。神经表示在训练集的可分性显著高于原输入摘要，而原validation没有相应收益；差距在骨干输出阶段已经出现，后续context/precursor投影前后相近。正式末轮事件概率AP仍为B4约.014917、B5约.008161；简单更换读取方式没有恢复验证区分能力。

这为“表示学习未能跨episode泛化”提供了更具体的证据，削弱“问题只在最后报警阈值/事件头”的解释。它不是完整输入信息丢失的证明：84维原输入摘要有损，类中心仅是固定线性读取，不能证明任意非线性头或其他训练后的骨干都无效。中列只排除诊断参考统计，神经权重仍曾用全部train训练，不是无泄漏神经交叉验证。

同工位最近邻的描述性结果：

| 表示 | train最近非自身邻居属于同episode / 595 | 跨episode正例比负例更近：train / 595 | 正例比负例更近：validation / 145 |
|---|---:|---:|---:|
| 共同原输入摘要 | 245 | 14 | 4 |
| B4骨干输出 | 377 | 90 | 9 |
| B4事件头输入 | 376 | 87 | 10 |
| B5骨干输出 | 394 | 130 | 7 |
| B5事件头输入 | 389 | 127 | 8 |

全部查询都有同工位正/负训练参考。原输入摘要的等距目标train/validation为24/8，神经视图均为0；不能把等距当任一类更近。骨干表示中同episode近邻占比上升，与训练事件特征依赖一致，但原输入摘要与表示的维数/几何不同，不能直接当作episode记忆的因果证明。

B4分析baseline_repr_transfer_b4s42_last20260913.json：1744090 bytes，SHA 9eb690dba3f6e5f0a544c1274e2486f72eb3999164579f36a315ef9aa1927350。B5分析baseline_repr_transfer_b5s42_last20260913.json：1744389 bytes，SHA ed290623a88742c8f16885b08b6a0d91fa48746aa4d769a1b21200fa894829b6。整批独立复核baseline_repr_transfer_verification20260913.json：14544 bytes，SHA f02a22360b1a89f7356fec2e9a42c334412ff0d548ab9a20a397364bfc0bed6f。

独立复核覆盖两结果/四缓存/摘要/源码与驱动测试来源、18项AP合法性及共同原输入结果相等、4440个逐查询记录唯一身份/参考类别/整episode排除和分组计数重算、两pane退出0与无相关进程、28当前权重SHA/6数据stat。没有重复计算全部距离矩阵或重训模型。服务器仍979c680、dev_xwt、tracked clean，正式best与upcoming命中不变，test未评，原目标未达成。

### 44.4 已核验的正则化位置，与下一项未验证假设

只读加载当前两颗seed42 last checkpoint，确认二者均为单层GRU128，实际gru.dropout=0.0。B4仅存在名为dropout的p=.2模块，作用在第一图层之后；B5另有两层GAT注意力dropout，均为p=.2。GRU输出到history_readout/事件头的路径没有额外dropout模块。与源码中的“仅gru_layers>1才给GRU设置config.dropout”一致。

记录baseline_temporal_regularization_audit20260913.json：3008 bytes，SHA 4772ba27cc5b36a43d4126a213f20176838d0524a91a36811e9715b9dec9acdf。实际权重SHA与44.1缓存来源一致，未执行模型forward。这里不能称整个网络没有正则：已有空间/输入dropout、AdamW权重衰减和早停仍有效；也没有据此断言缺少时序dropout就是根因，更未假设主prefix8用了某种dropout。

已查台账和长报告：旧规模/加大正则搜索指B3；当前208列出的B4/B5变体保持原dropout，尚无GRU读出dropout单项控制。下一项有证据支持的候选是假设“对共享时序表示施加训练时dropout可抑制当前表示层的过拟合”，需保持骨干类型、历史信息、参数量、推理评分和父对照一致，只改训练正则。**尚未实现、未登记具体训练配置、未开训。** 不能把本节当作新训练已经启动或指标已经改善。

## 45. 共享时序读出dropout单项对照登记

44节把泛化差距进一步定位到骨干表示。新增readout_dropout候选，以B4/B5共同near_precursor为父对照；这是针对表示过拟合的单项训练正则实验，不把主源码的某组件缺失当作既定根因。固定p=.2，不扫描概率；在GRU历史汇聚后、所有任务头之前施加一次共享表示dropout，推理时恒等，不修改GRU层数/隐藏维数/图结构/识别头，也不增加参数。

B4/B5各seed42/43，从头初始化。保持near的30×38×27输入及23维近窗摘要、同一数据/split/损失/均匀抽样/优化器和学习率/早停/阈值与正式选模。B4 batch24、lr3e-4、min10/patience10；B5 batch16、lr1.5e-4、min15/patience20；均max60、AdamW weight_decay=.01。Onset、F-beta、history_graph与vector_gat均关闭，B5恢复父对照scalar_additive。不同于当前留存的B4 joint/B5 vector负消融，必须对照已完成near结果2/2/1/2（各145）。

配置默认readout_dropout=0，确保旧checkpoint缺字段时继续按原路径运行。新模块无参数，构造不消耗RNG；p=0的train/eval输出及RNG与979c680旧模型源码逐值相同。p=.2初始所有eval输出与父对照相同，训练时增加dropout随机抽样并影响后续RNG流，这是该训练干预的一部分，不声称训练轨迹仍完全一致。所有头共享被正则化的历史表示；既有near投影保留，不能称对网络每条支路都加了dropout。

本地20项检查/22子检查通过：新4项涵盖旧源码/权重schema兼容、两seed参数/初始化/eval相同、真实训练mask与梯度、配置仅改p及非法值拒绝；另包括两模型无目录配置检查和既有history_graph/vector默认路径回归。未运行创建tmp_path目录的整份artifact_reuse。

准备preflight_baseline_readout_dropout.py：部署后先核验目前28模型SHA/6数据stat、near完整产物和各原archive；从train/validation按首次upcoming（不看预测）取真实24/16样本批。四组须通过同父配置、参数/初始化、初始eval逐值相同、训练mask比例/全损失有限/骨干梯度和GPU内存检查，结果只写既有D的新JSON。部署前再次核验没有旧进程；训练前旧文件经archive_manifest与原文件SHA复核归档，沿用原目录，不新建/删除目录。

拟定唯一标签readoutdrop20260913，尚未部署、未运行服务器测试/真实预检、未开训。完成后按正式best验证命中及整体P/R/F1判定收益，并用best/last train/validation诊断检查泛化差距；无稳定多seed提升不采用，不因单个seed结果自动扩展概率搜索，不用test。原目标仍未达成。

执行器 `run_baseline_readout_dropout.py` 固定四组训练后接续16份best/last、train/validation诊断。训练前检查所有目的文件与旧权重，沿用原归档校验流程，逐组保存训练快照；诊断中断只允许`--phase diagnose`复用已核验输出，不隐式重训。尚未在服务器启动。


### 45.1 服务器预检通过并启动已登记四组训练

部署前核实服务器原979c680、dev_xwt、tracked clean；旧pane585575/585571均退出0，无相关Python。28当前模型SHA与6固定数据stat不变，readout标签尚无产物。随后仅快进BSTAN的dev_xwt至ee838f59d2bdf2893a35ec00acae1308f9dd0e07，4项服务器独立内存测试通过，不创建测试临时目录。

真实预检pane609036退出0，四组均通过共同near配置/损失/训练预算、父near归档来源、初始化权重与RNG、全部初始eval输出逐值相同、训练mask及全多任务损失反传。B4参数273054，B5参数285982，无新增参数；训练丢弃比例约.200，峰值CUDA已分配B4约5.03GB、B5约4.66GB。预检不是新训练或改善证据。

完整预检baseline_readout_dropout_preflight20260913.json：36918 bytes，SHA a357eb599f05e5e3b715ab547c6f1d1ae3ce90ff02305e437d36b39bed3c0f9a。独立核验baseline_readout_dropout_preflight_verification20260913.json：609 bytes，SHA 74a98d2bee7bf07b38ad654633a600842d68da33be57206458da423350067413。独立读取结果、终态、10源码哈希、28当前模型哈希、6数据stat、4父near归档哈希和服务器测试来源；未重复CUDA预检forward。

已启动run_baseline_readout_dropout.py：训练pane/driver610573 live，首组B4 seed42实际Python610616。已读取新control=started及config：readout_dropout=.2、event_precursor=near、源码ee838f5、evaluate_test=false。日志readoutdrop20260913_train.log，训练按B4s42/B4s43/B5s42/B5s43串行，随后自动16份冻结诊断。运行中服务器必须固定ee838f5，不pull文档提交、不重启或并发同批训练；无新的最终指标。

本次是训练层正则化控制，父near正式验证命中2/2/1/2（各145）直接复用，不将当前B4 joint与B5 vector的不同负消融混为父对照。上一代best保存在原目录的已校验archive中。未动主仓库、未用test、未新建或删除目录，原目标仍未达成。

首组随后观察至epoch4，正式中间日志upcoming仍0；不提前判定候选有效或无效。独立启动复核baseline_readout_dropout_start_verification20260913.json：573 bytes，SHA 6b9cbdbd3078c1fcfcde0c743c4053b04ad7dd7fef8c5cfa90c1c0f053e5934f；新config所有模型/训练/损失字段与预检一致，旧archive 11数据成员CRC/SHA及6关键旧文件通过，另三组21当前文件与6固定数据stat未变。此时driver610573、训练Python610616仍live。


### 45.2 B4两次训练完成；补齐父对照诊断覆盖而不重训

B4 readout_dropout seed42/43均正常结束，训练快照status为training_completed_and_files_and_prior_archive_verified，实际control均validation_completed；各12当前文件SHA独立重验通过，batch pane610573仍live。

| B4 seed | best / total epoch | P | R | F1 | upcoming命中 / 145 | 父near命中 |
|---|---|---|---|---|---|---|
| 42 | 4 / 14 | .8333333333333334 | .684931506849315 | .7518796992481204 | 0 | 2 |
| 43 | 7 / 17 | .8103813559322034 | .6986301369863014 | .7503678273663561 | 3 | 2 |

B4整体F1均值.7511237633，与near约.7510203841基本持平；upcoming总命中3/290低于父near4/290，未取得稳定收益。尚无本轮完整训练/验证AP诊断，不能据此判断正则化是否降低表示层过拟合，不能改动已登记B5两组设置。

两快照baseline_readout_dropout_b4s42_training20260913.json：31349 bytes，SHA 3c2c80ec8d13ece9bf262dfe073b479409a54c9a79b97160c3cc33ec90da54d7；b4s43对应文件：31647 bytes，SHA 44936d1611757e6ba87bfb235f72d9135e41727aa86efb9b6702a071b9f39a04。当前模型源码仍固定ee838f5；B5和随后16份冻结诊断继续由原driver负责，不重启。

为做同父对照比较，按四颗near last权重SHA检索D内183份diagnostics或last诊断JSON：只找到B4已有四份last诊断，B5尚无对应last诊断。已找到两模型共8份best headprobe及B4四份last，合计12份，可按原权重SHA/epoch/split/目标计数复用；不能因它们的旧路径后来复用而去加载该路径当前的新权重。

新增compare_baseline_readout_parent.py仅补缺失的B5 seed42/43 last×train/validation四份，冻结父near存于model_before_onsetaux20260912.zip。先等待原pane610573及四训练/16诊断全部正常终态；不同时占用训练GPU、不重训父near、不重复现有12份、不选阈值或test。新增4项本地检查通过，覆盖旧松散checkpoint哈希、新ZIP与member哈希区分、split/epoch/manifest/计数不符拒绝，以及best正式指标一致性。尚未运行服务器检查或启动补充驱动。

补缺驱动源码9862d72ec778dd17fdc06429954b52439863baa2，SHA 1a9bb51118dbade8f5e0e22f900fd50c133cdfa5f77a20f07f2831ec0d5e6b44，已推送。准备只fetch Git对象并在内存检查，不更新运行中源码；但UU在发送命令前读取窗口状态时返回Mac已锁定、自动解锁失败，故服务器检查与补缺驱动尚未启动。已请求用户手动解锁，未重启训练；解锁后先核验原pane610573和结果文件。最后确认的两颗B4正式结果已完整保留。

接续访问状态更新：Mac锁定已解除；但恢复后的三轮检查仍无法通过UU进入独立Leo终端，只能读取远控画面。主窗口菜单、CUA会话重置及键盘菜单切换均无效，已请用户手动打开“终端 - Leooo的MacBook Air”。此阶段没有发送任何新SSH检查/训练/补缺诊断命令；B5与整批诊断状态未知，不将观测通道失败当作进程终止。最后已核验的B4结果、原pane610573及runtime ee838f5继续作为恢复定位信息。


### 45.3 四次训练、候选16诊断与配对父对照全部完成

UU独立Leo终端恢复后，先核验原pane610573，确认四次训练及16份冻结诊断均正常完成、exit=0，没有重启训练。服务器固定dev_xwt@ee838f59d2bdf2893a35ec00acae1308f9dd0e07、tracked clean。仅fetch较新Git对象，未pull运行checkout；父补缺源码9862d72通过服务器4项测试与12份真实已有父诊断来源校验后，在原diag pane654613补齐B5 last四份。该pane也退出0，12份旧诊断没有再次forward，父near没有重训。

| 模型/seed | best/total epoch | 父near正式命中 | dropout正式命中 /145 | 父near last AP：train / val | dropout last AP：train / val |
|---|---|---:|---:|---|---|
| B4/42 | 4/14 | 2 | 0 | .08213004 / .00954082 | .06148423 / .00901112 |
| B4/43 | 7/17 | 2 | 3 | .17993384 / .00947917 | .19555878 / .00963293 |
| B5/42 | 6/26 | 1 | 2 | .30190383 / .00912045 | .22502230 / .00947672 |
| B5/43 | 5/25 | 2 | 2 | .30942241 / .00863192 | .22969678 / .00995029 |

AP表保留8位小数，精确值在服务器完整JSON；AP不是正式召回。B4总命中由4/290变3/290，B5由3/290变4/290，没有稳定实质收益，不采用、不扫描相邻dropout概率。B5正式P/R/F1约为seed42 .85300926/.67305936/.75242471、seed43 .83557047/.68219178/.75113122。

更具体的结论是：本次正则化降低了B5在训练集的upcoming排序能力，但验证排序仍约.009–.010；它没有把被抑制的训练拟合变成明显的跨episode收益。与既有表示诊断一起支持泛化问题，但不能据此宣布“dropout无效所以一定是架构上限”，也不能排除其他正则化或表示偏差。两颗父near best的训练AP本来亦低，需继续区别早期尚未学会与后期只在训练集学会。原目标未达成。

独立候选核验覆盖四训练快照/48当前文件、四旧归档44数据成员、16诊断来源与canonical计数、配置/源码/预检和6数据stat；独立父核验覆盖16父诊断、16候选诊断、12旧结果复用/4新增ZIP成员来源、逐项AP与正式指标、48当前文件、6数据stat及两pane终态。没有重复模型forward或训练。父独立核验首次遇命令字符串换行转义错误，随后拒绝摘要中少一位的服务器测试SHA；已回读实际64位哈希并修正，最终通过。失败核验没有改动模型、数据或完成产物。

- 完整候选 baseline_dense_readout_dropout_metrics_20260913.json：2380302 bytes，SHA b599500b3dd973393d3a997569692959ae747f7a8eea022d2ad5db417f696c78。
- 独立候选核验 baseline_readout_dropout_final_verification20260913.json：17594 bytes，SHA 047740668c40a253e45c30827b9a40ff6a52fa498dc1dec75a99d8725b1166a7。
- 配对父汇总 baseline_readout_dropout_parent_comparison20260913.json：1288372 bytes，SHA 73dbbb0f81f43ac9422e1fb914247113a4936a9983694ea96648a291127b459c。
- 独立父核验 baseline_readout_parent_final_verification20260913.json：694 bytes，SHA b44d551ad06ac86bdc228173a9a7d47f35baf670ff4e5fc251e43edb2d5c3e8f。
- 父服务器测试 baseline_readout_parent_server_tests20260913.json：4053 bytes，SHA 172beaaba88a2cee09e8781ba73022784a3c9974b4839632b91b63281ad2de59。

访问经验：UU原生窗口先Raise，再开“窗口→网易UU远程”；确认主窗口工具栏为Leo后再次Raise主窗口，再点击独立“终端”。原生焦点切换仍可能不稳定，每次粘贴前重新确认终端标题；不能因工具失联而重跑已完成任务。未动主仓库、未用test、未新建或删除目录。


## 46. 真正未参与神经训练的 episode：先登记分组准备

45.3节的配对正则化结果仍不能区分“换新episode普遍难泛化”和“原validation有额外难度”。43–44节仅排除了统计探针参考episode，神经权重仍曾见过其输入/标签；那不是真正的神经训练留出对照。下一方向只在原138个训练episode内部建立独立诊断留出集，原30 validation和40 test分割文件不变，不将诊断数值替换正式baseline结果。

新增prepare_baseline_episode_holdout.py当前只生成分组计划及计数，不训练、不forward。每个原始raw run内部，以固定盐baseline_episode_holdout20260913_v1和group_id的SHA排序，留出约五分之一episode，按最近整数且每边至少保留1个episode；整个episode的所有窗口保持一起。选择函数不读标签/预测；分组冻结后直接复用44.1已有B4两份缓存的sample/node身份与标签做计数，不重复模型推理。原始validation只保留原组身份，不参与留出选择。

4项本地无目录测试通过：改变标签/行顺序不改变分组；整episode隔离和原validation不变；拒绝test、跨split episode、重复sample和单例run；缓存逐行身份/完整覆盖和计数检验。首次将测试置于Python包内部导致Isaac仿真包的toml依赖导入失败，已移到既有source/isaaclab_tasks/test目录，按既有独立工具测试入口通过；未安装依赖或创建目录。

当前仅完成上述分组工具与本地检查，服务器分组计划尚未生成，**没有启动新的神经训练**。实际计数核验后才登记具体诊断训练；须从头训练，并仅在保留fit episode上重拟合归一化。不能拿已见过被留出episode的旧权重充当该对照，也不能按留出/validation结果选择分组。若两类未见episode都弱，支持普遍跨episode泛化问题；若内部留出明显较好、原validation仍弱，支持进一步查split分布；若fit本身也学不好，则不能据此排除欠拟合或训练量影响。任何单次诊断仍不等于架构上限证明。

分组准备源码8bae227417b831704d09532dd9af5de7dab59518已提交并推送dev_xwt。完成45.3收尾之后，UU独立终端又切回远控画面，出现原生控制管道关闭以及窗口变更拒绝；重连与窗口切换尚未恢复稳定终端。新的服务器分组检查/审计未发送，无新神经训练。已请用户将Leo独立终端置前；这一访问问题不影响已核验两批正常终态，也不能成为重跑它们的理由。


### 46.1 分组审计完成：107 fit / 31 heldout / 原30 validation

服务器重新核验仍dev_xwt@ee838f5、tracked clean；原610573/654613均dead=1/exit=0，无相关旧进程或holdout产物。分组源码8bae227仅fetch对象运行，4项服务器检查通过，直接复用B4旧两份缓存的身份与标签，没有模型forward。分组审计及独立逐episode/样本/原run内哈希选择/原validation不变/6数据stat核验均通过。第一次粘贴后窗口切走，回读独立终端仍是旧状态核查命令，未发现该审计已执行；再次明确Raise独立终端、核对整段粘贴后才提交成功。不能把窗口切换当作任务已执行的证据。

| 诊断视图 | episode | 样本 | negative | ongoing | upcoming |
|---|---:|---:|---:|---:|---:|
| 保留fit | 107 | 18095 | 226202 | 3040 | 451 |
| 原train内整episode留出 | 31 | 5764 | 70950 | 1151 | 144 |
| 原validation | 30 | 5439 | 67331 | 950 | 145 |

分组计划baseline_episode_holdout_plan20260913.json：445141 bytes，SHA 459945152e82ae529dd50bcb7b6427951e3b9204f4ea4ef0fdb71b9ac0ab4e2a。服务器测试647 bytes，SHA c7fd43c054009ab06bc13aa14785f57e95a80cb4f527c8cacc09922aa90e7fb8。独立分组核验baseline_episode_holdout_plan_verification20260913.json：894 bytes，SHA c97523220d5bf711d4b678dd95f3a9eefed4207c91a0014eed8c16e35bb44ca6。分组已经冻结，不按结果再次挑选heldout。

### 46.2 真正神经留出诊断的固定方案与输入防泄漏检查

本轮只做B4/B5各seed42，一共两次诊断训练。使用共同near_precursor的原骨干、图、损失、均匀抽样、优化器/学习率/余弦调度；B4 batch24/lr3e-4，B5 batch16/lr1.5e-4，AdamW weight_decay=.01，无新增参数、readout dropout保持父near的0。训练固定到既有最大预算60轮，保留10/30/60三个预先指定checkpoint；两类未见集都不用于早停、选模或调阈值，固定阈值.70仅诊断计分，AP为主要排序观察。每个checkpoint比较fit/heldout/原validation，合计18份诊断；不自动增加seed或换留出分组。固定60轮和取消验证早停是本次诊断设计，不声称训练轨迹等同原正式near，不将107-episode诊断分数替换208正式baseline结果。

新增baseline_episode_holdout_inputs.py：先反变换冻结训练归一化，再只以107 fit episode中的适用且已观测节点计算均值/方差；内存中重归一化fit/heldout/validation，近窗摘要从同一新输入和fit均值重建，连缺测反变换使用的均值也不含heldout统计。原21连续通道以外的类型/劳动通道、图、标签和split不变，test输入不改且不构造test近窗特征。旧float32仿射反变换有通常的舍入误差，不将它当作精确实数重构。

4项新增本地检查通过：改变全部排除输入及标签不会改变fit统计/fit输入与near摘要；更换旧仿射归一化后同一raw输入给出一致新输入（容差内），缺测仍为0、常量列稳定；test输入及所有split/标签不变；拒绝episode跨视图、test进入fit、原validation变化。驱动import与两模型实际配置检查通过。新run_baseline_episode_holdout.py先做真实fit批的两模型CUDA预检，核对完整父near配置/参数、全多任务损失和有限梯度，不做optimizer step；完成预检后才允许固定训练。

新诊断文件只写既有D下专用episodeholdout20260913前缀，旧48当前正式/候选文件保持SHA不变，不覆盖其模型目录。每epoch保存进度，固定checkpoint分别保留；完成权重/已完成诊断按来源复用，部分训练须显式恢复，不能自动从头重启。NPZ内保存完整计分/来源，单独JSON写出中断可恢复，不重复已完成forward。当前代码和本地检查完成，服务器输入检查/真实预检与两次神经训练尚未启动。

判读仍按46节：fit能学到而两类未见集都弱，支持普遍跨episode泛化问题；内部heldout明显较好、原validation弱，支持查原split额外难度；fit也弱则需考虑欠拟合或减少训练支持。比较应保留episode层面的相关性与样本量限制，不因单次阴性诊断宣称架构上限。主prefix8原包缺口继续存在，主仓库和test均不动。


### 46.3 真实预检通过；启动前序列化检查修复，不重复预检

输入处理源码9782fff8d849503f8baac5bf101536c454ac1152通过服务器4项防泄漏检查；运行checkout仍EE，只在既有D保存专用源码副本。真实预检pane705965退出0，fit归一化SHA为7fcb11b26e2ebe8486225f1785e41865b67b94968d7cda88c1472e9c77f149d4。B4/B5的完整父near配置、参数273054/285982、真实24/16 fit样本批、多任务损失及梯度均通过，optimizer step=0。预检不表示性能改善。

完整预检baseline_episodeholdout20260913_preflight.json：237834 bytes，SHA f9fce5742c0cfafe68972ba443ad528795e509ae76a5c88f9f5b8dc4d24bf7da；独立核验baseline_episodeholdout20260913_preflight_verification.json：456 bytes，SHA de6dae22cb143b5f86e9281efc91faec1a1bfc77deaa3fb5b8d092ddda46666d。独立复核两case、18,095 fit索引、归一化哈希、4服务器检查来源、48当前文件、6数据stat和预检终态，全部通过。

随后首个训练driver pane708379退出1，位置在构造优化器之前的保存配置比较：asdict(training)仍含阈值元组，预检JSON已经将其保存为列表；数值相同但直接比较不等。没有完成任何神经训练epoch或optimizer step，也没有progress checkpoint，不能把此次退出当作训练崩溃后的重训。原driver、失败日志episodeholdout20260913_train.log及启动记录原地保留。

仅修正驱动：完整配置先转换为同一JSON表示再比较，不忽略任何科学设置；严格锚定原预检来源9782fff及原driver SHA，并要求输入处理源码、归一化、分组和全部科学协议一致后复用原真实预检。两项新增本地回归通过，覆盖真实阈值tuple/list往返兼容和学习率/阈值/损失等改值拒绝，及旧预检复用时拒绝输入源码/seed/分组/归一化/epoch协议变化。**不重复真实CUDA预检**，待服务器同两项检查与原预检真实配置检查通过，再使用新接续driver/日志启动原两次尚未开始的训练。


### 46.4 接续成功，B4已核验至第7轮真实优化器更新

修复源码ecc39e81c8f14483eaa50d90beea9743f9ac485c的两项服务器回归与两份真实已保存preflight配置检查均通过。旧预检、原driver、失败日志和旧启动记录均保留；没有重复真实CUDA预检。新D内driver为run_baseline_episode_holdout_resume20260913.py，SHA dc0742115f6eaac42a791449d73f9dc97129ce0af2fa7e1ba1c12765f9140e9a。仍只fetch对象，运行checkout保持dev_xwt@ee838f5、tracked clean。

已在原baseline_dense_v6 pane712854启动固定两模型批次，日志episodeholdout20260913_train_resume.log。B4已完成7/60轮；独立读取当时进度checkpoint的同一份字节，核验身份、fit归一化哈希、plan SHA、near/readout0配置、完整优化器状态及实际5278次更新（7×ceil(18095/24)）、权重/损失有限。第7轮训练损失1.41988306759668。这个中间loss不是upcoming改善证据；尚无本轮未见集诊断指标。

独立启动核验baseline_episodeholdout20260913_start_verification.json：974 bytes，SHA 4c1d62e2477879cb6808c8ebbdbff9473c46f4ce53a13e6946edfa6e244195b5。同时核验712854实际命令与live状态、48原模型文件/6数据stat不变和4旧失败/预检文件SHA保持。进度checkpoint是本轮可变文件，后续epoch会替换它，记录中的SHA只标识本次观察字节，不能要求其永远不变。首次只读检查使用Python -I导致训练库路径被忽略而无法import torch；已改用与训练相同的PYTHONPATH并通过，只重试读取检查，没有中断或重启训练。

服务器修复检查baseline_episodeholdout20260913_resume_server_tests.json：699 bytes，SHA 5a1866fc6f0e05be8398c8987788ec15c24513da13fafbffba4154d79d47cc67。新启动记录baseline_episodeholdout20260913_resume_launch.json：2526 bytes，SHA ac1b1432f176b54599434afa15ad325a60f22d6623e78593f74331b58aa3344d。

原driver将B4训练至60轮，再用预先保留的10/30/60权重各做fit/heldout/原validation三视图；完成后自动接续B5 seed42同一固定方案，共18份诊断。两个未见集不参与本轮神经梯度、归一化拟合、早停或选模。结果不替换正式208 baseline权重/分数，test未评。当前必须保持runtime EE，不pull、不重启原任务、不额外并发本批诊断；部分训练失败必须依据已保存进度明确恢复，不能从头重跑。完整终态产物将为baseline_episodeholdout20260913_complete.json，现尚未生成，原根因目标未达成。

### 46.5 登记整episode缓存统计，等待18份诊断完成后执行

训练期间新增analyze_baseline_episode_holdout.py，仅复用本轮18份NPZ和canonical JSON，不做模型forward或神经训练。固定512次bootstrap，分别在fit/heldout/原validation各视图的每个raw run内部有放回抽样整个episode，保持每个run的episode数量；seed按三个视图为20260913/14/15。同一视图的抽样权重在所有模型/轮次复用。不将同一episode的相邻目标窗口视为独立观测，不挑选轮次或阈值。

预先报告10/30/60三轮的upcoming AP、固定.70且起点容差3的召回、每episode及每raw run计数，以及fit-minus-heldout和heldout-minus-original_validation的AP/召回差值及95%百分位区间。区间只描述固定单seed权重、当前分组及run条件下的episode重采样波动，不代表重新训练/新raw run的不确定性，不据此宣布架构因果或上限。无正例的重采样记录为未定义并报告有效次数，不填0；按episode抽样后仍计算与canonical相同的窗口级micro AP，不混为每episode宏平均。

4项本地检查通过：带概率同分和重复episode的加权AP与实际展开重采样后的canonical指标一致；保持每个run的完整episode计数与确定性；无正例区间处理；读取缓存时排除ongoing/无效节点并正确区分概率报警与起点错误。只读源码确认原validation本身也在raw run内随机分组，不能预先称为按时间划分的晚期验证集。当前该分析只登记及本地检查，服务器尚未执行；现有训练不改动，分析须等本轮完整产物后再运行。


### 46.6 缓存统计已部署，等待原两模型固定训练正常结束

分析源码d4a92c1aaba81b3c4c6d54ec08f104fb4d89d698只fetch Git对象，运行checkout仍dev_xwt@ee838f5、tracked clean。D/analyze_baseline_episode_holdout.py SHA 368a4b13092173298f7a5619cab79f6dc74990ca207353ea5082f5ee757a57d6。4项服务器检查全部通过，测试记录baseline_episodeholdout20260913_analysis_server_tests.json：660 bytes，SHA b9a84b3c81f0d7bc9830c62855141dbf8b992eec0f3602a3671cd07751b60bc0。统计还将核验18个NPZ/JSON、6个固定权重和两个progress的来源、原sample_index SHA，以及48原模型/6数据stat/12运行源码保持。

部署前确认原训练712854 live、旧diag705965已exit0、专用源码/日志/启动记录尚不存在；只将已结束的diag pane复用为727985，训练没有重启。新日志episodeholdout20260913_episode_analysis.log输出WAITING_FOR_EXISTING_TRAINING 712854|0|。等待器要求原PID、完整两模型汇总和训练pane exit0，才执行纯缓存统计；若训练提前停止则报错，不能自行重跑。完整命令保存在baseline_episodeholdout20260913_analysis_launch.json：4332 bytes，SHA 86c1abda1d2bd48a5a205259839a2c73a69410835dfd1e014f0a9aaa61b93960。

独立读取实际727985命令、两live pane、分析源码SHA、4服务器检查记录及等待日志通过。核验文件baseline_episodeholdout20260913_analysis_wait_verification.json：695 bytes，SHA 07edec499c6cee79cfbe88f766e6f32e7acb4ea1641c6f44903d3704f4e2f45c。此时B4最新日志第31/60轮、loss1.073933892396314；这仍不是未见集性能。B5及18视图/统计尚未完成。后续预期产物baseline_episodeholdout20260913_episode_analysis.json与baseline_episodeholdout20260913_analysis_verification.json生成后仍须读取内容及核验终态，不能把已启动等待器当作分析已完成。原根因目标继续未达成，不用test、不改主仓库。


### 46.7 实时等待与统计结论范围复核

本次goal接续将上一轮归为实际进展（新增缓存统计已部署、4服务器测试及等待进程核验），当前对原handle再读：训练712854、等待727985均live，B4日志39/60轮，等待日志仍指向原712854。没有完整两模型汇总或未见集指标，不重启、不将等待当作根因已找到。

只读实际冻结plan：三视图各21个raw run。每run episode数直方图，fit为{2:11,5:1,6:3,9:1,10:2,11:3}；heldout为{1:15,2:2,3:4}；原validation为{1:15,2:3,3:3}。因此46.5固定run的episode重采样对两未见集中各15个单例run没有抽样波动。该区间仍是已登记的条件性描述，但不能用它宣称跨新run、不同训练seed或不同留出分组的稳健性，也不能把零宽区间理解为没有不确定性。既有分析和训练协议不变，没有按结果修改分组或选择轮次。

进一步核对现有源码而非重复数据审计：diagnose_upcoming_schedule_support.py::match_future_runtime显式要求event.target==onset.resource_id，故42节“无匹配同工位未来启动”仍允许其他资源新启动的影响，不能写成所有未来外部扰动已排除。factory_baselines.dataset::_split_groups在raw run内shuffle whole episodes，原validation不能预先称作按时间留出的较晚分布。上述是对既有证据适用范围的核验，不是新模型结论；主prefix8匹配原包仍缺，不据此重复主目录搜索或旧优化方向。

### 46.8 登记107-fit正例支持计数，避免混淆训练支持减少与过拟合

第40/42节支持统计使用全部138原train，不能直接代表本轮107-fit对两个未见集的支持。新增audit_episode_holdout_support.py只读固定plan与已有dense_event_support20260912.json（SHA 56d2dc07169d12e420ed1d2877bcf3290d2cd53d0abdd317ea6581845b32cd93），按fit的独立upcoming起点统计node/scenario/node×scenario支持，分别描述heldout与原validation中支持数≤0/1/2/3/5/10的独立起点及窗口目标数量。这是本轮新分组的计数，不重复原完整138的已完成模型诊断；不读概率、不forward、不训练，不更改分组或模型。

复用旧summarize_support_cells实现，但供给训练计数的只能是107 fit episode，两个未见集分别只用于描述目标。4本地检查通过：heldout不能充当fit正例；独立起点和两个重叠窗口不混计；节点支持与节点×scenario支持区分；另一未见集标签不影响本视图支持；拒绝跨视图、缺失coverage及test。相同工位/场景有正例支持也不等于历史特征匹配或可预测，不能把低支持计数当作根因证明。当前登记与本地检查完成，服务器计数待执行；原训练712854/统计等待727985不变。


### 46.9 107-fit支持审计完成：两未见集零联合支持比例接近

源码42735664e8e2c36482021fc3b2e1f8e7e3555a99只fetch对象，运行checkout仍EE；4服务器检查通过。复用固定plan及旧独立起点记录完成新107-fit计数，再用独立Counter重新计算两视图×三支持层级×六计数上限，逐项匹配。原训练712854/统计等待727985均live，B4当时第51/60轮；没有新增神经训练或forward。

| 视图 | episode | 独立upcoming起点 | 窗口目标 | 同node×scenario零fit正例：起点/窗口 | 有联合支持窗口 |
|---|---:|---:|---:|---:|---:|
| fit | 107 | 226 | 451 | N/A | N/A |
| heldout | 31 | 73 | 144 | 13 / 26 | 118 |
| 原validation | 30 | 73 | 145 | 15 / 30 | 115 |

两个未见集单独node和单独scenario的零正例支持都是0。joint零支持比例26/144=18.06%、30/145=20.69%，没有明显的validation单侧类别缺失。原138-train时validation零joint支持22/145，本轮排除31 episode后成为30/145，不能继续套用旧22计数。有训练joint支持的窗口分别118和115；支持数≤0/1/2/3/5/10的窗口数分别为heldout[26,58,93,107,115,138]、validation[30,64,90,99,111,141]，说明大多数目标的同类独立起点支持仍稀疏。这是描述，不是根因或可预测性证明，也不是按结果重选分组的理由。

审计baseline_episodeholdout20260913_fitting_support.json：56707 bytes，SHA 97b05d4f2a6f224a61865cc0aaa83772f6deee544274639f01bfd99e3cf52be4。四测试baseline_episodeholdout20260913_fitting_support_tests.json：580 bytes，SHA 9112783a053fd97c8ad87e79281dc73b65d33d69fe65fdae354b8c4cd61b4cfb。独立核验baseline_episodeholdout20260913_fitting_support_verification.json：598 bytes，SHA b665482580473af0fd76aa89e5eecd0a7a48ab54425f94c7b54f156eda31c2c4。完整stdout保留episodeholdout20260913_fitting_support_run.log。

后续须先读三个预定轮次的神经诊断。若两个未见集都差，可复用本轮NPZ按零/正joint支持分组检查漏报是否仍在有支持目标中普遍存在；此时不是重做旧138-train权重的42节诊断，而是新107-fit神经权重的支持解释。当前还没有本轮未见分数，不认定模型架构限制或根因已找到。


### 46.10 登记新权重的零/正joint支持缓存分组

针对46.9计数，新增analyze_holdout_support_predictions.py。每个模型必须先完整完成60轮及9视图，之后只读6个未见视图缓存，固定10/30/60轮、.70阈值、起点容差3，按训练node×scenario独立起点支持为0或>0分组，报告upcoming/negative数量、AP、命中/召回、概率/起点漏报、正例概率分位数。两模型分文件保存，允许B4已完成后先分析其缓存，B5继续原训练；不重复任何forward，不选择轮次或改变模型，不把结果替换正式208分数。

所有支持键必须从107-fit的完整独立起点重建，不能直接用46.9仅含query正例的coverage cell列表去划分negative，否则“query只有负例、fit有正例”的cell会被错分为零支持。本地代码审阅时即修正此未部署的分析路径，并加入明确反例测试。5本地测试通过：样本/节点身份正确、排除ongoing和无效节点；概率与起点漏报分解；query标签不改fit支持；空正例分组为未定义；完整fit cell包含负query类别且不借用heldout正例。已有46.9计数未用这种负例划分，不受影响。

需要真实模型缓存出现后校验NPZ与JSON/完成汇总、固定split与源SHA，且分组总目标/命中/负例须复现canonical，零支持目标须复现26/30。当前源码/本地检查完成，服务器测试与分组执行尚待完成。


### 46.11 B4完整诊断：真实未见episode同样低，训练拟合升至AP .931

原训练712854完成B4固定60轮/45240次optimizer更新，随后完成预定10/30/60权重的全部9视图，已自动接续B5。分析源码9551920bfe8994a64a481002df781291632dfd5a通过5服务器测试后，只读B4六个未见缓存完成支持分组，没有重复forward。独立复核B4完整60条history/更新数、9JSON+NPZ/三固定权重/progress来源、计划完整覆盖、分组汇总回归canonical和26/30零支持目标均通过；48原模型文件、6数据stat、12运行源码保持。

| 固定轮次 | fit AP | heldout AP | 原validation AP | fit命中/451 | heldout/144 | 原validation/145 |
|---|---:|---:|---:|---:|---:|---:|
| 10 | .0385378219 | .0071823703 | .0096086259 | 20 | 1 | 2 |
| 30 | .6773719248 | .0072721763 | .0070589213 | 289 | 7 | 4 |
| 60 | .9311272255 | .0076739832 | .0078973105 | 402 | 7 | 7 |

第60轮fit recall=89.14%，两个未见集分别4.86%/4.83%。随着训练继续，fit正例概率中位数从.0320→.8361→.9700；heldout从.00656→.0001068→约4.5e-7，原validation从.00832→.0000306→约1.2e-7。排序没有同步改善，模型对未见upcoming的否定却越来越强。以上显示用真正训练排除的episode仍有严重泛化差距，不再只是旧冻结表示探针的“排除参考episode”。三个checkpoint都报告，不选择最有利轮次，也不把固定60轮诊断成绩替换正式208基线。

九视图所有upcoming timing miss为0，遗漏均在固定.70事件概率阈值；第60轮fit49漏报，heldout137，原validation138。支持分组进一步限制解释：无joint支持组26/30个目标在三个轮次均0命中；有支持组heldout为1/118→7/118→7/118，原validation为2/115→4/115→7/115，末轮分别5.93%/6.09%。有支持组末轮AP分别.01068068/.01120561。因此原validation特殊难度或零node×scenario正例支持不能单独解释本次B4的低召回。正类别支持不等于匹配历史前兆，不能据此排除更细粒度分布差异。

当前可确认的是B4强烈的跨episode泛化失败/过拟合模式，而非完全没有表达upcoming的能力，也不是起点解码使大量已报事件失配。尚不能把它等同为“具体记住了某个episode字段”的因果证据，不能认定某个组件缺失、输入信息上限或所有B4架构的上限。早期10轮的两未见排序也弱，单凭缩短训练不能宣称足以修复。主prefix8匹配原包缺口仍限制.633的架构比较。

B4完整baseline_episodeholdout20260913_b4s42_complete.json：1837618 bytes，SHA 77c45d0250c7ac44c63c224266d2143613f9e8d57d7a9a94e1e60d2ec858b0d5。支持分组baseline_episodeholdout20260913_b4s42_support_strata.json：7708 bytes，SHA 369324c208cc9d6613af2e8c8e87ea4a3e72b50b44e4eb5cdb58807051e5ae93。5服务器检查baseline_episodeholdout20260913_support_strata_tests.json：669 bytes，SHA fb6a895f99fc11a03b171570ff3675d2f3a378b96165ec54115f2078215dc74a。独立终态baseline_episodeholdout20260913_b4_complete_verification.json：7494 bytes，SHA 008d408c3e7771e14ab75cdefcf248e3f8e6da5ea215de9095f1c37f2f6870c5。上表/分位数是屏幕精简显示的舍入值，完整精度保存在上述server JSON。

核验时B5已到第3/60轮，原712854与自动bootstrap等待727985均live，运行checkout仍EE。继续等待B5原60轮+9视图，之后执行已部署且5检查已通过的analyze_holdout_support_predictions.py --source_commit 9551920bfe8994a64a481002df781291632dfd5a --model b5一次；B5支持分组没有另外挂等待器，需要接续时确认完成后运行。全18视图episode bootstrap仍由原727985自动接续，不手动再开。B4所有训练/诊断/分组均完成禁止重跑，test未用，原目标仍未达到。


### 46.12 用已保存AP检验单一概率尺度解释，不重扫阈值

B4未见正例概率随训练趋近0，但低召回是否仅因阈值尺度不合适需要单独限定。旧34节两头前沿针对138-train的另一批权重，不能当作本轮107-fit三checkpoint的数值。新增bound_holdout_probability_calibration.py只读每模型完整9份JSON中的AP和U/O计数，不读NPZ或forward、不选择/扫描概率阈值，给出整体report precision≥.80时、单一全局阈值或保持排序的单调校准的必要上界。

令U为upcoming总数、O为ongoing总数、h为某阈值正确报告的upcoming。即便给所有ongoing正确报告，TP≤O+h。P≥4/5要求FP≤floor((O+h)/4)=F。在去掉ongoing的upcoming-vs-negative排序中，该阈值至少包含h个正例、至多F个负例；第j个正例的精度贡献至少j/(F+j)，所以AP≥sum(j/(F+j),j=1..h)/U。概率同分的组精度也不小于这个把所有负例提前放置的下界。忽略阈值后的全部正例、允许所有ongoing及起点完美，都让该条件偏乐观；因此最大满足条件的h只是可能命中的上界，不是可达到的具体阈值性能。

适用范围仅固定同一事件分数的全局阈值/单调重标度；节点特定校准、增加头或特征、重新排序/重训均在范围外。该界不能称为信息或架构上限。使用已固定的整体P≥.80要求，不拿算出的h选择checkpoint、阈值或参数；不变更既有.70诊断报告。

4项本地检查通过，包括3up/2ongoing/2negative的全部210种排序、32种正例起点成功组合与7个阈值逐一验证可行命中不超过界；另查同分、ongoing精度余量、低AP和参数拒绝。当前源码/推导/本地检查完成，服务器检查及实际B4代入待执行。B5实时核验第11轮，原712854/727985仍live，旧任务不重跑。


### 46.13 B4单一概率校准不足：不扫描阈值也可排除50%召回

源码01a940005a6ca5eb98d6cbd0b8347e37cba88eae只fetch对象，4项服务器检查通过；用B4完整已核验JSON的真实AP/U/O完成9视图解析界，再以50位Decimal独立枚举所有整数h、重算各h的必要AP下界并核对最大可能h，全部通过。没有扫描预测概率阈值，没有NPZ读取/forward/训练，没有采用新阈值。

| 固定轮次 | fit最多可能命中/451 | heldout最多可能/144 | 原validation最多可能/145 |
|---|---:|---:|---:|
| 10 | 178 | 24 | 26 |
| 30 | 451 | 25 | 22 |
| 60 | 451 | 25 | 23 |

这是46.12推导的必要、刻意乐观的上界，不是某个可达到的阈值表现。条件为整体report precision≥.80，且仅对同一事件分数作全局阈值或保持排序的单调校准；允许所有O个ongoing和所有起点完美预测，忽略阈值之后全部正例的AP贡献。末轮heldout上界17.36%，原validation15.86%，三个checkpoint均排除50%upcoming召回；不能把这结论扩展到节点特定校准、新头、重新排序或新训练后的架构。

要在两未见集取得至少一半upcoming且P≥.80，按计数得到的必要AP下界分别约.05177574/.06148197；当前末轮只有.00767398/.00789731。因而B4的问题不仅是.70阈值相对概率尺度偏高：目前未见排序本身不足以支撑所需的高召回/精度组合。这个结论补充46.11的过拟合证据，仍不证明具体特征捷径或架构上限，也没有替换正式baseline成绩。

解析文件baseline_episodeholdout20260913_b4s42_calibration_bound.json：6486 bytes，SHA 986033e42fd2348e326dfafa7bff659652af1b3db35eb8a4eb9a2e676255dc22。4检查baseline_episodeholdout20260913_calibration_bound_tests.json：584 bytes，SHA 9a9c5decfbdce7b77c3a2de5b52940c9030dc93a8f94adc6fb8499285044b42a。独立50位复核baseline_episodeholdout20260913_b4_calibration_bound_verification.json：1212 bytes，SHA e445130ddc155c02a91098294a3a434b9d1ccd4d689d2a2f108dbbe6cc3250e9。完整stdout保留episodeholdout20260913_b4_calibration_bound.log。

本次末尾实时核验B5已至38/60轮；原训练712854与bootstrap等待727985均live，运行checkout仍EE。继续等待原B5与18视图统计，不重启或更改运行源码。B5九视图完成后，已部署的analyze_holdout_support_predictions.py（9551920）和bound_holdout_probability_calibration.py（01a9400）各用--model b5执行一次；各自5/4服务器测试已经通过，不重复部署/测试或B4分析。原根因/合理改善目标仍未达成，test未用。


### 46.14 两模型整批完成：真正未见episode的泛化差距在B5重复

B5固定60轮/67860次更新及9视图完成；原等待器随后自动完成两模型18视图的512次episode统计。复用已部署9551920/01a9400及已通过的5/4服务器测试，仅新增B5支持分组与解析校准界，没有重复B4分析或模型forward。最终独立验证两模型60条history/更新数、18JSON+NPZ、6固定权重及两progress、完整汇总与两模型汇总一致、全部分组/统计点估计回归canonical，B5九个解析界用50位Decimal重算；48原模型/6数据stat/12源码保持。

| B5固定轮次 | fit AP | heldout AP | 原validation AP | fit命中/451 | heldout/144 | 原validation/145 |
|---|---:|---:|---:|---:|---:|---:|
| 10 | .0195154460 | .0071893368 | .0082614642 | 4 | 0 | 2 |
| 30 | .4048642061 | .0088962632 | .0128235672 | 158 | 2 | 9 |
| 60 | .6578702526 | .0094881962 | .0111405635 | 274 | 7 | 4 |

B5末轮召回fit60.75%、heldout4.86%、原validation2.76%。正例概率中位数fit约.83138，对两个未见集约.00020505/.00008761。支持分组heldout零支持命中10/30/60轮0/1/2（各26），有支持0/1/5（各118）；原validation零支持均0（各30），有支持2/9/4（各115）。末轮有支持仍仅4.24%/3.48%。B5全局阈值/保持排序的单调校准在整体P≥.80下的乐观命中界，heldout24/27/28（各144）、原validation24/30/28（各145），三轮都排除50%upcoming召回；界不表示可达到，也不是架构界。

全部18视图upcoming timing_misses=0，独立逐项验证report_recall_upcoming×U==n_matched_who_upcoming，故本批两种命中计数实际相同。一般情况下who命中不扣起点错误，不能将这个字段无条件当作report TP；本批结论依赖已核验零起点漏报。旧缓存分析中与who计数比对的保护断言在本批均通过，没有漏算起点错误；未来若有非零timing miss，必须改用report口径再接续，不能直接沿用这一等价。

512次固定run内episode重采样的末轮AP差值95%区间：B4 fit-heldout [.902598,.943781]，B5 [.612845,.681642]；heldout-original_validation分别[-.002870,.002593]、[-.004431,.001170]。这与巨大的拟合/泛化差距、一致较弱的两未见排序相符。区间仍只条件于21个固定run、当前分组和每模型单个seed，两未见集各15 singleton run，不当作跨新run/seed因果证据。全部三轮点估计与区间保存在完整统计中，没有据区间选轮次。

整批结论：两个不同图骨干都能显著提高训练upcoming拟合，但对真正排除训练/归一化的episode迁移不足；原validation特殊难度、零类别支持、起点解码和单一概率尺度均不足以单独解释。已确认泛化失败/过拟合模式，还不能锁定具体捷径、缺失组件或架构上限。保持208正式数据/分割及对比身份；107-fit仅诊断，不替换正式baseline。下一步先核对是否查过容量设置，再登记保持GCN/GAT+GRU结构的单项容量控制，检验限制记忆容量是否改善泛化；当前未登记/启动新的训练。

训练pane712854 dead且tmux exit0。统计pane727985 dead但tmux exit字段为空，实际/proc/727985/stat显示状态Z且第52字段exit_code=0，独立确认Python正常结束、等待tmux回收；不能仅据空字段称任务live、失败或重启。最终记录保留这一实际状态，两任务都不再执行训练/分析，runtime仍EE。

- baseline_episodeholdout20260913_b5s42_complete.json：1838365 bytes，SHA fa89dd945e0fa6c3b8f751a6bbd7e612b133d32bb9b876c34b1a3cc98a8c021d。
- baseline_episodeholdout20260913_complete.json：4422978 bytes，SHA acd1a6dc11ba2b204c0c2871c6adb998229e59ae379c19f72cf4802d6e099713。
- baseline_episodeholdout20260913_episode_analysis.json：264129 bytes，SHA 3a3279cebe56193bb59cdf217a35de225a273e54dbf91e43704cf7759c6924fa。
- baseline_episodeholdout20260913_analysis_verification.json：562 bytes，SHA 614845a114051ebfc579689d470654f215e6546a044cc5c958030852cecae0ea。
- baseline_episodeholdout20260913_b5s42_support_strata.json：7716 bytes，SHA ddb521f9f15b62d2b5d7243d84887517e7eb45b5a82b1a13852d86833671f9f5。
- baseline_episodeholdout20260913_b5s42_calibration_bound.json：6475 bytes，SHA 8b8d96f4435d63c78dbd876ef8119c07d6db92d652746f5e5f0686e48e2af8fc。
- baseline_episodeholdout20260913_final_verification.json：6740 bytes，SHA 0071a30c1af9e8d557853270a62ba868d60c18f7f8804a0de3ff6042fb2a5707。

所有数值为读取时的精简显示舍入值，完整精度/来源在server JSON。最终独立记录status=two_models_eighteen_views_bootstrap_support_strata_and_calibration_bounds_verified；未用test，原完整根因/改善目标尚未达成。


## 47. 单项GRU容量对照：已登记，尚未预检或训练

### 47.1 去重核验及假设

第46节真正未见episode整批完成后，服务器只读盘点D/models内B4/B5路径下config.json及其ZIP配置成员。baseline_gru_capacity_history_audit20260913.json为53098 bytes，SHA 7d032b4dffde116e1e96790840e9b1562f103f874732204d9769c7cdacd1b67c。匹配当前208 manifest的50份配置/50个配置SHA、其他manifest的56份/56个SHA，gru_hidden全部128，0份缺字段跳过。这是已保存配置的盘点，不把106份实例说成106次独立搜索，也不将所有其他manifest自动归为旧134；此范围没有已完成的缩小GRU宽度对照。

已确认两模型fit好、两个未见集差，单加读出dropout也未改善。当前检验容量是否偏大、缩小容量能否改善迁移；不是已经证明记忆了某个特征，更不是宣称缺某个组件。保持B4 GCN+GRU和B5 GAT+GRU身份，父对照统一使用已完成near_precursor，两seed旧成绩/冻结诊断复用，不重训父模型。

### 47.2 预先固定的唯一候选及评价协议

- variant=gru_capacity32，archive_tag=grucapacity20260913。唯一模型配置差异gru_hidden 128→32；既有宽度依赖的history_readout及预测头同时缩小，不能声称只有GRU权重改变。B4参数273054→31230，B5 285982→44158；GRU自身两者均74496→9408。
- 两层GCN/GAT空间hidden64、B5四head/原scalar注意力、单层GRU、last_mean、near23输入、node_embedding0、原dropout=.2、readout_dropout=0保持；不添加新输入、loss、头、图交互或采样策略。不同尺寸导致初始化/预测和后续RNG消耗可能不同，只保证相同seed下未改空间前缀初始化一致，不宣称完整配对权重或初始输出相同。
- 恢复原138 train/30 validation/40 test，23859/5439/6886样本及原训练归一化；107-fit仅诊断，不作为正式候选数据。当前manifest e3d7b2008ad7c5d0844c10a4c0670ff36c5ba961382706695689daf7a050244f。test不评估、不参与选择。
- B4/B5各seed42/43，从头训练共4次，B4 batch24/lr3e-4、B5 batch16/lr1.5e-4，原AdamW weight_decay=.01、max60/cosine、原最小轮数/早停/validation选择与阈值流程全部保持。只比较此128→32单点，不扫描宽度；不因中间结果改变训练轮数、损失或采样。
- 完成后best/last×train/validation共16个冻结视图，与现有near相同model/seed/checkpoint/split对照，报告AP、正例概率、正式report upcoming命中、整体P/R/F1。正式命中取report_recall_upcoming×n_true_upcoming，不能无条件取who命中。训练AP下降本身不是改进；需未见排序和正式召回跨seed共同改善且维持整体报告质量，才有采用依据；last仅诊断，不作为另一次选优。没有显著稳定改善则记录负结果，不把一两个命中增量说成突破。

### 47.3 实施与未完成事项

本地train_dense_baseline_control新增配置分支和明确容量元数据；模型类本身未改。新增3项实质检查全部通过：旧EE全部变体配置不变、两seed空间前缀一致/参数减少、真实30×38×27+near23输入/节点mask/事件梯度路径。新增preflight_baseline_gru_capacity.py及run_baseline_gru_capacity.py，通过源码编译检查。实际CUDA预检必须核对父near存档配置/输入/训练/损失、原48模型文件与6数据stat、四组真实train batch反传和validation形状/有限性，无optimizer step；通过并独立验证后才能启动。沿用原4输出目录和tmux会话，先验证归档再替换训练文件，不创建/删除目录。

服务器第46节训练/统计均已正常结束，运行checkout仍EE；本节此刻只有本地实现/测试，尚未commit/deploy、服务器测试、真实预检或开训。必须先复核无旧任务运行、固定源版本后FF更新，再执行一次新预检。旧readout负消融松散文件由baseline_dense_readout_dropout_metrics_20260913.json（SHA b599500b3dd973393d3a997569692959ae747f7a8eea022d2ad5db417f696c78）定位；不会误把它当near父权重。主仓库保持只读，无新主模型资料搜索。主prefix8不同cohort/缺匹配正式包的比较边界保持，原完整目标未达成。


### 47.4 真实预检通过并已开训：B4首组第12轮核验

服务器先核验原训练712854 dead/exit0、原统计727985 dead且/proc Z/exit0，无相关活跃Python；GPU0空闲28696 MiB。48原模型与6数据stat通过后，将BSTAN dev_xwt从EE快进至90a41a5b9c44e625f8c80083a8a950cd50f5eb13。3项服务器测试通过；真实四组CUDA预检pane776564正常exit0，参数/输入/损失/父存档与空间初始化/梯度路径均通过，无optimizer step。峰值B4两seed约206.24/213.87 MiB，B5均264.78 MiB。

独立核验14源码、48旧文件、6数据stat和106配置盘点通过。旧near JSON没有序列化后来增加且默认关闭的开关，独立验证从已核验源码AST读取实际默认值补齐后，四组唯一有效差异均为gru_hidden；不是数据或启用组件发生变化。实际CUDA预检未重跑。

原baseline_dense_v6会话已复用为pane783323、实际训练Python783366，首组B4 seed42核验到epoch12；旧readout训练文件在model_before_grucapacity20260913.zip的11成员及原控制记录逐项核验，另外3组各12旧文件不变。配置31230参数、near/GRU32、原训练/损失/test=false均正确。运行中轮次来自stdout；此训练器在训练结束后才写history.csv，不应因暂缺CSV当作训练失败。当前前12轮日志upcoming召回仍0，不能据中间轮次改方案或宣称候选最终无效。

训练日志grucapacity20260913_train.log；driver run_baseline_gru_capacity.py负责原四训练后16冻结视图与完整baseline_dense_gru_capacity_metrics_20260913.json。运行期间checkout固定90a41a5，禁止pull、重启训练或重新执行本节预检。部分失败必须复用已完成快照并定位，不能自动从头训练。

新增compare_baseline_gru_capacity.py仅配对现有near16缓存与新候选16缓存，3本地检查通过，覆盖非零起点漏报时who/report差别、保存阈值和16唯一父文件映射。尚未服务器测试或实际配对；需等整批完成后使用真实候选SHA执行一次，可通过固定Git对象源码使用，不更改运行checkout。父诊断全已具备，不需要任何父训练或forward。每个seed单独呈现，不将两seed同一145目标误说成290个独立标签。

- baseline_gru_capacity_preflight20260913.json：41087 bytes，SHA 4256633509d19c0539fc56055febb8b15d347e553e890e4633dc701387d5e510。
- baseline_gru_capacity_tests20260913.json：280 bytes，SHA eb58a4775758156a7573efcdc811c33d5a7fb868a458a3a477fb0271ade9b65b。
- baseline_gru_capacity_preflight_verification20260913.json：1438 bytes，SHA 68c1b9abb5b59fc6e7c540736c1d31b187db1ef5945f9dd0880ebe56d74c8d84。
- baseline_gru_capacity_launch20260913.json：588 bytes，SHA ceaace0244945d39cc3690dacc612bb37e8e2c190d73841e43a125848a0b42b7。
- baseline_gru_capacity_start_verification20260913.json：1708 bytes，SHA c184265762118a0f61f141686847795ceecebe679994d0180481d96fde8578ad。

实际任务已启动而非仍待批准；仍无新候选完整分数/可采用提升，test未用、主仓库未改，原根因或合理改善目标保持未完成。
