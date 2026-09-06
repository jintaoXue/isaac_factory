# Baseline 迭代实验报告

状态：进行中，尚非正式横向实验结论。当前版本仅整理验证证据；后续补齐实验、
选型、独立评估及对照一致性审计后才能冻结。未将目标标记为完成。

最新进度（2026-09-06）：第 18 节困难负例对照 12/12 完成，两模型均选回控制组，
没有综合提升；赢家的保存分数复现及前缀解码敏感性复核完成。本轮所有训练正常退出，
没有启动新的调参轮次。当前 v5 共 48 次验证运行，连同旧 v3 共 160 次开发运行。
随后第 20 节四个固定权重的 train/validation 诊断 8/8 完成，保存的 validation 指标
全部复现。第 21 节预注册训练抽样对照，当前不把未完成候选计入上述训练数量。

最新前置变化：随后 fetch 到 dev_tyx 的 `20c40e2`，主实验文档已更换数据包和事件定义。
这不是同一任务上的新增高分，不能与本报告 v5 直接排名。新产物路径和最终定义待确认，
具体源码核对、重评参数遗漏与后续迁移要求见第 19 节。尚未修改 baseline 的标签/评分。

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

最新补充：`baseline_validation_20260906_round2.json` 已收齐原 B5 搜索的 16 次、
B4 context/focal 对照的 8 次训练；总快照包含 88 次已完成 validation-only 运行。
原 B5 最佳仍是 `candidate_c0_stabilized`。B4 新轮按既定稳健排序选中 context，
其 F1=0.2848，低于 control 的均值 0.2916，但跨 seed 波动较小；不能写成平均效果提升。
Focal 单独和与 context 组合的 F1 均值分别为 0.0584、0.2095，upcoming recall 均为 0。

`baseline_main_episode_split_20260906.json` 实测双方 134 个 episode 身份和分配均一致，
train/validation/test 交集分别为 92/17/25；本批数据未触发第 5 节的软链接 split 风险。
`baseline_train_symmetry_20260906.json` 在 13723 个训练样本中只发现 62 个存在所检查的
精确工位交换对称，且对应工位的事件标签无冲突。不支持把差距解释为大量精确同输入异标签。

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

`STUDY=sampling batch_factory_baseline_staged.sh B4|B5` 沿用现有执行入口和执行副本，
不新建源码目录。study_config 在开跑前固定三组配置；每个运行的 metadata 保存
抽取数、目标窗口数和期望抽中比例。默认 factor=1 保持旧 loader 的精确 shuffle
顺序，不额外遍历标签；B3 也可显式使用同一训练选项，但本轮仅运行 B4/B5。

验证：67 tests、7 subtests passed，含真实 B4/B5 一轮训练、默认顺序复现、seed
确定性、固定抽样长度、有效节点门禁、禁止借评估集事件和 train-only 产物记录。
完成全部候选后按现有 validation 稳健排序选型，重做保存分数和前缀历史敏感性核验。
尚未得到本轮分数，不承诺这一训练因素会提高 precision 或 F1。
