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

最终报告还缺：B3 完整调参、当前 v5 数据上的必要重训与后续改进、主实验正式多 seed
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
