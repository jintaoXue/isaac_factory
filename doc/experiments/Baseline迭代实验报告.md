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
