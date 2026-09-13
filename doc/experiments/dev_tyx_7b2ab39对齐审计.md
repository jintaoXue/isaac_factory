# dev_tyx 最新代码对齐审计（7b2ab39）

**审计后的用户决定：**继续固定我们的208个episode，暂不处理双方数据来源差异；任务范围完整采用她最新start≤5/10/15三档，并对齐Min8、20格和will15评估协议。下文“正式路线待确定/等待共同包”的审计时建议已被该决定覆盖，不得继续以此阻塞工作。保持B4/B5骨干；新协议显式版本化，原Start≤2结果保留。通过UU已只读核验服务器主仓库实际HEAD也是7b2ab39。协议适配及34项本地检查、6子检查通过；真实208预检已6/6通过并独立核验，服务器19项协议检查通过；六阶段训练队列本地实现并登记，尚无新训练成绩。

本次范围：按用户要求获取 dev_tyx 最新代码，核对 baseline 与主实验要跑的数据和共同协议。未合并、未切换分支、未启动训练，未访问或修改服务器。本地仅增加审计记录；已有用户 DOCX 保持原样。第 49 节数量训练不因本次代码审计自动恢复。

- 本地工作分支：dev_xwt，审计时 HEAD `6f43a2bf33bfe4e127abd1942301a4f9d60b590b`。
- 已执行 `git fetch origin dev_tyx`；远端跟踪引用由 `20c40e230aedee6aef2429d352413fbcf0fa571a` 更新到 `7b2ab393e838c1b54a78b2f125d41e0066d53ce0`。
- 新提交时间：2026-09-13 17:33:56 +08:00，标题“fix: 三档实验达标，继续微调”。标题不替代指标产物核验。
- 下文主模型源码行号均属于该提交的 Git 对象，不能对应到尚未合并的本地同名文件。
- 审计以代码、继承展开后的配置及文档为证据；没有打开真实 test 数组、运行 test 评估或根据文档 test 分数选择 baseline 参数。

## 1. 当前到底有几套数据/任务

| 体系 | 数据证据 | 任务/状态 | baseline 处理 |
|---|---|---|---|
| 现有 baseline | 固定 208 episode，138/30/40；23859/5439/6886 个窗口 | 历史 30 分钟、未来网格 15 格、Start≤2、Min8；已完成大量排查 | 保留数据、划分、缓存与旧结论，不重跑旧搜索 |
| 文档旧 prefix8 | `dense_i1`，204 条目录项，140/28/36；20826/4331/5354 个窗口 | 旧 Start≤2/Min8，主表为工位＋起点容差 3 的 report F1 | 不能与 208 结果当作同数据架构比较 |
| 新三档 Min8 | 实际提交的 runner/config 指向 `raw_data/dense_i1` | Start≤5/10/15，未来网格 20 格；有旧匹配配置和新 opt 配置两组 | 需要先确定共同数据清单及正式配置，不能只更改 baseline max_start |
| 新 clean 研究入口 | `run_start15_min5.py` 的 audit 阶段指向 `raw_data/n10_i1_all_usable`，jobs_total≥10 | Start<15（代码 max_start=14）、Min5、20 格；计划 seeds 42/7/21 | 训练引用的三份 clean 配置不在提交中；实际训练数据路径、episode 数和划分尚不能确认 |

最新提交没有带来我们 `tools/factory_baselines` 的 B4/B5 实现更新，也没有提交上述 raw_data 包及共同 split manifest。发现新的 clean 路径不代表它已经成为双方正式实验数据。

**重要的新证据：**`实验问题汇总.md:447` 明确写出旧 dense_i1 含重复 episode 和跨 split 重复，相关实验只作历史结构复现/诊断，不能当作无泄漏论文主结果。这是文档披露，本次没有重新读取服务器原包量化重复比例，也没有证明全部新运行仍使用完全相同的物理文件。旧主模型较高 upcoming recall 因而需要重新核验数据可比性，不能直接作为 baseline 架构缺陷的证据。

## 2. 必须先对齐的项目

| 项目 | 最新代码事实 | 对照要求 |
|---|---|---|
| episode 身份与划分 | 主 loader 仍按 episode 名称、raw run 前缀分组；没有在该路径中按内容哈希去重 | 冻结同一份 episode 内容哈希、排除清单、train/val/test manifest；同一内容不能跨 split |
| split seed | `train.py:857` 使用 `cfg.get("split_seed", run_seed)`；三档继承展开后均没有 split_seed | 多训练 seed 时显式固定 split seed/manifest。不能以为 `--seed` 只影响训练 |
| 历史输入 | X 历史 30 格仍相同；主 dataset 仍计算窗口前额外历史的 precursor 摘要 | 先冻结双方允许读取的历史范围。区分共同输入信号和各模型自己的处理模块；不能只用 X 维度相同证明完整输入相同 |
| 标签观察范围 | 旧 baseline 为 15 格，新三档为 20 格 | 新标签必须生成匹配的数据/标签版本；不能直接复用旧 15 格目标或缓存称为新三档成绩 |
| 热状态平滑 | 新支持 legacy 与 close_then_filter；三档没有设置，实际默认 legacy | 正式配置显式记录顺序。Min5/hazard 文档中的 close_then_filter 不能自动套到 Min8 三档 |
| upcoming/ongoing 分母 | 新 `station_report_metrics` 有 hist_last_hot 时按最后历史格冷热分类，旧版本按 y_start 是否为 0 | 对齐分类定义并重新核对分母。冷历史＋未来首格起热可属于 upcoming，不能只看起点索引 0 |
| 主指标与选模 | opt 设置 `report_primary=will15`、`ckpt_metric=will15_f1` | 同时明确报告 who/will F1 与严格 report F1@1/@2/@3；两方采用同一主指标及 validation 选模规则 |
| 原因任务 | cause 输出编号为兼容旧包保留，但支持类由六类减为四类 | 对齐 loss ignore、推理可选类、macro recall 支持集；不能仅比较输出维数或将四类/六类分数混用 |
| 剩余时间 | opt 设置进度加权训练，且 `remain_eval_primary_phase=middle_weighted` | 保留全样本、无权全局 MAE；中期加权/分段 MAE 单列，不能与 baseline 全局 MAE 混比 |
| 训练预算与初始化 | 新 runner 为课程式热启；事件窗口过采样默认继承 4 倍；三档权重不同 | 记录所有前置训练、当前更新次数与抽样方式；统一冷启动或说明并公平分配预训练/微调预算 |

双方已具备可复用的共同基础包括分钟级窗口、30 格 X、Min8/ongoing 最短 1 格的标签机制、逐工位事件输出、validation 选阈值与冻结 test 的原则。新数据的实际节点顺序、27 维特征值、归一化、额外历史范围仍需以共同包核验，不能将旧核验结论自动移植。

baseline 当前 `factory_baselines/evaluation.py` 明确固定参考提交 20c40e2、max_start=2、start_tol=3。`factory_bn_shared/remain.py` 和 causes 包装器实际导入本地 canonical 主模块，因此直接合并最新主模块可能改变旧实验的分母/原因支持集；不能仅保持旧配置文件就宣称协议未变。

## 3. 本次发现的具体口径问题

### 3.1 “Start≤15 / Min8 / 20 格”不能覆盖全部声明起点

新三档同时配置 max_remain_windows=20、occupancy_horizon_windows=20、event_min_windows=8。`node_event_targets` 只检查截取网格内的连续热段，要求该段长度至少为 8。

只读提取最新函数，在内存构造单工位、冷历史、持续 8 格的合成例子，没有使用真实数据或模型：

| 未来网格长度 | max_start | 能生成正例的起点索引 |
|---|---|---|
| 20 | 15 | 0–12 |
| 23 | 15 | 0–15 |

因此索引 13–15 在当前 20 格目标里不足 8 格，必被过滤。这里严格使用代码的零基索引，不能直接当作已经核定的自然语言分钟边界。若协议要求最大起点索引 15 且完整观察 8 格，至少需要 23 格；也可以另行制定明确的删失标签规则。两种都属于共同任务协议修订，需要主模型和 baseline 一起更新，不能单边修 baseline 后横比。

此外 `will15_*` 在 `remain.py:1076` 附近只是 `who_*` 的别名，仍使用当前 max_start/min_windows 生成的真值。start5、start10 配置不会因为指标名叫 will15 就自动变成“全未来 15 分钟是否发生”。

### 3.2 同名脚本已不是文档中“只变起点”的实验

`run_start_horizon_min8_ep100.py:23–59` 现在默认 screen=15 epoch，main=100 epoch；调用的是 `12_3_*_opt` 配置，按 5→10→15 顺序将上一档 best 作为下一档 init_ckpt。第一档继承 prefix8 权重。

三档 upcoming/far-start 权重依次为 9/10/11；opt 的 patience 为 40，报告和选模使用 will15。文档 §12.3 则描述旧 `12_2_*` 同一 prefix8 热启、仅 max_start 不同的 100 epoch 对照。不能用现在的同名 runner 声称复现了旧匹配对照。

这不要求 baseline 复制 PDFormer 权重；应冻结各模型自己的初始化策略、训练来源和预算。尤其换 split seed 或换数据集后，旧 prefix8 是否见过新 validation/test 内容必须先核查。

### 3.3 交接内容尚未闭合

- `run_start15_min5.py` 引用的 `FactoryBN_clean_start15_min5_base.json`、`FactoryBN_clean_start15_min5_hazard.json`、`FactoryBN_clean_will15_min5_hier.json` 均不在最新 Git 树中。
- `run_start5_min5.py` 及实现文档引用的四份 `FactoryBN_dense_start5_min5_{base,hazard,hazard_nocluster,legacy}.json` 均缺失；文档所指 `实验问题汇总.md` 第 15 节也不存在于本提交。
- `实验问题汇总.md` §12.4 所指两份 `12_3_start{10,15}_min8_precision.json` 不在提交中，不能当成现有 opt 配置。
- `prefix8_best_metrics.tex` 明说取自留存笔记，工作区没有原始 last_metrics.json。因此本次更新仍未补齐旧 prefix8 的原包、原配置、对应 checkpoint 和原始指标的证据链。
- 部分旧 ablation runner 也引用已删/未提交的配置。本次仅登记缺口，不执行这些入口。

## 4. 哪些属于模型差异，应保留

保持 B4 GCN+GRU、B5 GAT+GRU及已登记共同任务适配。主模型的状态嵌入、时空注意、prefix 解码、hazard/分层起点头属于需解释或消融的处理路径，不能为追分强行移植。

三档 opt 未显式关闭 fuse_hist_cluster，主模型默认值为 true；但新增 hazard 和 hierarchical 开关默认 false，三档没有开启。不能把“代码增加某组件”误写成“最新三档已使用该组件”。

第 50 节已核验的原始状态信号一致性仍是旧固定包的证据；它不证明新主实验完整输入范围已经对齐。已有跨 episode 泛化失败结论也保留，但目前仍不能把准确根因确定为架构、数据量或旧主模型数据重复中的单一因素。

## 5. 后续最小执行顺序

1. 先冻结正式路线：Min8 的 5/10/15 三档，还是 clean 数据上的 Min5 研究路线；不能混用两者标签、配置或结果。
2. 取得该路线的实际数据清单、内容哈希、去重/排除记录、固定 split、有效配置及对应 checkpoint/validation metrics。若继续遵循用户固定 208 的决定，优先让主模型在同一 208 清单上形成匹配对照；不能自行换回存在重复问题的旧 204 包。
3. 用 train/validation 做标签边界和共同输入核验，先解决 20 格/Min8/Start15 的矛盾，再版本化 baseline 的数据生成、指标与选模适配。封存旧产物，不覆盖原 208 实验版本。
4. 若正式选择三档，每个模型对应 5/10/15 三个任务配置，核心 backbone 保持；共同训练 seeds/预算冻结后再计总运行数。不能把当前主脚本的单 seed 与另一个多 seed 研究入口混为正式要求。
5. 先做一档共同 train/validation 核对并检查 upcoming 分母与指标字段，再扩展其余档。test 不参与调参；已完成的 208 排查与预测缓存保持复用边界，不重启旧批次。

本次完成的是代码审计及合成边界验证。服务器数据是否更新、clean 包实际数量、新三档实际 checkpoint 的身份与成绩尚未核验，不把源代码配置当作已完成运行的证明。

## 6. 可定位证据

共同源码前缀为 `source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer/factory_bn/`，以下均固定在 `7b2ab393e838c1b54a78b2f125d41e0066d53ce0`：

- `configs/FactoryBN_dense_12_2_start5_min8_ep100.json:4`：dense_i1、20 格和 Min8。
- `configs/FactoryBN_dense_12_3_start5_min8_opt.json:5`：will15 主指标、精度/召回门控、加权剩余时间。
- `run_start_horizon_min8_ep100.py:23`：screen/main 预算、opt 配置、顺序热启。
- `train.py:857`：split_seed 回退到训练 seed；`:733`：按 who/will15 或 report 选择阈值。
- `remain.py` 的 node_event_targets、station_report_metrics：窗口内时长过滤、will15 别名及冷热分组。
- `dataset.py` 的 split_episodes_by_name、_build_samples：按名字分割、历史/远历史输入、整 episode 平滑标签。
- `model.py:543`、`:591`：新解码器默认关闭、hist_cluster 融合默认开启。
- `run_start15_min5.py`：clean audit 路径与缺失配置名。
- 仓库根 `实验问题汇总.md:447`：204 数据重复披露；§12.3/12.4：历史三档与 precision-first 结果。
