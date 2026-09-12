# Baseline 交接与远端执行指南

更新时间：2026-09-12。先读本文件，不需要把旧对话全部装入上下文。

## 1. 交接状态

### 最新用户决策：先固定 208 episode 优化 B4/B5

用户已授权沿用现有 baseline 208 episode，先查明并改善 upcoming 低召回。
主模型 204-episode 包暂不阻塞这一开发实验；正式共同重评仍保留该材料缺口。
首轮注册 `near_precursor`：仅使用原 30 分钟范围内的近窗摘要，B4/B5 各 seed42/43，
原骨干、损失、抽样、评分和选模规则不变，最多 60 epoch。详见长报告第 28 节和 JSON
`upcoming_208_resume`。不要把下方较早的等待主包记录当作新的用户停止指令。

近窗首轮四次训练已全部正常结束；B4 upcoming 均为 2/145，B5 为 1/145、2/145。
B4 best/last 各四份诊断均已结束且权重哈希通过；last train upcoming AP
0.0821/0.1799，validation 仍约 0.0095。独立事件支持诊断已完成：train/validation 的
595/145 个 upcoming 窗口目标对应 299/73 次去重起始事件；未来至起点才出现本地
扰动的目标分别为 300/82，不能据此推定不可预测或骨干上限。详见长报告 28.3。
用户随后要求主模型架构对照，源码及实际图边表核查见第 29 节：两跳未覆盖的目标节点
对为零，不能归因为看不到其他机器；主模型另有 onset 专门监督与占用预测补报路径。
STGNPP 在主参考配置中关闭。同一 best 权重的事件头/hot 头诊断已完成 8/8，
代码 `00d4c6f`，权重/源码/manifest/epoch/样本数核验通过，validation 原指标一致。
固定 hot 阈值额外覆盖的 upcoming 为 0/0/1/0，同时多出 33/16/11/21 个负例越阈值；
不采用直接补报。最新核验两项 tmux 均 dead=1、exit=0，无对应 Python 任务。

下一轮已登记 `onset_aux` / `onsetaux20260912`，以近窗方案为父对照，只加独立 onset
辅助头和按正负类分别取均值的 BCE（系数 1.0，ongoing 排除，缺类贡献零）。辅助头
不参与正式报警；骨干、历史信息、抽样、其他损失、阈值与选模不变。B4/B5 各 seed42/43，
最多 60 epoch，复用原目录。实现 `df9ee0e` 本地 29 tests / 11 subtests、服务器
11 tests 通过。B4 两颗 seed 已完成（best1/5、total11/15），正式 upcoming 为 0/145、
3/145，F1 均值 0.7447745；B4 best/last 的 train/validation 共 8 份诊断已全部核验。
辅助头 last train AP 为 0.282862/0.535091，validation 为 0.031005/0.032333，仍有明显
泛化差距，尚无正式提升。B5 seed42 实际 Python PID=104896，最近观察到 epoch10；
seed43 尚未开始。新普通路径随批次转为 onset_aux，未开始目录仍为近窗。服务器固定
训练源码 df9ee0e，不在训练中 pull 后续文档提交；实际状态必须重新查询，不能重复启动。

### 2026-09-12 新对话续接：输入审计已完成

用户已授权继续优化；下方旧对话停止记录仅用于追溯，不是本轮停止指令。
本轮先通过 UU/Leo SSH 核实 BSTAN 干净 `dev_xwt@6b2b216`，两项 dense 会话均
dead=1、exit=0，未发现相应训练/诊断进程。随后只部署输入审计代码 `2d6bbd0`，
本地及服务器各 3 项边界测试通过。未启动模型训练、未改原 v6 数据或权重。

服务器现有 benchmark 中新增 `baseline_precursor_history_audit_20260912.json`：
train 23859/validation 5439 个样本全部通过样本身份、历史时间范围和前缀不变性检查；
有非零远历史摘要的样本分别为 23721/5409，有完整额外 30 窗历史的分别为 19719/4539。
未读取 test episode 特征数组、未使用标签。全部节点的统计不能当作 upcoming 指标。
详情见长报告 27.11 和 JSON 的 `resume_input_audit_20260912`。

主仓库当前仍为旧 `f322dbf`，其 `raw_data` 和 `libcity/cache/model_cache` 中无
`dense*` 目录。`origin/dev_tyx` 新 fetch 仍为 `20c40e2`；正式 204-episode 包、实际
split、`dense_i1_a1_prefix8/BNPDFormer_best.pt` 和运行配置仍缺现有路径。
已向用户询问这些路径；不要将旧 n10 权重冒充 dense 对照。下一项输入对照只有预案，
尚未开训，也没有宣称共同输入已经对齐。

### 旧对话收尾记录

用户要求完成当前运行、停止旧对话自动优化、整理资料，在新对话继续。
这不是“优化目标已完成”。不应在没有用户继续指令时启动新实验。

- 服务器训练会话 `baseline_dense_v6`：dead=1、exit=0。
- 服务器诊断会话 `baseline_dense_diag`：dead=1、exit=0。
- 收尾时未发现 `train_dense_baseline_control.py`、`train_b2/3/4/5_*`、
  `diagnose_baseline_events.py` 或 baseline 调参 Python 进程。
- 最后权重 train/validation 诊断 8/8 完成，数据、权重哈希、epoch 和样本数核验通过。
- 没有停止其他人的进程，没有清理目录，没有新建仓库或实验目录。
- 运行代码为 `79a5df6`；三分类训练代码为 `33628fc`。交接文档提交在其后，
  最新文档 commit 用 `git --no-pager log -1` 查询，不混作训练代码版本。

## 2. 背景与约束

研究任务是制造工厂瓶颈预测，主实验为同组同学的 BNPDFormer，baseline 为：
B2 XGBoost、B3 LSTM、B4 GCN-GRU、B5 BSTAN-style GAT-GRU。
当前重点提升 B4/B5 的提前预报，并保证主实验与 baseline 可公平比较。
不承诺每种模型达到指定高分，不通过修改事件定义、放宽容差或筛除困难正例造高分。

必须遵守：

1. 只在 `dev_xwt` 改代码；服务器只写 `/home/sci/work/BSTAN_isaac_factory`。
2. `dev_tyx` 和服务器 `/home/sci/work/BNPDFormer/_isaac_factory` 只读参考，不能改动。
3. 不再按 commit 新建目录或 worktree。复用已有目录，允许新增有意义的结果文件。
4. 旧结果用已有归档逻辑及 Git 保留；禁止覆盖未归档权重，禁止自动删除旧目录。
5. 不加无效旧版本兼容路径。二分类/三分类是有实验意义的对照，不是兼容兜底。
6. 调参只看 train/validation。当前 dense 搜索未评 test；不得用 test 决定候选。
7. 每次报告说明数据版本、split、seed、权重来源、阈值和统计方式。
8. 不改服务器其他人的训练、仿真、VPN、SSH 配置；不把登录密码写入文档或 Git。

用户曾询问旧 `BSTAN_v4_*`、`BSTAN_v5_*` 目录是否可迁移删除，但明确没有授权删除。
只读核查发现若干是 detached Git worktree；`BSTAN_baseline_dev_xwt_v5` 是独立仓库。
ignored 数据、依赖和软链接尚未完整核验，不能凭 tracked clean 判断可删除。

## 3. 工作目录

本地仓库：

```text
/Users/xuwantong/Desktop/hku courses/capstone/code/isaac_factory
```

服务器仓库：

```text
/home/sci/work/BSTAN_isaac_factory
```

当前 benchmark（目录名 v3 不等于张量版本）：

```text
/home/sci/work/BSTAN_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3
```

原始数据只读来源：

```text
/home/sci/work/BNPDFormer/_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset
```

同一服务器 `/home/sci/work/isaac_factory` 可能指向主实验仓库；不能误当 baseline 工作目录。

## 4. 数据与评估

| 项目 | 当前 baseline dense v6 |
|---|---|
| 数据版本 | `factory_baseline_dataset_v6` |
| 标签 | `factory_ops_event_30m_to_15m_dense_v2` |
| 评估契约 | `factory_dense_i1_eval_v1` |
| Raw 审计 | 217 个 episode，接纳 208，拒绝 9 |
| Episode split | train/validation/test = 138/30/40 |
| Sample split | 23859/5439/6886，总计 36184 |
| 名义输入/预测 | 30 分钟历史、15 分钟未来，每窗 1 分钟 |
| Train 目标 | ongoing 4191、upcoming 595、negative 297152 |
| Validation 目标 | ongoing 950、upcoming 145、negative 67331 |

目标计数是相关的窗口-工位目标，不是独立物理事件数。全样本共享导出对齐检查通过。
旧 134 个 episode 扩充至 208；不能与旧 v5 的 F1 约 0.3 直接比较来宣称模型提升。

当前 manifest SHA-256：

```text
e3d7b2008ad7c5d0844c10a4c0670ff36c5ba961382706695689daf7a050244f
```

当前计分规则：

- Upcoming：真实 start 为未来索引 1 或 2，持续至少 8 窗。
- Ongoing：真实 start=0，历史末窗 hot 且未来延续，剩余至少 1 窗。
- Report 命中：工位正确，起点误差不超过 3 窗；持续时间不是该命中的硬门槛。
- Precision=命中报告/预测报告；Recall=命中报告/真实事件目标；F1 为调和平均。
- Upcoming recall 单独按真实 start>0 的目标分组，总 P/R/F1 同时包含 ongoing/upcoming。
- 起点/持续时间 MAE 在成功匹配工位的目标上计算，单位分钟；无支持样本为 N/A。
- Remain MAE 在所有样本锚点计算到订单全部完成的剩余时间误差，单位分钟。
- Hot F1 是未来窗口-节点格子的指标；will AP 是全事件排序，不能替代 upcoming AP。
- Validation 阈值搜索先满足 P>=0.80 再比较 F1，否则按 F1；当前可行门为 P>=0.80、R>=0.70。
- 起点容差比 1--2 窗提前范围更宽；upcoming recall 不是严格的提前量准确率。

当前 v6 未评 test，但其中部分历史 episode 曾在旧版开发中被查看；不宣称全新未触碰测试集。

## 5. 当前结果与诊断结论

以下为 validation 的 seed42/43 均值，不是 test，也不是挑单个最大值。

| 模型/候选 | P | R | F1 | Upcoming recall |
|---|---:|---:|---:|---:|
| B4 原 history 控制组 | 0.8078 | 0.6968 | 0.7482 | 0.0103 |
| B5 原 history 控制组 | 0.8298 | 0.6826 | 0.7490 | 0.0138 |
| B4 三分类候选 | 0.8296 | 0.6872 | 0.7515 | 0.0034 |
| B5 三分类候选 | 0.8459 | 0.6785 | 0.7530 | 0.0138 |
| B4 近窗摘要候选 | 0.8221 | 0.6913 | 0.7510 | 0.0138 |
| B5 近窗摘要候选 | 0.8382 | 0.6758 | 0.7482 | 0.0103 |
| B4 onset 辅助监督候选 | 0.8331 | 0.6735 | 0.7448 | 0.0103 |

B5 onset 候选尚在训练，不填入未完成均值。当前 B4 onset 的独立分支只用于训练，
其诊断 AP 不能替代正式事件头的 report 指标。

已完成控制组、图上下文、upcoming 正例权重 4->12、三分类事件头四组对照，
每组两模型各 seed42/43。没有稳定的 upcoming 提升，三分类不提升为正式最优方案。
本次开训前普通路径中的 best.pt/last.pt 是近窗候选；下一轮会先存入
`model_before_onsetaux20260912.zip`。必须核验 config 与归档来源，不能按路径猜候选。

新诊断最重要的证据：

| 最后一轮权重：upcoming 类概率 AP | B4 s42 | B4 s43 | B5 s42 | B5 s43 |
|---|---:|---:|---:|---:|
| Train | 0.226719 | 0.275094 | 0.446460 | 0.467045 |
| Validation | 0.035218 | 0.024133 | 0.020829 | 0.011249 |

这是 upcoming 对无事件负例的诊断 AP，不能与全事件 AP 混用。
较早 best.pt 在训练集也弱，但 last.pt 的训练集排序明显改善、验证集没有跟上。
因此不能断言“模型完全学不会”，也不能认为“再多训练几轮就好”。
不得把 last.pt 的训练单项分数拼入 best.pt 的正式结果。

完整 P/R/F1、who、hot、时间误差、各 seed 和负结果见长报告第 24--27 节及
`baseline_dense_v6_20260911.json`；旧 B2/B3 结果未完成当前 v6 公平重跑。

## 6. 主实验差异与下一步目标

2026-09-12 `git fetch origin dev_tyx` 成功，仍为 `20c40e230aedee6aef2429d352413fbcf0fa571a`。
主模型文档记录 `dense_i1_a1_prefix8`，204 episode，split 140/28/36。
其 validation P/R/F1 约 0.883/0.881/0.882、upcoming recall 约 0.633；test 约
0.852/0.856/0.854、upcoming recall 约 0.455。它是续训阶段单个 best checkpoint，
不是多 seed 均值；对应正式完整包/权重尚未找到并完成同口径重评。

新增确认的输入差异：主分支 `PDFormer/factory_bn/remain.py` 定义
`PRECURSOR_FAR_WINDOWS=30`；`dataset.py` 构造编码窗口之前最多 30 分钟摘要；
`model.py::_fuse_precursor` 实际进入事件头。当前 baseline 没有这些远历史摘要。
这是额外过去信息，不是未来泄漏；但“双方基础输入完全一致”尚不成立。
尚不能证明上述文档 checkpoint 使用了非零摘要权重，更不能把全部分差归因于此。

新对话继续时建议按顺序推进：

1. 核对主模型实际数据包、split、所用 checkpoint 与输入配置；204/208 差异不能忽略。
2. 显式约定共同历史信息范围：纯 30 分钟，或统一提供额外历史摘要。
   若扩展输入，应登记新特征契约，不能改数据后仍冒称原 v6 完全等价。
3. 在共同输入下做单因素对照，保留 GCN/GAT-GRU 骨干，不机械复制主模型专用解码器。
4. 同时报 train/validation 的 upcoming AP、recall、误报与 MAE，优先处理泛化差距，
   不再重复已无收益的权重/续训盲搜。
5. 获得稳定方案后再冻结多 seed 验证、共同数据和正式 test 协议，补齐 B2/B3。

还需共同讨论历史 hot 用完整 episode 平滑可能导致的前视信息问题；当前只记录，
未擅自改变共同标签。输入对齐不等于所有数据质量问题已经解决。

## 7. 结果在哪里

以下均相对第 3 节 benchmark，不创建新目录：

```text
models/tuning/b4_representation_v1/candidate_history/seed42/
models/tuning/b4_representation_v1/candidate_history/seed43/
models/tuning/b5_representation_v1/candidate_history/seed42/
models/tuning/b5_representation_v1/candidate_history/seed43/
```

每个目录中：

| 文件 | 内容 |
|---|---|
| `model_before_upcontext20260911.zip` | 原 history 控制组 |
| `model_before_upweight20260912.zip` | 图上下文候选 |
| `model_before_upclass20260912.zip` | upcoming 加权候选 |
| `model_before_nearprec20260912.zip` | 三分类候选 |
| `best.pt` / `last.pt` | 最新候选；当前开训前为近窗，后续以 config/运行记录为准 |

benchmark 根目录有四份完整 metrics 汇总：
`baseline_dense_control_metrics_20260911.json`、`baseline_dense_context_metrics_20260912.json`、
`baseline_dense_weighted_metrics_20260912.json`、`baseline_dense_three_class_metrics_20260912.json`，
另有 `baseline_dense_near_metrics_20260912.json`（四次近窗完整结果及文件哈希）。
`baseline_dense_onset_aux_b4_20260912.json` 保存已完成的 B4 两次完整结果、配置、文件
哈希及八份诊断摘要；整批 onset 四次尚未完成，不能把该 B4 导出当作四次汇总。
诊断文件模式 `b[45]_seed4[23]_{train|validation}_diagnostics_<tag>.json`，标签有
`upcoming20260911`、`upcontext20260912`、`upweight20260912`、`upclass20260912`、
`upclasslast20260912`。精确完整指标保存在服务器，Git 中为审计与摘要记录。

## 8. 通过 UU 操控远端的实际步骤

### 8.1 连接链路

```text
Codex 所在本机 -> UU远程 -> Leooo的MacBook Air 的终端 -> ssh my_sci -> sci@sci 服务器
```

Leo 端连港大 VPN。本机不能代替 Leo 直连服务器；不要反复用本机 SSH 证明不可达。
历史上试过端口映射，但当前已验证可用的是 UU 终端内 SSH，不依赖本机 2222。
SSH 认证由用户提供或已有会话完成；不在本文件保存密码。

### 8.2 新对话初始化 CUA

使用 `mcp__cua_repl.js`，首次或 reset 后只执行入口调用，先读返回的 API 文档：

```javascript
var uu = await cua.getApp('com.netease.uuremote');
```

再读实际终端画面：

```javascript
await uu.getAXStateAndScreenshot();
```

必须确认窗口为 `终端 - Leooo的MacBook Air`、AX 中有 `ID: TerminalWindow`。
SSH 登录后画面应为 `sci@sci`；若是 `leooo@...`，先在该终端执行 `ssh my_sci`。
如果显示的是 UU 首页/远控桌面，用最新 AX 定位“终端”入口；不要在错误窗口粘贴命令。

可定义短命令助手：

```javascript
async function terminalCommand(command) {
  const state = await uu.getAXState({emit:false, disableDiffing:true});
  if (!state.includes('ID: TerminalWindow')) {
    throw new Error('UU terminal not selected; nothing sent');
  }
  await uu.paste(command, {format:'text'});
  await uu.pressKey('Return');
  await uu.getAXStateAndScreenshot();
}
```

终端输出主要通过截图读取。每次只发短命令；长 heredoc 曾发生粘贴不完整。
截图尚未显示执行结果时，先重新读画面，不要重复启动训练。
确认只是未提交完的只读命令时可用 `pressKey('ctrl+c')` 取消；不要对训练终端乱发中断。
API 键名使用小写 xdotool 风格，`Ctrl+C`/`Control+c` 不适用。
Git 查看用 `git --no-pager ...`；若进入 `(END)`，发 `q`，UU 终端有时还需 Return。
遇 `native pipe closed`，重新获取 app/状态，先确认命令是否执行；不要据此认定远端任务停止。

### 8.3 登录后的只读前置检查

在服务器终端逐条执行：

```bash
builtin cd "$HOME/work/BSTAN_isaac_factory"
git status --short --branch
git --no-pager log -1 --oneline
tmux display-message -p -t baseline_dense_v6 'dead=#{pane_dead} exit=#{pane_dead_status}'
tmux display-message -p -t baseline_dense_diag 'dead=#{pane_dead} exit=#{pane_dead_status}'
pgrep -af '[t]rain_dense_baseline_control.py|[d]iagnose_baseline_events.py|[t]rain_b[2345]_'
nvidia-smi
```

本次收尾两个会话均正常退出。新对话仍需重新检查，不能依赖旧 PID 或锁文件。
其他人的仿真可能占 GPU，不得终止。没有匹配进程时 `pgrep` 返回码 1 是正常情况。

### 8.4 Python 环境

服务器曾遇 Isaac Sim `PYTHONHOME/PYTHONPATH` 引起 SRE mismatch，以及清掉路径后
torch 不可导入。已经成功使用的环境恢复方式如下，仍需实际核验：

```bash
unset PYTHONHOME
export PYTHON=/home/sci/repos/miniconda3/envs/env_isaaclab/bin/python
export PYTHONPATH="$(tmux show-environment -g PYTHONPATH | cut -d= -f2-)"
export PYTHONDONTWRITEBYTECODE=1
export DENSE_DIR="$PWD/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
export TOOLS="$PWD/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
"$PYTHON" -c 'import sys,torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())'
```

曾验证 torch 2.7.0+cu128。若 tmux 没有保存 PYTHONPATH，先检查实际环境，不凭印象重装。
仅需 JSON/文件读取时用 `/usr/bin/python3 -I`，隔离污染路径。

### 8.5 修改、部署与实验

- 本地先确认 `dev_xwt`、读取 dirty diff，用 `apply_patch` 改动；保留用户修改。
- 测试后提交并 push `origin dev_xwt`。服务器确认无运行任务、tracked clean 后
  `git pull --ff-only origin dev_xwt`，核验 commit。不要在训练期间更新训练模块。
- 当前入口为 `batch_factory_baseline_dense.sh`；训练实现为
  `tools/train_dense_baseline_control.py`，诊断为 `tools/diagnose_baseline_events.py`。
- 用户已授权 208-episode B4/B5 受控优化。先查进程和已完成标签，再设置唯一
  `ARCHIVE_TAG`、`TRAIN_VARIANT`、`TRAIN_SEEDS`、`DEVICE` 并复用已有 tmux/模型目录。
- `ARCHIVE_TAG` 唯一且旧权重先归档；不创建新源码副本。不得重用已存在输出标签。
- 诊断 ZIP 可用 `--archive_member best.pt` 直接验证读取，不解压覆盖当前权重。
- 只对确认已退出的 pane 使用 respawn；正在运行时等待。SSH 断开不等于 tmux 任务停止。
- 避免终端一次打印完整大型 JSON；提取关键字段，完整结果文件留在服务器。

本地测试解释器为 `/tmp/factory-baseline-tests/bin/python`（临时路径，新对话要查是否仍存在）。
较早收尾验证 152 tests / 15 subtests；新 onset 实现的验证记录在第 30 节，不能混用计数。

## 9. 给新对话的起始提示

> 请先读 `doc/experiments/Baseline交接与远端执行指南.md` 和
> `doc/experiments/baseline_dense_v6_20260911.json`，按需查长报告第 27 节。
> 继续优化 B4/B5 upcoming，同时保持与 BNPDFormer 的公平对照。
> 先核验服务器现状和共同历史输入范围，不立即重复已完成搜索。
> 只在 dev_xwt、服务器 BSTAN_isaac_factory 内修改；通过 UU 的 Leo 终端 SSH 操作，
> 不新建目录、不删旧目录、不动主实验仓库、不用 test 调参。
> 上次全部训练和诊断已经正常结束，最新发现是训练/验证泛化差距及远历史摘要输入不一致。
