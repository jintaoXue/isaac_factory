# T0 热启动：递进实验命名与方案（E0–E6）

> **现行实验协议**（原 `t0_finetuning_research_plan.md`）。旧版 T0–T4 / +RHC 看板见 `docs/experiment_protocol_old.md`。  
> 入口：`./run_2026_journal_experiments.sh <实验名> [cuda:0] [--dry-run]`。  
> **已接入口：** E0 / E1 / E1.5 / E2 / E2.5 / E3 / E3-no-oru / E3.5 / E4 / E4-no-oru / TEACHER。  
> **仅命名 / stub：** E5 / E6 / `E6-no-*` / `E6-plus-*` / `E2-random-data`（调用会提示未实现）。

**目标：固定 N=10，从同一个 T0 checkpoint 出发，在有限新增预算下改善 makespan 与成功率。**

**论文主线：教师经验复用 → 教师引导探索 → 层级信用分配 → 自回归决策改进 → 完整组合。** 重点验证各机制如何改善层级协作；组合本身不等于已证明的新颖性。

**机制勿混用：**

| 机制 | 实验落点 | 作用面 |
|---|---|---|
| **ORU（教师数据）** | E2 起默认开；消融名 `*-no-oru` | **学习**：offline 混合回放 |
| **教师探索（guide）** | E3 起默认开；消融名 `*-no-guide` | **行为**：ε 分支教师贪心 vs 随机 |
| **信用缩放 / B-score** | E4 全开；`.5` 只开缩放 | **学习**：奖励尺度 / B 探索率 |

二者都用教师，但一个改 **batch**，一个改 **动作**；用 `E3` vs `E3-no-oru` vs `E1` 拆开看。

## 教师模型选择（2026-09-07）

按训练曲线低点选择教师候选，已实时读取 [T0 run zynalxhz](https://wandb.ai/rl-driving/HcFactory_TPA/runs/zynalxhz) 的 135 条 episode 记录。

| 项目 | 选择 |
|---|---|
| 训练 run | `hier_hard_K10_N10_T40000`（`zynalxhz`） |
| 指标最低点 | `MetricFullorderCore/05_makespan = 14600`，episode 70 |
| 对应训练步数 | `Train/step = 1290374`（不是 W&B 内部 `_step=12972`） |
| 教师候选 checkpoint | **step 1290000**：距低点最近的保存点，用户已确认文件存在 |
| 源机器模型目录 | `/home/sci/work/isaac_factory_tpa/logs/rl_games/HcFactory/hier_2026-08-27_23-17-41/nn/` |
| 当前状态 | 用户已提供远端文件列表：encoder、A、B、C、D_human、D_robot 共 6 个文件齐全且非空；尚未验证反序列化、结构兼容性和实际加载 |

选定此组作为教师：E0 评测、E1–E6 热启动及冻结教师统一使用 step 1290000。14600 是训练 episode 的结果，并非该 checkpoint 的独立评测成绩，正式使用前做 E0 评测与完整性校验。

## Seed 约定（强制）

| 用途 | Seed | 说明 |
|------|------|------|
| **训练**（T0 已发生；E1–E6 主跑） | **42** | wandb 名 `…-S42`；确认实验若再开训练种子，避开 43–52 |
| **评测**（E0 及所有 E* 横比） | **43–52**（10 个，每 seed **1** 局，共 10） | 与训练主 seed 42 错开；`HC_TEST_TIMES` 默认 1 |
| **教师采库** | 可用 **42**（分布对齐）或 42+其它 | **不要**用 43–52 整段扫满当库；入口：`./run_2026_journal_experiments.sh TEACHER`（默认 50 ep，ε=0，按 `offline_replay/episodes/ep_XXX` 落盘） |

## 训练预算（强制对齐）

| 用途 | 默认 | 说明 |
|------|------|------|
| **微调**（E1–E6 及消融） | **`HC_MAX_TRAIN_EPISODES=60`** | 略低于原 T0 低点（~70 ep）；横比必须同预算 |
| **Hard train**（T0/T1 / curriculum） | **`HC_MAX_HARD_EPISODES=100`** | 覆盖原低点（~ep70 @ step≈129万）并留余量；原 run 共约 135 ep |
| Explore 采库 | `HC_EXPLORE_EPISODES=20` | 独立 |
| 教师采库 | `HC_TEACHER_EPISODES=50` | 独立；E2+ORU 用子集 |

**step 1290000 ≈ 第 70 个 episode**：教师低点 `makespan=14600` 记在 ep70，`Train/step≈1290374`；均值约 `1290374/70 ≈ 1.84e4` step/ep。设 hard=100 是「过低点后再多训一段」的上限，不是要重跑到 129 万步。

设为 `0` 可关闭上限（不推荐用于正式横比）。

## E0 运行入口

`bash run_2026_journal_experiments.sh E0 cuda:0`（先加 `--dry-run` 可预览）。权重固定  
`logs/rl_games/HcFactory/hier_2026-08-27_23-17-41`（step 1290000）。

固定 N=10、K=10、每步最多 10 次 C/D dispatch、实际 T=40000、epsilon=0；**评测 seed 43–52** 各 1 局，共 10 局。后续 E1–E6 评测沿用该协议，不直接混用历史 5 seeds（42–46）结果。内部 `t_max_anchor=64000` 经 N10/16 换算为 40000，不能直接改成 40000。

## E1 运行入口

`bash run_2026_journal_experiments.sh E1 cuda:0`（`--dry-run` 可预览）。从同一教师 ckpt 热启，低 lr（Q `2e-5` / enc `1e-5`）、ε=0.05、seed 42，**最多 `HC_MAX_TRAIN_EPISODES`（默认 60）局**。

## E2 运行入口

先 TEACHER 采库，再训：

```bash
./run_2026_journal_experiments.sh TEACHER cuda:0   # 已完成后可跳过
./run_2026_journal_experiments.sh E2 cuda:0
```

E2 = E1 热启设定 + `--oru` 读教师库  
`env_checkpoints/policy_explore/N10_T40000__E2_teacher_ep50/offline_replay`  
（可用 `HC_EXPLORE_CATALOG_DIR` / `HC_TEACHER_EPISODES` 覆盖）。`oru_mix_start=0.25`（约 25% 教师 + 75% 在线），`oru_warmup_updates=0`（自动 warmup）。wandb：`Hier4TPA-E2-N10-S42`。

## E3 运行入口

```bash
./run_2026_journal_experiments.sh TEACHER cuda:0   # 已完成后可跳过
./run_2026_journal_experiments.sh E3 cuda:0
```

E3 = E2 全套 + `--teacher_explore`：ε 探索分支上，以衰减的教师比例选冻结 T0 贪心动作，其余为 mask 合法随机；利用分支仍用学生 ε=0。默认 `teacher_explore_ratio` 1→0 / 300k env steps。wandb：`Hier4TPA-E3-N10-S42`。预算同 E1/E2（`HC_MAX_TRAIN_EPISODES`，默认 60）。**协议纯净 E3 不含信用缩放**（关 H → A=B=1.0）。

## E3-no-oru（优先消融：去掉教师数据）

```bash
./run_2026_journal_experiments.sh E3-no-oru cuda:0
```

**E3-no-oru = E1 + 教师探索，不开 ORU**（不需要教师 offline 库）。用于回答：教师探索收益是否依赖 ORU。  
wandb：`Hier4TPA-E3-no-oru-N10-S42`。对照：`E3`（有 ORU）/ `E1`（两者都无）。

同族接口（已接）：**`E4-no-oru`** = E4 去掉 ORU（保留教师探索 + 层级学习）。  
规划 stub：`E6-no-oru`（完整方法上去掉 ORU）。

## E1.5 / E2.5 / E3.5（信用缩放消融支线）

```bash
./run_2026_journal_experiments.sh E1.5 cuda:0
./run_2026_journal_experiments.sh E2.5 cuda:0
./run_2026_journal_experiments.sh E3.5 cuda:0
```

| 编号 | 设置 | 说明 |
|---|---|---|
| **E1.5** | E1 + A×2.0 / B×1.5，**无** `b_score_rl` | 只开信用缩放 |
| **E2.5** | E2 + A×2.0 / B×1.5，**无** `b_score_rl` | E2 + 只开信用缩放 |
| **E3.5** | E3 + A×2.0 / B×1.5，**无** `b_score_rl` | E3 + 只开信用缩放；相对 E4 少 B-score |

历史污染跑已改名归档：  
- `m3nz6opg` → `Hier4TPA-E1.5-N10-S42`  
- `2mq6zow7`（30ep）/ `zvjalw62`（60ep 重跑）→ `Hier4TPA-E2.5-N10-S42`  
- `31rt1h7i` → `Hier4TPA-E3.5-N10-S42`  

（本地有落盘的 E1.5/E3.5 已同步 `params`/`metrics`/`RELABEL.md`；E2 主要在远端 W&B。）

## E4 运行入口

```bash
./run_2026_journal_experiments.sh E4 cuda:0
```

E4 = E3 全套 + `--hierarchical_credit` + `--b_score_rl`：A/B 决策奖励缩放（默认 A×2.0、B×1.5），B 排序探索率减半。关闭 H 时缩放强制为 1.0（不再被 YAML 误开）。wandb：`Hier4TPA-E4-N10-S42`。预算同 E1–E3。

## 开关契约（防静默泄漏）

| 开关 | 默认（`hier.yaml`） | 生效规则 |
|---|---|---|
| `hierarchical_credit` | `False` | **关** → 有效 `credit_scale_A/B` **强制 1.0**（忽略 YAML/CLI 里的 2.0/1.5） |
| `credit_scale_A/B` | `1.0` | **仅当** `hierarchical_credit=true` 时生效；E4 / E*.5 显式设 `2.0/1.5` |
| `b_score_rl` | `False` | 与 H **正交**；`--hierarchical_credit` **不会**自动打开它 |
| `teacher_explore` | `False` | E3 / E3-no-oru / E3.5 / E4 / E4-no-oru 由 journal 打开 |
| `oru` | `False` | E2+ 主线默认开；`*-no-oru` **强制关**，且不要求 catalog |

**启动日志**必含：`hier_credit=… (A=… B=…) b_score_rl=… teacher_explore=… oru=…`。  
对已知 `algo_variant`，`HierarchicalTPA` 会按表 **断言** 有效开关；不一致直接 `RuntimeError`。

| 编号 | H | b_score | A/B | teacher_explore | oru |
|---|---|---|---|---|---|
| E1 | 关 | 关 | 1/1 | 关 | 关 |
| E1.5 | 开 | 关 | 2/1.5 | 关 | 关 |
| E2 | 关 | 关 | 1/1 | 关 | 开 |
| E2.5 | 开 | 关 | 2/1.5 | 关 | 开 |
| E3 | 关 | 关 | 1/1 | 开 | 开 |
| **E3-no-oru** | 关 | 关 | 1/1 | **开** | **关** |
| E3.5 | 开 | 关 | 2/1.5 | 开 | 开 |
| E4 | 开 | 开 | 2/1.5 | 开 | 开 |
| **E4-no-oru** | 开 | 开 | 2/1.5 | 开 | **关** |
| E5（规划） | 关 | 关 | 1/1 | 开 | 开 |
| E6（规划） | 开 | 开 | 2/1.5 | 开 | 开 |

## E5 / E6（规划；代码入口为 stub）

| 编号 | 设置 | wandb 名（规划） |
|---|---|---|
| **E5** | **E3 + 仅自回归**（分层 ε、步内候选采样等）；**不含** E4 的层级信用 / B-score | `Hier4TPA-E5-N10-S42` |
| **E6** | **E5 + E4** = E3 + 自回归 + 层级学习（完整方法） | `Hier4TPA-E6-N10-S42` |

旧方案曾把「E4+自回归」叫作 E5；现改为 **E5=只 AR，E6=AR+H**，便于单独量自回归收益（E5 vs E3）与完整组合（E6 vs E5 / E4）。

## 1. 命名规则与主实验表

**主线用 E0–E6，编号大体每增加 1 增加一项技术**（`.5` / `-no-*` / `-plus-*` 为支线，不占主线递进）。E 表示 Experiment，不是旧协议的教师探索 `+E`；展示时写“E3 教师探索版”，不再拼接技术字母。

E0 只评测；E1–E6 分别从同一个 T0 checkpoint 初始化，教师固定为 T0。递增的是技术配置，不是接着上一组模型继续训练，也不表示性能必然递增。

| 新编号 | 直观名称 | 相对对照 | 完整设置 | 旧协议对照 |
|---|---|---|---|---|
| **E0** | 原始版 | — | 已有 T0，仅评测 | T0 |
| **E1** | 微调版 | 权重热启动后继续训练 | T0 权重＋低学习率在线微调 | T0+W |
| **E1.5** | 微调+信用 | 仅 A/B 信用缩放 | E1＋A×2.0/B×1.5（无 b_score） | 消融支线 |
| **E2** | 数据复用版 | 教师数据＋ORU | E1＋冻结 T0 采库、混合回放 | T2+W |
| **E2.5** | 复用+信用 | 仅 A/B 信用缩放 | E2＋A×2.0/B×1.5（无 b_score） | 消融支线 |
| **E3** | 教师探索版 | 教师引导在线探索 | E2＋探索分支中教师/随机动作混合，教师比例衰减 | T2+E+W |
| **E3-no-oru** | 探索无数据 | 去掉 ORU | E1＋教师探索（无 offline 混合） | 消融：guide vs data |
| **E3.5** | 探索+信用 | 仅 A/B 信用缩放 | E3＋A×2.0/B×1.5（无 b_score） | 消融支线 |
| **E4** | 层级学习版 | B-score RL＋A/B 信用缩放 | E3＋完整层级学习机制 | T2+H+E+W |
| **E4-no-oru** | 层级无数据 | 去掉 ORU | E4 去掉教师 offline 混合 | 消融 |
| **E5** | 自回归版 | 分层 epsilon＋步内候选采样 | **E3＋自回归**（**无** E4 的 H） | T2+A+E+W |
| **E6** | 完整方法 | E5＋E4 | E3＋自回归＋层级学习 | T2+H+A+E+W |

**先跑 E0–E4（及需要的 `E3-no-oru`），再实现 E5，最后 E6。** E1 是公平微调基线；**E6 是候选完整方法**。相邻/消融比较：E3-no-oru vs E3 看 ORU，E3-no-oru vs E1 看教师探索，E4 vs E3 看层级学习，E5 vs E3 看自回归。

## 2. 消融和扩展命名

**消融用 `编号-no-模块`，扩展用 `编号-plus-模块`，替换用 `编号-random-data`。** 分支不占主线编号。完整方法以 **E6** 为锚；机制拆开可在 E3/E4 上先做（不必等 E6）。

| 名称 | 状态 | 设置 | 对照 / 目的 |
|---|---|---|---|
| **`E3-no-oru`** | **已接** | E3 去掉 ORU（保留教师探索） | vs E3：ORU 贡献；vs E1：教师探索贡献 |
| **`E4-no-oru`** | **已接** | E4 去掉 ORU | vs E4：完整层级设定下 ORU 贡献 |
| `E2-random-data` | stub | E2 的教师库换为随机库 | vs E2：教师数据质量 |
| `E6-no-oru` | stub | E6 去掉 ORU | vs E6：完整方法下 ORU |
| `E6-no-guide` | stub | E6 去掉教师探索 | vs E6：教师探索贡献 |
| `E6-no-hier` | stub | E6 去掉层级学习，等同 E5 | 复用 E5，不重复训练 |
| `E6-no-ar` | stub | E6 去掉自回归，等同 E4 | 复用 E4，不重复训练 |
| `E6-plus-replay` | stub | E6＋旧 R：PER、Dueling | 后端增强 |
| `E6-plus-curriculum` | stub | E6＋旧 Cr | 课程 |
| `E6-plus-staged` | stub | E6＋分阶段解冻 | 分阶段微调 |

**命名约定（统一用短模块名）：**

| 模块短名 | 含义 | 对应开关 |
|---|---|---|
| `oru` | 教师 offline / ORU 混合回放 | `oru=false`，不加载 catalog |
| `guide` | 教师引导探索 | `teacher_explore=false` |
| `hier` | 层级学习（信用缩放 + B-score） | `hierarchical_credit` + `b_score_rl` |
| `ar` | 自回归决策改进 | （E5/E6，未实现） |

完整方法有效后，优先补 `no-oru` / `no-guide` / `no-hier`；三个 `plus` 最多先选一个。不做全组合搜索，不做去热启动的从零训练消融。

**W&B / 输出目录统一格式：** `Hier4TPA-{实验名}-N{产品数}-S{训练种子}`，例如 `Hier4TPA-E3-N10-S42`、`Hier4TPA-E3-no-oru-N10-S42`。评测加 `-eval`；重复运行可加 `-r2`。checkpoint 来源、教师版本和详细超参数放配置中。旧实验保留原名，新旧对应以表为准。

## 3. 最小实施约定

- **热启动（旧 W）**：先接通训练加载，严格校验 encoder 和各 Q head；更新前与教师输出一致。当前 checkpoint 是权重热启动，不是完整续训。
- **教师数据（旧 T2 / ORU）**：教师仅在训练订单采库。建议先取 25% 教师＋75% 在线样本，不做大量离线 warmup；当前 `oru_warmup_updates=0` 是自动设置，不是关闭。消融入口名统一 **`*-no-oru`**（不要再用含糊的 no-data）。
- **教师探索（旧 +E）**：只在 epsilon 探索分支选择教师/随机动作，教师比例 1→0 / 300k env steps；动作必须满足当前 mask。消融名统一 **`*-no-guide`**。训练期即使教师退出探索分支，仍保留 ε 随机（期刊默认 ε=0.05）；评测 ε=0 才是纯学生。
- **层级学习（旧 H）**：`hierarchical_credit` 与 `b_score_rl` **正交**（E*.5 只开缩放；E4 两者都开）。`hierarchical_credit=false` 时 A/B 缩放**强制为 1.0**。启动时对已知 `algo_variant` 做开关断言。
- **历史污染（已改名）**：2026-09-11 修复前，YAML 写死 `credit_scale_A/B=2.0/1.5` 且关 H 仍生效。原 E1/E2/E3 污染跑已改名为 **E1.5 / E2.5 / E3.5**。修复后脚本重跑的 E1–E3 才是纯净版（关 H → A=B=1.0）。
- **自回归增强（旧 A）**：先实现分层 epsilon 和少量合法候选采样。当前 **E5 / E6 入口为 stub**。

## 4. 统一预算与论文指标

- **起始参数**：Q 学习率 `2e-5`、encoder `1e-5`、基础 epsilon `0.05`，均为建议起点；教师探索、层级学习、自回归增强的差异按表显式记录。
- **筛选预算**：微调默认 **`HC_MAX_TRAIN_EPISODES=60`**；从零 hard 默认 **`HC_MAX_HARD_EPISODES=100`**。同时记录梯度更新数、候选推理开销、采库与墙钟时间。环境、硬件和其他超参数保持一致。
- **主结果**：成功率、成功订单 makespan、含失败惩罚的整体指标；效率报告达到预定性能目标的新增步数与实际时间，未达到则明确标注。
- **机制证据**：教师探索看早期退化/失败，层级学习看各层样本与 TD 误差，自回归增强看候选改选率及最终调度收益；ORU 消融看 `E3` vs `E3-no-oru`；不只报告总 reward。
- **确认实验**：主跑训练 S42；额外训练种子作方差时再开 S；**评测固定 43–52×1**；配对差值与置信区间；验证集选配置，测试集不参与选择。单个 T0 起点结论限于该起点，预训练和采库成本单列。

本文为现行 E0–E6 方案；旧协议看板已归档为 `docs/experiment_protocol_old.md`。  
E0 加载旧 T0 权重时，loader 会自动把 pre-Rainbow 的 `net.0/2/4` 映射到当前 `feature`+`net`（无需重训）。
