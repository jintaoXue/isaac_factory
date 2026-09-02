# Hier4TPA 实验协议与运行手册

本文档对应论文 **长时域训练协议 ablation（T0–T4）**、**HRTPA 基线评测** 与 **Hier4TPA 训练/评测** 的统一说明。  
仿真与训练代码位于本仓库根目录；命令均在仓库根目录下执行。

```bash
conda activate isaaclab   # 或你的 Isaac Lab 环境名
```

---

## 1. 术语

| 术语 | 含义 |
|------|------|
| **Catalog** | 离线数据索引库：explore 阶段按 **progress key** 去重后保存的车间 checkpoint（`.pkl`）及 `catalog.jsonl` |
| **Progress key** | 决策等价指纹（在制任务、资源占用、`not_started` 等；**不含**路径几何） |
| **ORU** (offline replay update) | 从 catalog / 离线 replay 采样 transition **计算 loss 并更新 actor–critic**；progress key 用于去冗余。 |
| **MetricCatalog/** | 采库 / policy catalog 专用指标（`HcFactory_Catalog` project） |
| **Curriculum** | 倒序分段（8→10, 6→10, …, 0→10），规定各段的订单进度与 `T_budget`；**不重载**神经网络权重 |
| **Hard train** | 整单 N=10、无 curriculum 分段 |
| **K** | `max_parallel_cd_dispatch`：单步最多并行 C/D 派工数（默认 10） |

> **实现说明**：当前 `batch_train.sh` job **22/27** 仅完成 **采库 + catalog 仿真 reset**（`pick_by_nfin` restore），**尚未**接入完整的 offline replay update；job 27 是 T2 的占位/legacy 路径，**不等于**论文定义的 T2。

Catalog 默认路径（**未设 `HC_CATALOG_TAG` 时的 legacy 路径**）：

```
env_checkpoints/random_explore/N10_T40000/
├── catalog.jsonl
├── by_nfin/
└── rounds/r001/ckpts/*.pkl
```

**推荐命名**（设 `HC_CATALOG_TAG` 后）：

```
env_checkpoints/{source}/N{n}_T{t}__{tag}/
```

示例：

| 实验 | `HC_CATALOG_SOURCE` | `HC_CATALOG_TAG` | 完整路径 |
|------|---------------------|------------------|----------|
| T2 随机采库 | `random_explore` | `T2_random_ep10` | `env_checkpoints/random_explore/N10_T40000__T2_random_ep10/` |
| T4 策略采库 | `policy_explore` | `T4_policy_ep20` | `env_checkpoints/policy_explore/N10_T40000__T4_policy_ep20/` |
| 复现实验 v2 | `random_explore` | `T2_random_ep10_v2` | 同结构，**tag 不同即不冲突** |

> **22 写库与 27 读库必须使用相同的 `HC_CATALOG_TAG` 或 `HC_EXPLORE_CATALOG_DIR`。**

---

## 1b. Catalog 要多大？

### 功能需求（curriculum 各段需 catalog 覆盖的 `start_nfin`）

倒序 curriculum 各阶段需要的 **`start_nfin`**：

| Stage | start_nfin | 需要 catalog？ |
|-------|------------|----------------|
| 0 | 8 | ✅ 至少 1 条 |
| 1 | 6 | ✅ 至少 1 条 |
| 2 | 4 | ✅ 至少 1 条 |
| 3 | 2 | ✅ 至少 1 条 |
| 4 | 0 | ❌ 空车间即可 |

`pick_by_nfin(n)` 在缺少精确 `n` 时会 **fallback 到更小 nfin**，但仍建议 **8/6/4/2 各桶至少 1 条**。

### 推荐规模

| 级别 | explore episodes | 预期 unique keys | 磁盘（粗估） | 适用 |
|------|------------------|------------------|--------------|------|
| **最小可用** | 5–10 | 30–80 | 50–200 MiB | 调试、验证 catalog 覆盖 |
| **论文默认** | **20**（`HC_EXPLORE_EPISODES` 默认） | 50–200 | 100–500 MiB | T1/T2 采库 Phase A |
| **稳妥** | 20–30 | 150–400 | 300 MiB–1 GiB | `start_nfin` 桶缺失时 |
| **过量** | >50 | 边际递减 | >1 GiB | 一般不必 |

说明：

- 每条 catalog 记录对应 **一次成功 dispatch 后的决策等价状态**；`progress key` 相同则 **不重复写入**。
- 10 个 explore episode 通常能覆盖 8/6/4/2 四档；若 `inspect_catalog.py` 报 MISSING，先 **加 episode 或换 seed 重采**，不要盲目加到 100+。
- 单个 `.pkl` 约 **1–3 MiB**（含逻辑状态 + 少量 articulation/prim）；100 条约 **100–300 MiB**。

### 采库后检查命令

```bash
# 默认 legacy 路径
python scripts/inspect_catalog.py

# 命名 catalog
python scripts/inspect_catalog.py env_checkpoints/random_explore/N10_T40000__T2_random_ep10
```

输出应包含 `OK: all curriculum start_nfin buckets (8,6,4,2) present`。

### 命名规范（避免重复）

1. **每次新实验** 设唯一 tag：`T2_random_ep10`、`T2_random_ep20_seed43`、`T4_policy_run1`。
2. **source 区分采库方式**：`random_explore`（T1/T2） vs `policy_explore`（T3/T4，待实现）。
3. **22 与 27 成对使用同一 tag**：

```bash
export HC_CATALOG_TAG=T1_random_ep20
export HC_EXPLORE_EPISODES=20   # 已是默认值，可省略

# Phase A：写库
./batch_train.sh 22 cuda:0

# 检查覆盖
python scripts/inspect_catalog.py "env_checkpoints/random_explore/N10_T40000__${HC_CATALOG_TAG}"

# Phase B：训练（legacy job 27；完整 T2 待 ORU 实现）
./batch_train.sh 27 cuda:0
```

4. **不要用默认 legacy 路径跑多组并行实验**（都会写到 `N10_T40000/`，round 会递增但混在同一 catalog.jsonl 里，难以区分实验）。

5. wandb run 名已带 tag 后缀：`explore_...__T2_random_ep10`、`hier_curriculum_...__T2_random_ep10`。

### Catalog 指标（wandb project: `HcFactory_Catalog`）

job **22** / **33** 写入 `HcFactory_Catalog`。核心看 **规模、加入 vs 未加入、curriculum 覆盖**。

**Train/step（仿真过程中实时）**

| 指标 | 含义 |
|------|------|
| `01_unique_keys` | catalog 中 decision-equivalent unique key 总数 |
| `02_joined_cumulative` | 累计**加入**次数（new + updated） |
| `03_not_joined_cumulative` | 累计**未加入**次数（progress key 重复跳过） |

**MetricCatalog/episode（每个 episode 结束）**

| 指标 | 含义 |
|------|------|
| `01_unique_keys` | 截止本 ep 的 unique key 总数 |
| `02_new_keys` | 本 ep **全新** key 数（不含 updated） |
| `03_joined` | 本 ep 加入次数（new + updated） |
| `04_not_joined` | 本 ep 未加入次数（与 `03_joined` 对比） |
| `05_new_keys_since_run` | 自本次 run 起新增的 unique key 数 |
| `06_nfin_buckets_covered` | curriculum 所需 nfin 桶覆盖数（8/6/4/2，满分 4） |
| `07_joined_cumulative` / `08_not_joined_cumulative` | run 累计加入 / 未加入 |
| `09_join_fraction` | 本 ep 加入占比 = joined / (joined + not_joined) |
| `10_join_fraction_cumulative` | run 累计加入占比 |

wandb 建议同图对比：`03_joined` vs `04_not_joined`（按 episode），或 `02_joined_cumulative` vs `03_not_joined_cumulative`（按 step）。

job **33** 另含常规训练指标：`MetricTrain/*`、`MetricLoss/*`、`MetricCore/*`。

---

## 2. 长时域训练 Ablation（T0–T4）

> 论文目标：**explore 建 catalog → offline replay update →（可选）curriculum 分段**。  
> 当前代码：**仅 Phase A 采库（job 22）+ legacy 仿真 reset（job 27）**；**ORU 与完整 T1/T2 均未实现**。

| ID | 名称 | 流程概要 | 代码状态 |
|----|------|----------|----------|
| **T0** | Hard train | 空车间 → 整单 N=10 在线 RL | ✅ job **28** |
| **T1** | Explore catalog + offline replay update | 随机采库 → 离线数据驱动 loss 更新 + hard train | 🔲 未实现 |
| **T2** | Explore catalog + offline replay update + curriculum | 随机采库 → ORU + 倒序 curriculum | 🔲 **未实现**（论文 proposed） |
| **T3** | Policy-guided catalog + offline replay update | 策略引导采库 → ORU + hard train | 🔲 未实现 |
| **T4** | Policy-guided catalog + offline replay update + curriculum | 策略引导采库 → ORU + curriculum | 🔲 未实现 |

**Legacy（非 T2，勿与论文 proposed 混淆）**：`22 → 27` 仅用 catalog 做 **segment 仿真 reset**，在线 interaction 产生 replay，**没有**从离线库直接做 loss update。

预期性能（实现 ORU 后写 ablation）：**T0 < T1 < T2 ≤ T4**。

---

### 2.1 T0 — Hard train

- **语义**：无 catalog；每个 episode 从空车间开始；整单 10 件；`T_max = 40000`（默认 anchor 下）。
- **batch 序号**：**28**
- **wandb 名**：`hier_hard_K10_T40000`

```bash
./batch_train.sh 28 cuda:0
```

等价 `train.py` 核心参数：

```bash
python train.py --task HRTPaHC-v1 --algo hier --headless \
  --wandb_activate --wandb_project HcFactory_TPA \
  --wandb_name "T0_hard_train" \
  --max_parallel_cd_dispatch 10 \
  +t_max_anchor=64000 \
  --device cuda:0
```

---

### 2.2 T1 — Explore catalog + ORU（一键命令）

**Phase A（已实现）**：job **22**，20 episode，wandb → `HcFactory_Catalog`，记录 `MetricCatalog/*`。

**Phase B（ORU 未实现）**：当前 job **28** 为 online hard train（`HcFactory_TPA`）。

```bash
# 一键：采库 → hard train（连续）
export HC_CATALOG_TAG=T1_random_ep20
./batch_train.sh T1 cuda:0

# 仅 Phase A 采库
export HC_CATALOG_TAG=T1_random_ep20
./batch_train.sh 22 cuda:0

# Policy-guided catalog（80 ep，同一 catalog wandb project）
export HC_CATALOG_TAG=T3_policy_ep80
export HC_CATALOG_SOURCE=policy_explore
./batch_train.sh 33 cuda:0
```

---

### 2.3 T2 — Explore catalog + offline replay update + curriculum（论文 Proposed）

- **语义**：
  1. Phase A：job **22** 随机 explore 建库。
  2. Phase B：倒序 curriculum 各段内，用 catalog 离线数据做 **loss update**；curriculum 规定段内 `start_nfin` / `T_budget`；catalog 亦可提供段起点状态，但**核心是 ORU 而非 reset**。
- **代码状态**：🔲 **未实现**
- **Legacy 占位**：job **27**（`22 → 27`）目前只有 catalog **仿真 reset** + 在线 replay，**不能当作 T2 实验结果**。

```bash
# Legacy 占位（开发调试用，论文勿标为 T2）
export HC_CATALOG_TAG=T2_random_ep10
./batch_train.sh 22 cuda:0
./batch_train.sh 27 cuda:0
```

**目标实现 checklist（T2）**：

- [ ] explore 存 transition + checkpoint（progress key 去重）
- [ ] 训练阶段 offline replay update（replay 混合或预训练若干 step）
- [ ] curriculum 段调度与 catalog `start_nfin` 对齐
- [ ] 去掉对 stagnation restore 的论文依赖（可选保留代码）

**Curriculum 阶段（`curriculum.py`，实现 T2 时沿用）**：

| Stage | start_nfin | ΔN | 段内 T_budget |
|-------|------------|-----|---------------|
| 0 | 8 | 2 | 8000 |
| 1 | 6 | 4 | 16000 |
| 2 | 4 | 6 | 24000 |
| 3 | 2 | 8 | 32000 |
| 4 | 0 | 10 | 40000 |

段内目标：已完成 `start_nfin` 件 → 训练目标 `target_nfin = 10`。  
晋级条件（默认）：近 20 ep 成功率 ≥ 0.7 且停滞率 < 0.2。

---

### 2.4 T3 — Policy-guided catalog + offline replay update

- **语义**：Phase A 用 **在线 RL 策略**（非纯 random）交互并采库；Phase B 同 T1（ORU + hard train）。
- **代码状态**：🔲 **未实现**
- **实现要点**：
  - 训练期 `learn=True` 且写入 catalog / offline replay（`HC_CATALOG_SOURCE=policy_explore`）；
  - Phase B：offline replay update，无 curriculum。

---

### 2.5 T4 — Policy-guided catalog + offline replay update + curriculum

- **语义**：Phase A policy-guided 采库；Phase B 同 T2（ORU + curriculum）。
- **代码状态**：🔲 **未实现**

---

## 3. 默认仿真与训练 Setting

以下默认值来自 `batch_train.sh` / `hier.yaml`，可用环境变量覆盖。

| 参数 | 默认值 | 环境变量 / 说明 |
|------|--------|-----------------|
| 任务 | `HRTPaHC-v1` | — |
| 算法 | `hier` | `--algo hier` |
| 训练订单 N | 10 | curriculum / hard train |
| 评测全订单 N | 16 | job 29–32；`HC_TRAIN_N_PRODUCTS` |
| `T_max` anchor（N=16） | 64000 | `HC_T_MAX_ANCHOR` |
| `T_max`（N=10） | 40000 | `per_T_max × 10`，`per_T_max = anchor/16` |
| WIP cap | 10 | `parallel_producing_limit` |
| K（并行 C/D） | 10 | `HC_MULTI_K` |
| 并行 env 数 | 1 | `HC_NUM_ENVS` |
| Explore episodes | 20 | job 22：`HC_EXPLORE_EPISODES` |
| DQN replay | 50000（A: 5000） | `hier.yaml` |
| ε decay | 1.0 → 0.05 / 1.5M steps | — |

---

## 4. 评测协议（Eval）

### 4.1 统一约定

- **Seeds**：`42,43,44,45,46`（`HC_TEST_SEEDS`）
- **每 seed 重复**：2 次（`HC_TEST_TIMES`）→ 共 **10 episodes / run**
- **指标**：makespan、success、truncation、completion rate（见 `logs/.../eval/`）
- **评测输出**：`<exp_dir>/eval/episodes.jsonl`、`eval_summary.json`

### 4.2 Hier4TPA 评测（加载训练 checkpoint）

- **batch 序号**：**29**
- **必须设置**：`HC_LOAD_DIR` 指向含 `nn/` 的训练目录

```bash
export HC_LOAD_DIR=logs/rl_games/HcFactory/hier_2026-08-27_23-18-44

# N=16 全订单（默认 HC_TRAIN_N_PRODUCTS=16）
./batch_train.sh 29 cuda:0

# 仅 N=10
HC_TRAIN_N_PRODUCTS=10 ./batch_train.sh 29 cuda:0

# 指定 checkpoint step
HC_LOAD_STEP=2415000 HC_LOAD_DIR=... ./batch_train.sh 29 cuda:0
```

**论文 wrapper**（`run_2026_journal_experiments.sh`）：

```bash
# 训练 T2
./run_2026_journal_experiments.sh train cuda:0

# 评测 curriculum 模型（N16 + N10）
HC_LOAD_DIR=logs/rl_games/HcFactory/hier_YYYY-MM-DD_HH-MM-SS \
  ./run_2026_journal_experiments.sh hier-eval-curr cuda:0

# 评测 hard train 模型
HC_LOAD_DIR=... ./run_2026_journal_experiments.sh hier-eval-hard cuda:0
```

### 4.3 HRTPA 基线矩阵

| 基线 | K | N | batch job | wandb project |
|------|---|---|-----------|---------------|
| Rule 单产品 | 1 | 10 | **24** | `HcFactory_TPA_Eval` |
| Rule 多产品 | 10 | 10 | **25** | 同上 |
| Hier random（ε=1） | 10 | 10 | **26** | 同上 |
| Rule 单产品 | 1 | 16 | **30** | 同上 |
| Rule 多产品 | 10 | 16 | **31** | 同上 |
| Hier random（ε=1） | 10 | 16 | **32** | 同上 |

```bash
# 一次跑齐 N=10 + N=16 六个基线
./batch_train.sh E cuda:0

# 或分项
./run_2026_journal_experiments.sh rule-n10 cuda:0
./run_2026_journal_experiments.sh rule-n16 cuda:0
./run_2026_journal_experiments.sh random-n10 cuda:0
./run_2026_journal_experiments.sh random-n16 cuda:0
```

**Two-level TPA 对照（论文）**：Hier 评测时 **`K=1`**（单产品决策焦点），与 Hier4TPA **`K=10`** 对比；可通过单独 eval 命令指定：

```bash
python train.py --task HRTPaHC-v1 --algo hier --headless --test \
  --load_dir "${HC_LOAD_DIR}" \
  --max_parallel_cd_dispatch 1 \
  --train_n_products 10 \
  --test_seeds 42,43,44,45,46 --test_times 2 \
  +t_max_anchor=64000 --device cuda:0
```

---

## 5. 推荐实验矩阵（按优先级）

### 5.1 长时域训练 Ablation（主表）

| 实验 | 命令 | 备注 |
|------|------|------|
| T0 | `./batch_train.sh 28 cuda:0` | ✅ 下界 |
| T1 | 22 → ORU hard train | 🔲 未实现 |
| T2 | 22 → ORU + curriculum | 🔲 **论文 proposed，未实现** |
| T3 / T4 | policy 采库 + ORU（+ curriculum） | 🔲 未实现 |
| Legacy | `22 → 27` | 仅 catalog reset，**勿标 T2** |

每个 T* 训练完成后：

```bash
HC_LOAD_DIR=<对应 logs/.../hier_* 目录> ./batch_train.sh 29 cuda:0
```

### 5.2 基线对比（辅表）

与 T2 同 setting（N=10, T_max=40000, K=10）：

```bash
./batch_train.sh 24 25 26 cuda:0
```

泛化（N=16）：

```bash
./batch_train.sh 30 31 32 cuda:0
```

### 5.3 完整流水线（开发调试）

```bash
# C 组：采库 + 全部 N=10 基线 + curriculum 训练
./batch_train.sh C cuda:0
```

---

## 6. 环境变量速查

| 变量 | 作用 | 示例 |
|------|------|------|
| `HC_T_MAX_ANCHOR` | 全订单时间上限 | `64000` |
| `HC_MULTI_K` | 并行 C/D 数 K | `10` |
| `HC_CATALOG_TAG` | catalog 实验标签（22/27 须一致） | `T2_random_ep10` |
| `HC_CATALOG_SOURCE` | 采库类型目录 | `random_explore` |
| `HC_EXPLORE_CATALOG_DIR` | 显式 catalog 根路径 | 覆盖 TAG |
| `HC_EXPLORE_EPISODES` | job 22 采库 episode 数 | `20` |
| `HC_WARMSTART` | 指定单个 catalog/stagnation pkl | `env_checkpoints/.../*.pkl` |
| `HC_LOAD_DIR` | 评测加载训练目录 | `logs/rl_games/HcFactory/hier_*` |
| `HC_LOAD_STEP` | 评测 checkpoint step | `2415000` |
| `HC_TRAIN_N_PRODUCTS` | 评测订单数 10/16 | `10` |
| `HC_TEST_SEEDS` | 评测 seeds | `42,43,44,45,46` |
| `HC_TEST_TIMES` | 每 seed 重复次数 | `2` |
| `HC_WANDB_MODE` | `online` / `offline` | 网络不稳用 `offline` |
| `HC_WANDB_PROJECT` | 训练 project | `HcFactory_TPA` |
| `HC_WANDB_CATALOG_PROJECT` | 采库 / policy catalog 专用 project | `HcFactory_Catalog` |
| `HC_POLICY_CATALOG_EPISODES` | job 33 episode 数 | `80` |

**wandb 本地配置**（可选，gitignore）：仓库根 `.wandb_local.env`

```bash
HC_WANDB_API_KEY=...
HC_WANDB_ENTITY=your_entity
```

---

## 7. 日志与产物路径

| 类型 | 路径 |
|------|------|
| 训练日志 | `logs/rl_games/HcFactory/hier_<timestamp>/` |
| 网络权重 | `.../nn/` |
| 训练 metrics | `.../metrics.jsonl` |
| 评测明细 | `.../eval/episodes.jsonl` |
| Explore catalog | `env_checkpoints/random_explore/N10_T40000/` |
| Hydra 配置快照 | `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/` |

---

## 8. batch_train.sh 序号一览

| 序号 | 类型 | 说明 |
|------|------|------|
| **22** | 采库 | explore，ε=1，写 catalog |
| **23** | debug | 可视化 + `HC_WARMSTART`（需手动指定 pkl） |
| **24** | eval | rule K=1, N=10 |
| **25** | eval | rule K=10, N=10 |
| **26** | eval | hier random ε=1, N=10 |
| **27** | train | curriculum（T2 Phase B） |
| **28** | train | hard train（T0） |
| **29** | eval | 加载 nn，N=10/16 |
| **30–32** | eval | rule / random, N=16 |
| **C** | 组合 | 22 → 24 → 25 → 26 → 27 |
| **33** | train | policy catalog_collect + hard train（80 ep） |
| **T1** | 组合 | 22 → 28（T1 一键流水线） |
| **E** | 组合 | 24 → 25 → 26 → 30 → 31 → 32 |

---

## 9. 论文章节 ↔ 实验映射

| 论文内容 | 对应实验 |
|----------|----------|
| Hier4TPA vs Rule / Random | jobs 24–26, 29–32 |
| Hier4TPA vs two-level TPA | 同 Hier，`K=1` vs `K=10` |
| 长时域协议 ablation T0–T4 | §2 本表 |
| Proposed training protocol | **T2**（offline replay update + curriculum，**未实现**） |
| Catalog / progress key | job 22 Phase A；`env_checkpoint.progress_key` |
| Legacy job 27 | 仅 segment 仿真 reset，非 ORU |

---

## 10. 待实现清单（边写论文边开发）

- [ ] **ORU 核心**：explore 存 transition；训练从离线 replay 采样做 loss backward（progress key / 相似度去重）
- [ ] **T1**：ORU + hard train（job 28 扩展）
- [ ] **T2**：ORU + curriculum（论文 proposed）
- [ ] **T3/T4**：policy-guided catalog + ORU（± curriculum）
- [ ] （可选）`soft_cosine` 相似 transition 过滤

---

## 11. 常见问题

**Q: 跑 27 提示 catalog miss / falling back to empty start？**  
A: 先跑 **22** 建库，或设置 `explore_catalog_dir` 指向已有 `N10_T40000` 目录。

**Q: T0 和 T2 的差别？**  
A: T0 纯 online hard train；T2 在 explore catalog 基础上做 **offline replay update** + curriculum。当前 job 27 **不是** T2。

**Q: `--warmstart` / `pick_by_nfin` 算 ORU 吗？**  
A: **不算**。它们只是把仿真 restore 到某一 checkpoint；ORU 要求用离线 transition **算 loss 更新网络**。

**Q: explore 模式会训练网络吗？**  
A: **不会**。`explore=True` 时 `should_learn=False`，仅采 catalog。

---

*文档版本：T2 = offline replay update + curriculum（未实现）；job 22/27 为 legacy 占位。*
