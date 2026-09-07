# Hier4TPA 实验协议（精简看板）

> wandb entity：`rl-driving` · 快照日期：**2026-09-07**  
> 训练 project：`HcFactory_TPA` · 评测：`HcFactory_TPA_Eval` · 采库：`HcFactory_Catalog`  
> 主指标（训练曲线）：`MetricFullorderCore/05_makespan`（越低越好；≈40000 多为截断失败）

```bash
conda activate isaaclab
cd <repo_root>
```

---

## 1. 命名与协议总表

### 1.1 公式

```text
Hier4TPA-{T*}[+R][+H][+C{r|f}][+A][+E][+W]
```

字母顺序固定：**T → R → H → C → A → E → W**。

| 字母 | 轴 | 含义 |
|------|----|------|
| **T0 / T1 / T2** | 协议（互斥） | 无库 hard / 随机 catalog+ORU / **policy** catalog+ORU |
| **+R** | 后端 | Double DQN + PER + Dueling（无 Noisy / n-step / C51） |
| **+H** | 层级学习 | B-score RL + A/B 信用缩放 |
| **+C** | 课表 | 叠在 T1/T2 的 online 上；**Cr**=倒向（8→0），**Cf**=正向（0→难） |
| **+A** | Autoregressive（自回归）训法 | Scheduled sampling / TF、分层 ε、步内 multi-sample（**不是**新 T 号）；下文简称 **AR** |
| **+E** | 教师探索 | **在线**探索步用教师（如 T0）出动作，而非均匀随机；≠ 持续学习 |
| **+W** | 热启 | 学生网络从教师 ckpt **加载权重**再训 |

**三件套（勿混）**

| 代号 | 教师帮你做什么 |
|------|----------------|
| **T2** | **存数据**：用教师/好策略滚 `offline_replay` 再 ORU |
| **+E** | **在线探索**：训练中 ε-探索时问教师要动作 |
| **+W** | **起跑权重**：初始化=教师 ckpt |

**旧名对照**

| 旧 | 新 |
|----|-----|
| T2（随机+ORU+curriculum） | **T1C** / 主推 **T1RHC(Cr)** |
| T3（policy+ORU+hard） | **T2** |
| T4（policy+ORU+curriculum） | **T2C** / **T2RHC** |
| AR 改进 | **+A**（Autoregressive；如 `T1RA`），不占 T3 |
| 教师在线探索 | **+E**（如 `T1E` / `T2E`） |
| 教师权重热启 | **+W**（如 `T2EW`） |
| job 27 curriculum reset | **legacy**，≠ 任何 `T*C`（无 ORU loss） |

### 1.2 总表

| 短名 | Catalog | ORU | Online | 可叠加 | 命令 / 状态 |
|------|---------|-----|--------|--------|-------------|
| **T0** | 无 | 无 | Hard | +R +H +A +E +W | `./batch_train.sh T0` · ✅ 已训 |
| **T1** | 随机 ε=1 | ✅ | Hard | +R +H +C +A +E +W | `./batch_train.sh T1` · ✅ 已训 |
| **T1R** | 同 T1 | ✅ | Hard | | `./batch_train.sh T1R` · 🟡 训中 |
| **T1RH** | 同 T1 | ✅ | Hard | | `./batch_train.sh T1RH` · ❌ 未跑 |
| **T1C(Cr/Cf)** | 同 T1 | ✅ | Curriculum | 可无 R/H | 🔲 ORU+课表未接 |
| **T1RHC** | 同 T1 | ✅ | Curriculum | 主推课表形态 | 🔲 |
| **T1E / T1WE** | 同 T1 | ✅ | Hard+教师探索 | | 🔲 |
| **T2** | **Policy** | ✅ | Hard | +R +H +C +A +E +W | 🔲 未实现 |
| **T2E / T2EW** | Policy | ✅ | +教师探索 / +热启 | | 🔲 推荐消融 |
| **T2RHC** | Policy | ✅ | Curr. | | 🔲 |
| **\*A**（如 T1RA） | 随基底 | | | AR 技巧 | 🔲 |

```text
T0 ──► T1 ──► T1R ──► T1RH ──► T1RHC(Cr)     ← 随机库主链
         │      └(+E)(+A)(+W)
         │
         └──► T2 ──► T2E / T2EW ──► T2RHC     ← policy 库 + 教师探索/热启
```

**经验**：随机库强 ORU（T1）**不保证**优于 T0；下一步优先 **T2（教师存数据）**，再试 **+E / +W**；**+A** 治 AR exposure，与教师正交。

**消融**：课表可报 `T1C(Cr)`；教师三件套不要绑死——主表可只报 `T0 | T1 | T1R | T2 | T2E`。

---
## 2. 训练状态分类（对照 wandb）

### 2.1 主线

| 版本 | 状态 | 代表 run（`HcFactory_TPA`） | wandb | 备注 |
|------|------|------------------------------|-------|------|
| **T0** | ✅ 已训（可停） | `hier_hard_K10_N10_T40000` | `zynalxhz` | 曲线已平台 |
| **T1** | ✅ 已训（可停） | `hier_hard_ORU_…__legacy` | `jmy3yhun` | `p469o8sx` 重启可停 |
| **T1R** | 🟡 训中 | `hier_T1R_…__T1_random_ep20` | `78tdddig` | 5090 |
| **T1RH** | ❌ 未跑 | — | — | 代码已有 |
| **T1C / T1RHC** | ❌ 未实现 | `hier_curriculum_*` = legacy 27 | — | **勿标 T1C** |
| **T2…** | ❌ 未实现 | — | — | policy catalog（教师**存数据**） |
| **+E / +W** | ❌ 未实现 | — | — | 教师探索 / 热启 |
| **+A** | ❌ 未实现 | — | — | AR 技巧 |

### 2.2 配套 / 基线

| 类别 | 状态 | 代表 | 说明 |
|------|------|------|------|
| Explore 采库 | ✅ | Catalog: `…__T1_random_ep20` | 供 T1/T1R |
| Rule / Random eval | 🟡/⚠️ | 见 §3.2 | N=16 random crashed 等 |
| Hier 正式 eval | ⚠️ | `hier_eval_*` 部分 crashed | 缺统一 seed 协议 |

| 标签 | 含义 |
|------|------|
| ✅ | 可引用 |
| 🟡 | 进行中 / 未满程 |
| ⚠️ | 不完整 |
| ❌ | 未训或未实现 |

---

## 3. 表现速览（训练曲线 · 非最终论文表）

口径：`MetricFullorderCore/05_makespan` **后期 1/3 episode**；`sr` = success 均值。

### 3.1 Hier 训练（N=10 hard）

| 方法 | Run | Ep | late_med ↓ | late_std | min | trunc | sr | 相对 T0 |
|------|-----|----|------------|----------|-----|-------|-----|---------|
| **T0** | `zynalxhz` | 135 | **18300** | **1060** | **14600** | 0 | **1.00** | 基准；最稳 |
| **T1** | `jmy3yhun` | 112 | **17923** | 1071 | 15495 | 5 | 0.93 | 中位略好，尖峰多 |
| **T1R** | `78tdddig` | 82 | 18159 | 1271 | 15464 | 1 | 0.98 | 接近；未满程 |
| Legacy curr. | `yrmr4uxp` | 173 | ~17916 | 1439 | — | 3 | 0.97 | ≠T1C |
| Random 探 | `1r3e2wux` | 95 | 19144 | 3769 | 15788 | 3 | 0.97 | 下界 |

1. T0 最稳；T1 受随机 ORU 分布错配影响。  
2. 论文数字靠统一 **eval**，不靠训练曲线。

### 3.2 Eval 片段（`HcFactory_TPA_Eval`）

| 方法 | Run | n | med | 备注 |
|------|-----|---|-----|------|
| Random N=10 | `q850okke` | 10 | ~24890 | finished |
| Rule 短跑 N=10 | `k0zko3r6` | 10 | ~24200 | 非正式 |
| Hier≈1.6M | `7zeqtyf1` | 9 | ~19731 | crashed |
| Hier≈2.5M | `pmya4v6p` | 10 | ~22376 | ckpt 需核对 |
| Rule N=16 K10 | `e29hbc1s` | 10 | ~34988 | 勿与 N=10 横比 |

---

## 4. 最短命令

```bash
./batch_train.sh T0 cuda:0
HC_CATALOG_TAG=T1_random_ep20 ./batch_train.sh T1 cuda:0
HC_CATALOG_TAG=T1_random_ep20 ./batch_train.sh T1R cuda:0
HC_CATALOG_TAG=T1_random_ep20 ./batch_train.sh T1RH cuda:1

HC_CATALOG_TAG=T1_random_ep20 ./batch_train.sh 22 cuda:0   # 采库

HC_EVAL_VARIANT=T0 HC_LOAD_DIR=logs/rl_games/HcFactory/<dir> ./batch_train.sh 29 cuda:0
./batch_train.sh 24 25 26 cuda:0    # Rule/Random N=10
./batch_train.sh 30 31 32 cuda:0    # N=16
```

Catalog：`env_checkpoints/random_explore/N10_T40000__${HC_CATALOG_TAG}/`（需 `offline_replay/`）。

---

## 5. 环境变量（常用）

| 变量 | 作用 |
|------|------|
| `HC_CATALOG_TAG` | 采库/读库标签 |
| `HC_LOAD_DIR` / `HC_LOAD_STEP` | 评测权重 |
| `HC_TEST_SEEDS` / `HC_TEST_TIMES` | 评测 |
| `HC_ORU_*` | ORU warmup / mix 衰减 |

---

## 6. 下一步

| # | 动作 |
|---|------|
| 1 | 停 T0/T1；T1R 对齐步数后停 |
| 2 | 统一 eval：T0 / T1 / T1R |
| 3 | 跑 **T1RH** |
| 4 | 实现 **T2**（T0 作教师滚库）→ 可选 **T2E / T2EW** |
| 5 | **T1C(Cr)+ORU** → **T1RHC(Cr)**（≠ job 27） |
| 6 | **+A** 消融（如 T1RA）；收齐基线 eval |

---

## 7. Job 速查

| Job | 用途 |
|-----|------|
| 22 | explore + `offline_replay` |
| 24–26 | Rule/Random eval N=10 |
| 27 | legacy curr. **reset only** ≠ T1C |
| 28 | hard train（±ORU/PER） |
| 29 | hier eval |
| 30–32 | N=16 基线 eval |
