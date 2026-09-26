# 人因实验说明（奖励 / D 配对 / C 配对 / 耗时辅助）

> E0–E6 协议见 `experiment_protocol.md`；纯逻辑仿真见 `ideal_simulation_backend.md`。

训练与数值评测默认 **logic 后端**（不跑 PhysX）。

## G. gap 动力学新系列（主推，序号 G0–G4）

旧 E/T0 教师在 **legacy** 表上训成；直接热启到 **gap** 不公平。gap 实验请走 **G0–G4**（脚本会拒绝把 `HC_HUMAN_SKILL_PROFILE=gap` 挂在 E*/T0 上）。

| 入口 | 含义 | 目录 / W&B |
|--|--|--|
| **G0** | gap hard 新教师/基线（对位旧 T0；默认 100 ep；**无教师**） | `hier_G0` / `Hier4TPA-G0-N10-S42` |
| **G0-human-match** | 同 G0 的 scratch 配方 + 人因奖励 + D match（**无教师**；默认 100 ep） | `hier_G0-human-match` / `Hier4TPA-G0-human-match-N10-S42` |
| G1 / G2 / G3 | 可选：无教师 AR / G0 热启 no-oru / 仅人因 | `hier_G1` … |

**G 系列节奏（相对旧 E/T）：** 疲劳 strong/gap **×1.5**（不再 ×2）；静态加工时 /1.5；AGV/人 waypoints 与龙门 `move_speed` ×1.5；horizon **T=35000**（`t_max_anchor=56000`）。E 系列仍 T=40000。gap 早期 makespan 曾到 ~28k（legacy 人因 ~15–18k），上述缩放把墙钟与截断风险压回可训区间。

评测：`eval-G0`、`eval-G0-human-match`（seeds 43–52）。

```bash
# 两台可并行（都不依赖教师权重）
bash run_2026_journal_experiments.sh G0 cuda:0
bash run_2026_journal_experiments.sh G0-human-match cuda:0
```

教师目录可用 `HC_G_TEACHER_DIR`（默认 `hier_G0`）覆盖。

## 0. E 系列命名 + skill 表（legacy 动力学）

旧协议仍用 legacy 教师：`logs/rl_games/HcFactory/hier_2026-08-27_23-17-41`，`HC_LOAD_STEP=1290000`。

**W&B / 训练目录**：`{入口}-{legacy|strong|fast}`  
例：`E5-human-legacy`、`E5-human-match-fast`。防撞才设 `HC_HUMAN_RUN_TAG=r2`。**不要**再写 `*-gap` 挂在 E* 上。

**四档对比**（`HC_HUMAN_SKILL_PROFILE`；相对标称 skill=1 的机床操作）

| | **legacy**（默认，E 系列） | **strong** | **fast** | **gap**（仅 G 系列） |
|--|--|--|--|--|
| 设计意图 | 原表 | **错派更痛** + 疲劳快 | **专工更快**，错派只略慢 | **专工更快 + 错派更痛 + 疲劳×1.5** |
| 专工 / 错工艺 / 物流干工艺 | 1.40 / 0.58 / 0.70 | 1.55 / **0.42** / 0.45 | **1.75** / 0.62 / 0.68 | **1.75** / **0.42** / **0.45** |
| 子任务机床（专工 / 物流工） | ~1.3 / 0.72 | ~1.4 / **0.55** | **~1.5** / 0.85 | **~1.5** / **0.55** |
| 疲劳 work/recover | 原 | **×1.5** | 同 legacy | **×1.5（同 strong）** |
| clip(skill) | [0.35, 1.80] | [0.30, 2.20] | [0.35, 2.80] | [0.30, 2.80] |
| 对口专工时长（F=0） | ~0.55× 基准 | ~0.46× | **~0.37×（最短）** | **~0.37×（同 fast）** |
| 错派时长（F=0） | ~2.5–3.6× 专工 | **~4–7× 专工** | ~2.5–3× 专工 | **~5–8× 专工（最大）** |
| 全局 makespan 倾向 | 中性 | 易被错派/疲劳**拉长** | 选对人时**整体偏短** | 选对更短、选错更长 |

```bash
# E 系列（legacy 教师 / legacy 或 fast 表）
HC_MAX_TRAIN_EPISODES=60 bash run_2026_journal_experiments.sh E5-human cuda:0
HC_HUMAN_SKILL_PROFILE=fast HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human-match cuda:0

# gap → 必须 G0–G4（见 §G）
```

## 1. 当前机器分工（G 系列开工）


| 机器          | 入口                | 说明            |
| ----------- | ----------------- | ------------- |
| 本机（4090 工位） | `G0`              | gap 新教师       |
| 5090        | `G1`              | no-teacher 并行 |
| （G0 完成后）   | `eval-G0` → `G2` / `G4` | 评测与对照 |


E 系列旧分工可归档；暂缓 pair 系。

评测（E 系列；训 strong/fast 时评测也要同一 profile）：

```bash
bash run_2026_journal_experiments.sh eval-E5-human cuda:0
HC_HUMAN_SKILL_PROFILE=fast \
  bash run_2026_journal_experiments.sh eval-E5-human-match cuda:0
```

协议：N10/K10/T40000、seeds 43–52×1、ε=0；主指标 `MetricFullorderCore/09_mean_makespan`。评测关 shaping。

## 2. 入口一览（E 系列）


| 入口                    | D 配对 | C 配对 | 耗时辅助 | 人因 reward                     |
| --------------------- | ---- | ---- | ---- | ----------------------------- |
| `E5-human`            | 关    | 关    | 关    | 开（`HC_HUMAN_REWARD=false` 可关） |
| `E5-human-pair`       | 开    | 关    | 关    | 开                             |
| `E5-pair`             | 开    | 关    | 关    | 关                             |
| `E5-human-pair-c`     | 开    | 开    | 关    | 开                             |
| `E5-human-pair-aux`   | 开    | 关    | 开    | 开                             |
| `E5-human-pair-c-aux` | 开    | 开    | 开    | 开                             |
| `E5-human-match`      | match 双塔 | 关 | 关 | 开 |
| `E5-human-match-c`    | match | match 汇总 | 关 | 开 |


底座均为 E5-no-oru：T0 热启 + 教师探索 + AR，ORU 关。命名见 §0；`HC_HUMAN_RUN_TAG` 仅防撞可选。

## 3. 人因奖励（P0）

疲劳/技能已改工时；共享完工奖对单次派工信用弱，故在**成功派工**处加小 shaping（不改 mask / 动力学）。

`v(i,t)=η(F_i)×skill(i,t)`（task skill，非残留 subtask skill）。  
`g_d=clip(1-v(chosen)/max_{空闲}v, 0, 1)`。


| 分项               | 公式要点             | 默认                              |
| ---------------- | ---------------- | ------------------------------- |
| `human_mismatch` | `-λ_m Σ g_d`     | 0.05 / `HC_HUMAN_MISMATCH_COEF` |
| `human_overwork` | 在职工人 F>0.8 的二次惩罚 | 0.01 / `HC_HUMAN_OVERWORK_COEF` |
| `human_recovery` | 默认关              | 0 / `HC_HUMAN_RECOVERY_COEF`    |


限幅 `human_shaping_cap=0.04`（相对时间惩罚 −0.08/step）。看 `MetricReward/*`、`MetricHuman/ep_mismatch_rate`，最终仍看协议 makespan。

## 4. D-human 配对残差

`Q(s,t,h)=Q_old + MLP([encode_D(s,t), pair(t,h)])`。  
pair 八维：fatigue、η、skill、η×skill、疲劳增长率、恢复率、可用性、存在标记。末层零初始化，热启后与旧 Q 一致。旧 ckpt 可加载；含 pair 的 ckpt 必须用 pair 系评测入口。

## 5. C 头任务–人员汇总

对空闲人员集 H：`φ(t)=[max v, mean v, |H|/H_tot, argmax 的 fatigue, 1(|H|>0)]`。  
`Q_C_new = Q_C_old + MLP_C([encode_C, φ(t)]) × 1(|H|>0)`。零初始化；批次内人员占用写入 replay。

## 6. 耗时辅助（D）

在 D 配对共享隐层后加 Softplus 头；`y=log(1+d/1000)`，`L = L_TD + 0.05×Huber`。  
`d` = 完成步 − 派工步（含等待，非反事实）。缓冲 2048、满 32 才训；失败/未完成不造零标签。W&B：`MetricAux/*`、`MetricNetwork/*`。

## 7. 测试

```bash
python tests/test_human_aware_reward.py
python tests/test_human_extensions.py   # pair / C / aux
python tests/test_human_match.py
```

不保证 makespan 涨点；以协议评测为准。
