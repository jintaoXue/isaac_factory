# 人因实验说明（奖励 / D 配对 / C 配对 / 耗时辅助）

> E0–E6 协议见 `experiment_protocol.md`；纯逻辑仿真见 `ideal_simulation_backend.md`。

训练与数值评测默认 **logic 后端**（不跑 PhysX）。教师权重：`logs/rl_games/HcFactory/hier_2026-08-27_23-17-41`，`HC_LOAD_STEP=1290000`（六件套）。

## 0. 命名 + skill 表切换（三档）

**W&B / 训练目录**：`{入口}-{legacy|strong|fast}`  
例：`E5-human-legacy`、`E5-human-match-fast`。防撞才设 `HC_HUMAN_RUN_TAG=r2`。

**三档对比**（`HC_HUMAN_SKILL_PROFILE`；相对标称 skill=1 的机床操作）

| | **legacy**（默认） | **strong** | **fast**（推荐家里） |
|--|--|--|--|
| 设计意图 | 原表 | **错派更痛** + 疲劳快 | **专工更快**，错派只略慢 |
| 专工 / 错工艺 / 物流干工艺 | 1.40 / 0.58 / 0.70 | 1.55 / **0.42** / 0.45 | **1.75** / 0.62 / 0.68 |
| 子任务机床（专工 / 物流工） | ~1.3 / 0.72 | ~1.4 / **0.55** | **~1.5** / 0.85 |
| 疲劳 work/recover | 原 | **×3** | 同 legacy |
| clip(skill) | [0.35, 1.80] | [0.30, 2.20] | [0.35, 2.80] |
| 对口专工时长（F=0） | ~0.55× 基准 | ~0.46× | **~0.37×（最短）** |
| 错派时长（F=0） | ~2.5–3.6× 专工 | **~4–7× 专工** | ~2.5–3× 专工 |
| 全局 makespan 倾向 | 中性 | 易被错派/疲劳**拉长** | 选对人时**整体偏短** |

```bash
# 本机 / 5090 / 服务器（legacy）
HC_MAX_TRAIN_EPISODES=60 bash run_2026_journal_experiments.sh E5-human cuda:0
# → E5-human-legacy

# 家里（fast：往短 makespan）
HC_HUMAN_SKILL_PROFILE=fast HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human-match cuda:0
# → E5-human-match-fast
```

## 1. 当前机器分工


| 机器          | 入口                              | skill 表     | 说明               |
| ----------- | ------------------------------- | ---------- | ---------------- |
| 本机（4090 工位） | `E5-human` + `E5-human-match-c` | legacy     | 基线 + D/C match   |
| 5090        | `E5-human-pair-c-aux`           | legacy     | D pair + C + aux |
| 家里台式        | `E5-human-match`                | **fast**   | D match；往短 makespan |
| 服务器         | `E5-human-pair-aux`             | legacy     | D pair + aux     |


暂缓：`E5-human-pair`、`E5-pair`、`E5-human-pair-c`。

```bash
# 本机
HC_MAX_TRAIN_EPISODES=60 bash run_2026_journal_experiments.sh E5-human cuda:0
HC_MAX_TRAIN_EPISODES=60 bash run_2026_journal_experiments.sh E5-human-match-c cuda:0

# 5090
HC_MAX_TRAIN_EPISODES=60 bash run_2026_journal_experiments.sh E5-human-pair-c-aux cuda:0

# 家里
HC_HUMAN_SKILL_PROFILE=fast HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human-match cuda:0

# 服务器
HC_MAX_TRAIN_EPISODES=60 bash run_2026_journal_experiments.sh E5-human-pair-aux cuda:0
```

评测（默认找新名 `hier_{入口}-{legacy|strong}`；没有则回退旧目录如 `*-logic-v1`。训 strong 时评测也要设同一 profile）：

```bash
# legacy（本机 / 5090 / 服务器）
bash run_2026_journal_experiments.sh eval-E5-human cuda:0
bash run_2026_journal_experiments.sh eval-E5-human-match-c cuda:0
bash run_2026_journal_experiments.sh eval-E5-human-pair-c-aux cuda:0
bash run_2026_journal_experiments.sh eval-E5-human-pair-aux cuda:0

# 家里 fast match
HC_HUMAN_SKILL_PROFILE=fast \
  bash run_2026_journal_experiments.sh eval-E5-human-match cuda:0

# 或显式：HC_LOAD_DIR=... HC_LOAD_STEP=300000
```

协议：N10/K10/T40000、seeds 43–52×1、ε=0；主指标 `MetricFullorderCore/09_mean_makespan`。评测关 shaping。

## 2. 入口一览


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


底座均为 E5-no-oru：T0 热启 + 教师探索 + AR，ORU 关。命名见 §0（`{入口}-{legacy|strong}`）；`HC_HUMAN_RUN_TAG` 仅防撞可选。

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