# 人因实验说明（奖励 / D 配对 / C 配对 / 耗时辅助）

> E0–E6 协议见 `experiment_protocol.md`；纯逻辑仿真见 `ideal_simulation_backend.md`。

训练与数值评测默认 **logic 后端**（不跑 PhysX）。教师权重：`logs/rl_games/HcFactory/hier_2026-08-27_23-17-41`，`HC_LOAD_STEP=1290000`（六件套）。目录已存在时换 `HC_HUMAN_RUN_TAG`。

## 1. 当前机器分工

| 机器 | 入口 | 说明 |
|---|---|---|
| 本机（4090 工位） | `E5-human` | 原网络 + 人因奖励 |
| 5090 | `E5-human-pair-c-aux` | D 配对 + C 汇总 + 耗时辅助 |
| 家里台式 | `E5-human-pair-c` | D 配对 + C 汇总 |
| 服务器 | `E5-human-pair-aux` | D 配对 + 耗时辅助 |

暂缓：`E5-human-pair`、`E5-pair`（消融）。

```bash
# 本机
HC_HUMAN_RUN_TAG=human-logic-v1 HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human cuda:0

# 5090
HC_HUMAN_RUN_TAG=c-aux-logic-v1 HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human-pair-c-aux cuda:0

# 家里
HC_HUMAN_RUN_TAG=c-logic-v1 HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human-pair-c cuda:0

# 服务器
HC_HUMAN_RUN_TAG=aux-logic-v1 HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human-pair-aux cuda:0
```

评测（训练目录 + 预固定步数，勿用协议 seeds 挑点）：

```bash
bash run_2026_journal_experiments.sh eval-E5-human cuda:0
bash run_2026_journal_experiments.sh eval-E5-human-pair-c-aux cuda:0
bash run_2026_journal_experiments.sh eval-E5-human-pair-c cuda:0
bash run_2026_journal_experiments.sh eval-E5-human-pair-aux cuda:0
# 或显式：HC_LOAD_DIR=... HC_LOAD_STEP=300000
```

协议：N10/K10/T40000、seeds 43–52×1、ε=0；主指标 `MetricFullorderCore/09_mean_makespan`。评测关 shaping。

## 2. 入口一览

| 入口 | D 配对 | C 配对 | 耗时辅助 | 人因 reward |
|---|---|---|---|---|
| `E5-human` | 关 | 关 | 关 | 开（`HC_HUMAN_REWARD=false` 可关） |
| `E5-human-pair` | 开 | 关 | 关 | 开 |
| `E5-pair` | 开 | 关 | 关 | 关 |
| `E5-human-pair-c` | 开 | 开 | 关 | 开 |
| `E5-human-pair-aux` | 开 | 关 | 开 | 开 |
| `E5-human-pair-c-aux` | 开 | 开 | 开 | 开 |

底座均为 E5-no-oru：T0 热启 + 教师探索 + AR，ORU 关。Hydra：`human_aware_reward`、`human_pair_head`、`task_pair_head`、`human_duration_aux`（后三者默认 false；aux 需 D 配对开）。

默认 tag：human=`formal-v1`，pair=`pair-formal-v1`，C=`c-v1`，aux=`aux-v1`，组合=`c-aux-v1`。逻辑重跑请用 `*-logic-v1` 等新 tag。

## 3. 人因奖励（P0）

疲劳/技能已改工时；共享完工奖对单次派工信用弱，故在**成功派工**处加小 shaping（不改 mask / 动力学）。

`v(i,t)=η(F_i)×skill(i,t)`（task skill，非残留 subtask skill）。  
`g_d=clip(1-v(chosen)/max_{空闲}v, 0, 1)`。

| 分项 | 公式要点 | 默认 |
|---|---|---|
| `human_mismatch` | `-λ_m Σ g_d` | 0.05 / `HC_HUMAN_MISMATCH_COEF` |
| `human_overwork` | 在职工人 F>0.8 的二次惩罚 | 0.01 / `HC_HUMAN_OVERWORK_COEF` |
| `human_recovery` | 默认关 | 0 / `HC_HUMAN_RECOVERY_COEF` |

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
```

不保证 makespan 涨点；以协议评测为准。
