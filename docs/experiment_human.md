# 人因实验说明（奖励 / pair / match 网络）

> E0–E6 协议见 `experiment_protocol.md`；纯逻辑仿真见 `ideal_simulation_backend.md`。

训练与数值评测默认 **logic 后端**（不跑 PhysX）。教师权重：`logs/rl_games/HcFactory/hier_2026-08-27_23-17-41`，`HC_LOAD_STEP=1290000`（六件套）。目录已存在时换 `HC_HUMAN_RUN_TAG`。

## 1. 当前机器分工

| 机器 | 入口 | 说明 |
|---|---|---|
| 本机（4090 工位） | `E5-human` | 原网络 + 人因奖励 |
| 5090 | `E5-human-pair-c-aux` | D pair + C 汇总 + 耗时辅助 |
| **家里台式** | **`E5-human-match`** | **★ 网络改进：D 双塔 match 打分** |
| 服务器 | `E5-human-pair-aux` | D pair + 耗时辅助 |

可选加跑：`E5-human-match-c`（D+C 都用 match）。暂缓：`E5-human-pair`、`E5-pair`。

```bash
# 本机
HC_HUMAN_RUN_TAG=human-logic-v1 HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human cuda:0

# 5090
HC_HUMAN_RUN_TAG=c-aux-logic-v1 HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human-pair-c-aux cuda:0

# 家里（网络改进）
git pull origin master
# 教师六件套放到 logs/rl_games/HcFactory/hier_2026-08-27_23-17-41/nn/
HC_HUMAN_RUN_TAG=match-v1 HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human-match cuda:0

# 服务器
HC_HUMAN_RUN_TAG=aux-logic-v1 HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human-pair-aux cuda:0
```

评测：

```bash
bash run_2026_journal_experiments.sh eval-E5-human-match cuda:0
# 或 HC_LOAD_DIR=... HC_LOAD_STEP=300000
```

协议：N10/K10/T40000、seeds 43–52×1、ε=0；主指标 `MetricFullorderCore/09_mean_makespan`。

## 2. 入口一览

| 入口 | D | C | aux | reward |
|---|---|---|---|---|
| `E5-human` | 旧 | 旧 | 关 | 开 |
| `E5-human-pair` | pair 残差 | 旧 | 关 | 开 |
| `E5-human-pair-c` | pair | pair 汇总 | 关 | 开 |
| `E5-human-pair-aux` | pair | 旧 | 开 | 开 |
| `E5-human-pair-c-aux` | pair | pair | 开 | 开 |
| **`E5-human-match`** | **match 双塔** | 旧 | 关 | 开 |
| **`E5-human-match-c`** | **match** | **match 汇总** | 关 | 开 |

Hydra：`human_match_head` / `task_match_head`（与 `human_pair_head` / `task_pair_head` **互斥**）。默认 tag：`match-v1` / `match-c-v1`。

## 3. ★ E5-human-match 网络改进（家里跑这个）

针对旧 D「全局 Q + 浅残差、特征与工时不对齐」：

1. **特征与时长对齐**：`speed = η × skill_task × skill_sub(control_machine)`，另含 `log_speed`、疲劳、速率等（10 维）。  
2. **双塔打分**：`Q = Q_context(base) + ⟨tower_h(human), tower_c(ctx)⟩ + feat_mlp + prior·log_speed`。  
3. **`prior_weight` 初值 = 1**：热启 T0 后立刻偏向更快工人（与旧 pair 零残差「完全等于旧 Q」不同）。  
4. C 侧 `E5-human-match-c` 用空闲池的 max/mean speed 等做同样双塔。

实现：`source/algo/hierarchical/hc_factory/human_match.py`。W&B：`MetricNetwork/human_match_head`、`task_match_head`。

## 4. 人因奖励（P0）

`v(i,t)=η×skill_task`（未乘 sub；match 网络已用对齐 speed）。  
`human_mismatch` / `overwork` / cap 同前。见 `HC_HUMAN_*` 环境变量。

## 5. 旧 pair / aux（简述）

- pair：`Q_old + MLP([z, pair8])`，末层零初始化。  
- C-pair：空闲人 η×skill 汇总残差。  
- aux：完成耗时 Huber，仅 pair 路径。

## 6. 测试

```bash
python tests/test_human_aware_reward.py
python tests/test_human_pair.py
python tests/test_human_extensions.py
python tests/test_human_match.py
```
