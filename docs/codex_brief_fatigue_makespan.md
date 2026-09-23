# Codex brief: 拉大 vs E0 的 makespan 差距 — 聚焦 human fatigue / skill / efficiency

> 给 Codex / 下一任 agent 用的工作简报。日期：2026-09-23。  
> 作者意图：不同 E* setting 算法差距不大；认为 **进一步提高 makespan 的核心在 human fatigue + skill efficiency**；当前并行在挑 **E5-no-oru** ckpt。

---

## 0. 用户当前状态（必读）

1. **现象**：E0–E6 / 消融在 **协议评测**（N=10，seeds 43–52×1，ε=0）上差距很小。  
   - 已满 10 局最好：`E5-no-oru @ step 335000`，mean makespan ≈ **17220**  
   - E0（教师）≈ **17576** → 相对提升仅 ~2%，不够当强主结果。  
2. **正在做**：挑 `E5-no-oru`（`hier_2026-09-18_21-14-57` / W&B `a2n538na`）更好 ckpt；近峰 10 + 窗外 10 在扫。  
3. **判断**：再堆 ORU / AR / H / teacher_explore 的 **setting 差** 收益有限；应改 **人因相关的学习信号与 D（分配）决策**。  
4. **约束**：协议评测不变；不要用评测 seeds 选 ckpt；改算法后需新训 + 同协议 eval。

---

## 1. 现状诊断（代码事实）

### 1.1 人因已在「仿真动力学 + 观测」里，但几乎不在「优化目标」里

| 层级 | 状态 | 位置 |
| --- | --- | --- |
| Fatigue 动力学 | ✅ 已实现 | `cfg_human.py`：`human_step_fatigue` |
| F→η efficiency | ✅ 已实现 | `human_efficiency`；行走/子任务时长乘 `η × skill_effective` |
| 异质 skill | ✅ 已实现 | `HUMAN_SKILL_TASK` / `HUMAN_SKILL_SUBTASK`，5 名工人 idx 0–4 |
| Obs 编码 | ✅ 进网络 | `data_preprocess_for_buffer.py` + `hier_networks.py`（fatigue / efficiency / skill_* / rates） |
| W&B 监控 | ✅ 只日志 | `HumanFatigueMonitor` → `MetricHuman/*`（**不进梯度**） |
| **Reward** | ❌ **无 fatigue/skill 项** | `task_progress_manager`：完成增量 + 时间惩罚 + success_bonus；`compute_team_reward` 读 `rl.reward` |

结论：环境已经「疲劳会拖慢节拍」，但 RL 仍主要学 **共享 makespan / 完工**。E* 开关（H / b_score / ORU / TE / AR）**都不直接强化「按疲劳与专长派工」**，所以 setting 曲线挤在一起、相对 E0 难拉开，是合理现象。

### 1.2 为何训练曲线好看、协议评测拉不开

- Journal 线：`ε=0.05` 锁死 +（E3+）`teacher_explore` 前 300k 探索支大量是冻结 T0 → 训练 FO 虚高。  
- 评测：ε=0 纯学生；E5-no-oru 训练 peak（360k）协议 eval 差，近峰 335k 才略好于 E0。  
- 详见对话结论：训测不一致 + 单局噪声；**选 ckpt 必须看协议 eval mean，不是训练 lowest makespan**。

### 1.3 E5-no-oru 选点进度（给 Codex 别重复造轮子）

- Run dir：`logs/rl_games/HcFactory/hier_2026-09-18_21-14-57`  
- 入口：`eval-E5-no-oru-near`（**340k–380k**，已跳过满评的 335k）、`eval-E5-no-oru-far`（窗外次优 10）  
- 文档：`docs/eval_checkpoint_selection.md`  
- 回家扫参包（若相关）：`_ckpt_export/E5-no-oru_home_eval/`（勿写 Obsidian）

---

## 2. 改进方向（按推荐优先级）

目标：**让 D（及必要时 C）显式学会「疲劳感知 + 技能匹配」的分配**，使 makespan 相对 E0 出现 **稳定、可复现** 的差距（协议 10 局 mean）。

### P0 — 学习信号（最可能拉开差距）

1. **Human-aware shaping（加进 `rl.reward` / `reward_parts`，可消融开关）**  
   - 例：错配惩罚 — 派 `skill_effective` 低的人做专工任务 → 负奖励；高匹配 → 小正奖励。  
   - 例：过劳惩罚 — `max_fatigue` 或 `mean(η)` 过低时逐步惩罚（避免只卷短期完工、把工人打满 F）。  
   - 例：恢复奖励 — 高疲劳工人被安排空闲/低 λ_s 子任务时小正奖励。  
   - **要求**：所有 shaping 进 `reward_parts`，W&B 可拆；默认关，journal 新 variant 打开。  
   - **警告**：shaping 过猛会偏离 makespan；用小系数 + 协议 eval 守门。

2. **Dense efficiency proxy（仍对齐 makespan）**  
   - 在决策间隔上增加与「预期剩余工时」相关的项：例如用当前 `η×skill` 估计的 busy 人力有效产能。  
   - 或对 `mean_per_product_span` 的增量给短视惩罚（已有相关指标思路，见论文笔记）。

### P1 — 策略结构（D 头 / 表征）

3. **D 分配显式 skill-fatigue 特征**  
   - 确认 agent_D 的 Q 输入里，候选人与当前 task 的 `skill_effective`、当前 `fatigue/η` 是否足够突出（勿被全局 state 淹没）。  
   - 可选：D 的 action 侧加 **pair feature**（human_i × task_t 的 skill 标量 + fatigue）再进 Q（小改网络）。  

4. **b_score / 优先级与疲劳耦合（若开 E4/E6）**  
   - 今日 E5 无 H；若完整方法 E6，可让 B 的可学习分在 WIP 上偏好「有合适低疲劳专工」的产品。  
   - 单独消融，避免和 shaping 缠死。

5. **Action mask 软约束（慎用）**  
   - 例如 `F > F_crit` 时禁止再派重负荷 subtask（硬 mask）或降权。  
   - 硬 mask 可能损害可行完备性；优先 shaping，mask 作安全网。

### P2 — 数据与探索（配合人因）

6. **教师 / ORU 要「会用人」**  
   - 若 TE/ORU 仍是旧 T0（未必按专长疲劳优化），学生会被拉向非人因最优。  
   - 可选：rule 基线加入 **skill-aware + fatigue-aware** 启发式，采库给 ORU；或重新采教师库。  

7. **评测报表加人因 KPI**  
   - 协议 eval 除 makespan 外固定报：`ep_mean_fatigue`、`mean skill_effective of assigned`、错配率。  
   - 用于证明「变快是因为用人，而不是偶然」。

### P3 — 先别做（除非 P0/P1 无效）

- 再开一长串 E* 开关排列组合（用户已认为差距不够）。  
- 只靠调 ε / AR 候选数指望大幅超过 E0。  
- 用训练曲线 lowest 当论文主 ckpt。

---

## 3. 建议的落地顺序（给 Codex 执行时用）

1. **读码确认**：`cfg_human.py`、`human.py` 时长缩放、`task_progress_manager.update_rl_signals`、`hier_rl_agents` D 头 obs、`hier_networks` human 特征维。  
2. **实现可开关 shaping**（Hydra / `algo_variant` 或 `human_aware_reward=true`），写清公式与默认系数。  
3. **单测**：假 obs 上 reward_parts 数值；疲劳升高时 duration 变长（已有动力学则回归即可）。  
4. **短训冒烟**（如 5–10 ep）看 `MetricHuman` + makespan 方向。  
5. **正式**：与 E0 / 当前 E5-no-oru@335k **同协议** 对比；报告 mean±分位 + 人因 KPI。  
6. **并行**：用户继续扫 E5-no-oru ckpt；新算法用新 wandb 名，勿覆盖旧 eval。

---

## 4. 成功标准（论文口径）

- 协议 10 局：相对 E0 mean makespan **明显且稳定**优于 ~2%（具体阈值由用户定，建议至少稳定 >5% 或置信区间不重叠）。  
- Ablation：关掉 human-aware 项后优势消失或明显变小。  
- 人因 KPI 与 makespan 同向（不是靠把工人打满疲劳换短时完工）。

---

## 5. 相关文件速查

```
source/isaaclab_tasks/.../env_asset_cfg/cfg_human.py          # fatigue/skill/η
source/isaaclab_tasks/.../src/human.py                         # step 更新
source/isaaclab_tasks/.../src/task_progress_manager.py         # rl.reward
source/algo/hierarchical/hc_factory/hier_utils.py              # compute_team_reward
source/algo/hierarchical/hc_factory/hier_networks.py           # obs 里的 human 因子
source/algo/hierarchical/hc_factory/wandb_metrics.py           # MetricHuman
docs/experiment_protocol.md                                   # E* 旋钮表
docs/eval_checkpoint_selection.md                             # ckpt / eval
2026_Journal_Paper.md §8.31                                   # 人因已实现说明
```

---

## 6. 一句话任务给 Codex

> 在 **不破坏现有协议评测** 的前提下，把 **human fatigue / skill / efficiency** 从「只影响仿真与观测」推进到 **可学习的奖励与/或 D 分配表征**，使新训策略在 seeds 43–52 上相对 E0 拉开 makespan；与用户并行的 E5-no-oru ckpt 扫参分开记账。
