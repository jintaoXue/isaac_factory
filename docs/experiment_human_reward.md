# E5-human：第一版人因奖励（2026-09-23）

以 `codex_brief_fatigue_makespan.md` 为准。P0 已实现，P1 暂不改。原 E0–E6、E5-no-oru 训练与 near/far 扫参入口保留。

## 判断

- fatigue/skill 已影响工时和观测；共享完工奖励对单次派工的信用分配较弱，现有 E* 旋钮没有直接补充人因信号。
- ε=0.05＋教师探索的训练结果不能代表 ε=0 的学生表现；训练单局低点也不是稳定收益证据。
- 先在成功派工处增加效率错配惩罚，并对正在工作的高疲劳工人增加小惩罚；不改网络、mask 或仿真动力学。
- 没有证据保证此改动超过 2% 或 5%；先看 shaping/KPI 是否生效，再比较协议 makespan。

## 新入口与隔离

| 入口 | 底座 / 初始化 | 奖励 | 目录与名称 |
|---|---|---|---|
| `E5-human` | 原 E5-no-oru 的 TE＋AR；T0 step1290000 初始化 | 默认开启 P0 | `hier_E5-human-{tag}` / `Hier4TPA-E5-human-N10-S42-{tag}` |
| `HC_HUMAN_REWARD=false ... E5-human` | 同上 | 关闭 shaping，保留人因 KPI | `hier_E5-human-off-{tag}` / 独立 W&B 名 |
| `eval-E5-human` | 显式指定目录和步数 | shaping 关闭，KPI 开启 | 新时间戳/进程号标记的 eval 目录和 W&B run |

默认配置 `human_aware_reward=false`、`human_reward_metrics=false`，旧方法的 `rl.reward` / `reward_parts` 保持原样。通过 `agent.params.config.human_aware_reward=true` 可以单独启用 Hydra 开关。训练目录已存在时拒绝覆盖，重复实验请换 tag。新 run 清除继承的 W&B run ID，不接续扫参 run。

## 公式与默认值

对当前任务 t 和当前可用工人 i，定义任务级速度代理：

`v(i,t) = human_efficiency(F_i) × human_effective_skill(i,t,None)`。

这里使用 task skill，**不是空闲工人残留的上一个 subtask 的 skill_effective**；也不是包含路程/子任务组合的精确工时预测。

每个真实成功派工 d，令 `A_d` 为该次派工前仍空闲的工人（与当前 human mask 规则一致）；连续派工逐次排除已占用工人。`g_d = clip(1 - v(chosen,t)/max_{i∈A_d}v(i,t), 0, 1)`。

| reward_parts | 原始分项 | 默认参数 / journal 环境变量 |
|---|---|---|
| `human_mismatch` | `-λ_m × Σ_d g_d` | `human_mismatch_coef=0.05` / `HC_HUMAN_MISMATCH_COEF` |
| `human_overwork` | `-λ_o/H × Σ_i active_i × [max(0,F_i-f*)/(1-f*)]^2` | `human_overwork_coef=0.01` / `HC_HUMAN_OVERWORK_COEF` |
| `human_recovery` | `+λ_r/H × Σ_i idle_i × 1[F_before≥f*] × max(0,F_before-F_after)` | `human_recovery_coef=0` / `HC_HUMAN_RECOVERY_COEF` |

`H` 为工人数；active/idle 取本步动力学执行前的真实 human subtask，wait/done/none/free 不算工作；F 取本步动力学执行后值。失败的派工不会获得错配奖惩。

- `human_fatigue_threshold=0.8` / `HC_HUMAN_FATIGUE_THRESHOLD`。
- `human_shaping_cap=0.04` / `HC_HUMAN_SHAPING_CAP`。
- 各分项统一乘 `min(1, cap/Σ|分项|)`，再加入原 `rl.reward`；日志记录的是限幅后实际进入 reward 的值。
- 原时间惩罚为 `-0.08/step`，默认 shaping 绝对值总和不超过 `0.04/step`；恢复项默认零。建议首轮不提高 cap，防止鼓励等待、牺牲全局 makespan。
- 这些是奖励塑形启发式，不是保证最优策略不变的 potential-based shaping。技能匹配可能与路程/资源等待冲突，因此用小权重并保留关掉 shaping 的对照。
- 原决策回报缩放、SMDP 折扣和各层 TD 更新保持不变；D 网络权重可直接热启动。

## 运行命令

在当前 **xue@sci**（或同步代码和教师权重后的训练机）执行；当前 cwd 为 `/home/xue/work/isaac_factory`，conda 环境 `isaac-lab`。GPU 编号按空闲设备改，不停止其他实验。

```bash
conda activate isaac-lab
cd /home/xue/work/isaac_factory

# CPU 单测，不启动 Isaac Sim
python tests/test_human_aware_reward.py

# 预览：不创建 run，不占 GPU
HC_HUMAN_RUN_TAG=smoke-v1 HC_MAX_TRAIN_EPISODES=5 \
  bash run_2026_journal_experiments.sh E5-human cuda:0 --dry-run

# 独立 5 局冒烟，训练 seed=42
HC_HUMAN_RUN_TAG=smoke-v1 HC_MAX_TRAIN_EPISODES=5 \
  bash run_2026_journal_experiments.sh E5-human cuda:0

# 正式训练，仍从同一 T0 初始化，不从冒烟模型接着训
HC_HUMAN_RUN_TAG=formal-v1 HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human cuda:0

# 关闭 shaping 的匹配对照（独立名字与目录）
HC_HUMAN_REWARD=false HC_HUMAN_RUN_TAG=control-v1 HC_MAX_TRAIN_EPISODES=60 \
  bash run_2026_journal_experiments.sh E5-human cuda:0
```

教师默认目录：`logs/rl_games/HcFactory/hier_2026-08-27_23-17-41`，step1290000 的 encoder、A/B/C/D-human/D-robot 六个文件必须齐全。可用 `HC_HUMAN_TEACHER_DIR` 指向另一位置（包含 nn/）。

下例**预先固定新增 step300000**，不根据 seeds43–52 结果选择 checkpoint。若要选点，应另设与训练/协议测试不相交的验证集；本轮不实现新的扫参流程。已有 E5-no-oru@335k 可作背景参考；严格归因需对照同初始化、同新增步数的 human-off@300k。

```bash
# 正式训练保存 step300000 后：先预览，再去掉 --dry-run 评测
HC_LOAD_DIR="$PWD/logs/rl_games/HcFactory/hier_E5-human-formal-v1" \
HC_LOAD_STEP=300000 \
  bash run_2026_journal_experiments.sh eval-E5-human cuda:0 --dry-run

HC_LOAD_DIR="$PWD/logs/rl_games/HcFactory/hier_E5-human-formal-v1" \
HC_LOAD_STEP=300000 \
  bash run_2026_journal_experiments.sh eval-E5-human cuda:0
```

协议固定 **N10、K10、T40000、seeds43–52×1、ε=0**，沿用 job29 的分块/完整性检查。评测 shaping 关闭，既不改变策略动作也不改变原 makespan/return 定义；只增加人因 KPI。对照目录替换为 `hier_E5-human-off-control-v1` 即可，评测不写原 sweep 输出。

## W&B 检查

训练项目 `HcFactory_TPA`，评测项目 `HcFactory_TPA_Eval`；同内容写入各 run 的 `metrics.jsonl`。

| Key | 预期 / 用途 |
|---|---|
| `MetricReward/enabled` | 新训练=1；消融/协议评测=0 |
| `MetricReward/ep_human_mismatch_sum` | 存在比所派工人更快的空闲工人时为负 |
| `MetricReward/ep_human_overwork_sum` | 工作工人疲劳超过0.8时为负；未超过则0合理 |
| `MetricReward/ep_human_recovery_sum` | 默认0，显式开启后可能为正 |
| `MetricReward/ep_human_total_sum`、`ep_human_total_mean` | 实际加到 reward 的 shaping，重点核对量级 |
| `MetricReward/ep_step_sum`、`ep_total_sum` | 比较原时间成本与总奖励；不要拿 shaped return 横比旧 return |
| `MetricReward/ep_cap_hit_mean` | 限幅命中率，过高说明原系数太激进 |
| `MetricHuman/ep_dispatch_count` | 成功派工次数，不是每 tick 重复计数 |
| `MetricHuman/ep_assigned_skill_mean` | 派工时当前任务的 task-level skill 均值 |
| `MetricHuman/ep_assigned_speed_mean`、`ep_assigned_gap_mean` | η×skill 代理及相对最佳空闲候选差距 |
| `MetricHuman/ep_mismatch_rate` | `g_d>0.05` 的派工比例；关注下降 |
| `MetricHuman/ep_mean_fatigue` | 沿用现有人因指标 |
| `MetricFullorderCore/09_mean_makespan` | 主结果，结合成功率、满10局判断 |

单步 `MetricReward/human_*` 可能多数为0（派工稀疏），应优先看 episode 分项。所有分项会有 `ep_*_sum/mean`；终止步也被计入。

## 改动文件与验证

- `src/human_aware_reward.py`：纯 Python 的公式、派工事件和限幅。
- `src/task_progress_manager.py`：成功派工挂接、`rl.reward/reward_parts`。
- `hc_single_env_base.py`：动力学前疲劳/工作快照。
- `hc_vector_env_base.py`：Hydra 配置传到每个 TaskManager。
- `algo_cfg/hier.yaml`：默认关闭的配置项。
- `source/algo/hierarchical/hc_factory/wandb_metrics.py`：分项及派工 KPI 聚合。
- `source/algo/hierarchical/hc_factory/hierarchical_tpa.py`：新 variant 校验、终止步奖励日志。
- `run_2026_journal_experiments.sh`、`batch_train.sh`：独立训练/消融/协议评测入口。
- `tests/test_human_aware_reward.py`：无需 Isaac Sim 的真实方法 CPU 测试。

上述前5项位于 `source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/` 下。已通过13项CPU测试、shell/Python语法检查、实际Hydra覆盖参数解析和旧训练/扫参函数不变检查。**尚未运行实际 Isaac Sim 五局冒烟或正式训练，未验证性能提升。**
