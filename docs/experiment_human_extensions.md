# 人因网络进阶实验：C 配对与耗时辅助学习

保留现有 E5-human-pair，新增两个默认关闭的独立开关。原 QNetwork、encode_C、encode_D 接口不变；不改 reward、仿真动力学或动作可行性 mask。

## 实验表与切换

| 入口 | D 配对 | C 配对 | 耗时辅助 | 建议机器 |
|---|---|---|---|---|
| E5-human | 关 | 关 | 关 | 原奖励基线 |
| E5-human-pair | 开 | 关 | 关 | 当前运行 |
| E5-human-pair-c | 开 | 开 | 关 | 第二台 |
| E5-human-pair-aux | 开 | 关 | 开 | 第三台 |
| E5-human-pair-c-aux | 开 | 开 | 开 | 单项有效后组合 |

以上入口都保留人因奖励、T0 热启动与教师探索、AR；ORU 关闭。新实验使用独立目录/W&B 名。E5-pair（只有 D 配对、无人因奖励）仍可用。Hydra 配置位于 `agent.params.config`：`human_pair_head`、`task_pair_head`、`human_duration_aux`；三者默认 false，aux 要求 D 配对开启。

## 1. C 头配对：选任务时考虑可用人员

对每个任务 t，将当前尚未被本批次占用的真实空闲人员组成 H，定义 v(h,t)=η(h)×skill(h,t)。构造五维特征：

`φ(t) = [max_H v, mean_H v, |H|/真实人员数, argmax_H v 对应人员的 fatigue, 1(|H|>0)]`。

none 任务和无可用人员时特征全零；当前环境所有空闲人员均可执行各任务，技能差异影响效率而非硬可行性。任务、机器等限制仍由原 C mask 决定。

`Q_C_new(s,b,t) = Q_C_old(s,b,t) + MLP_C([encode_C(s,b), φ(t)]) × 1(|H|>0)`。

各任务共享一个隐藏层 64、ReLU、输出 1 的 MLP，末层零初始化。旧 T0 加载后的初始 Q 完全一致。训练仍用 C 的 TD loss；不是强制采用最快任务的贪心规则。批次中实际使用的人员可用性保存到 replay；下一状态使用下一状态自己的可用人员。

意义：补足“任务已经选定后 D 才考虑技能疲劳”的结构限制。收益假设是更合适的任务顺序可减少等待，需由 makespan 验证。

## 2. 耗时辅助头：给 D 表征增加完成记录监督

在 D 配对残差的共享 64 维隐藏层后增加独立 Linear→Softplus 预测头。Q 分支和耗时头共同更新该隐藏层及状态编码器；耗时预测不直接替换 Q、不改变 reward。

标签：成功派工前保存观测、任务和人员；真实完成时，以 `d = 完成步末时刻 − 派工时刻` 配对标签。该量是整项任务的实际经过时间，含协作/等待，不等于人员净作业时间，也不代表换另一个人的反事实耗时。

`y = log(1 + d / 1000)`；`L_D = L_TD + 0.05 × Huber(ŷ, y)`。

默认 `duration_aux_weight=0.05`、`duration_aux_scale=1000`、Huber beta=1；CPU 缓冲最多 2048 个完成样本，每次抽 32 个，积累满 32 个后开始辅助训练。每次 D 在线学习合并一次辅助 loss，避免 K 次派工把权重放大。参数可用 `HC_DURATION_AUX_WEIGHT` / `HC_DURATION_AUX_SCALE` 设置。

失败派工不生成 start；未完成/超时任务不伪造零标签；终止步已完成的任务保留，剩余 pending 丢弃；环境恢复跳转清空对应 pending。输入只使用派工前信息。辅助 replay 不存进模型 checkpoint，重新启动后重新积累；模型参数和预测头正常保存/恢复。

风险：只有完成样本，存在对长时间未完成任务的样本偏差；局部耗时也不等于全局 makespan。因此采用小权重，与无 aux 版本做同预算比较，不以辅助 loss 下降宣称策略改善。

## 运行命令

在另外两台机器各自同步本次代码和同一 T0 checkpoint，激活本机 `isaac-lab` 环境并进入仓库。默认教师路径为 `logs/rl_games/HcFactory/hier_2026-08-27_23-17-41`；不同位置设置 `HC_HUMAN_TEACHER_DIR`（指向含 nn 的目录）。

```bash
conda activate isaac-lab
cd /home/xue/work/isaac_factory  # 另一台机器改为实际仓库路径

# 第二台：C 配对；先预览、再 5 局冒烟、最后正式训练
HC_HUMAN_RUN_TAG=c-smoke bash run_2026_journal_experiments.sh E5-human-pair-c cuda:0 --dry-run
HC_HUMAN_RUN_TAG=c-smoke HC_MAX_TRAIN_EPISODES=5 bash run_2026_journal_experiments.sh E5-human-pair-c cuda:0
bash run_2026_journal_experiments.sh E5-human-pair-c cuda:0

# 第三台：耗时辅助
HC_HUMAN_RUN_TAG=aux-smoke bash run_2026_journal_experiments.sh E5-human-pair-aux cuda:0 --dry-run
HC_HUMAN_RUN_TAG=aux-smoke HC_MAX_TRAIN_EPISODES=5 bash run_2026_journal_experiments.sh E5-human-pair-aux cuda:0
bash run_2026_journal_experiments.sh E5-human-pair-aux cuda:0

# 在各自训练结束后，评测预先固定的 300000 步（可先在末尾加 --dry-run）
bash run_2026_journal_experiments.sh eval-E5-human-pair-c cuda:0
bash run_2026_journal_experiments.sh eval-E5-human-pair-aux cuda:0
```

默认 tag：C 为 `c-v1`、aux 为 `aux-v1`、组合为 `c-aux-v1`；评测默认读取相应目录的 step300000。可用 `HC_HUMAN_RUN_TAG`、`HC_MAX_TRAIN_EPISODES`、`HC_LOAD_DIR`、`HC_LOAD_STEP` 覆盖。旧时间戳目录须显式设置 `HC_LOAD_DIR`，不会自动搜索或挑选。

训练固定 N10/K10/T40000、seed42；60 局与原入口一致。正式比较另记录总交互步数，因每局长度不同，60 局不代表严格等步数。新目录已存在时脚本拒绝覆盖，重跑换 tag。组合版使用 `E5-human-pair-c-aux` / `eval-E5-human-pair-c-aux`。

评测仍为 N10、seeds43–52×1、epsilon=0；shaping 关闭，aux 不执行梯度更新，主指标 `MetricFullorderCore/09_mean_makespan`。保留 aux 架构开关以严格加载预测头。不得用协议 seeds 挑 checkpoint；使用预先固定步数或独立验证 seeds。旧权重可以初始化新头；新 checkpoint 需对应评测入口，错误关闭已训练结构时会报错。

## W&B 确认与验证边界

- `MetricNetwork/task_pair_head`、`human_pair_head`、`human_duration_aux`：确认实际开关。
- `MetricAux/duration_samples` / `duration_buffer`：应随真实完成记录增长；`duration_updates` 在满 32 样本且 D 学习后增长。
- `MetricAux/duration_loss` / `duration_mae_steps`：最近 100 次辅助更新的训练采样误差；不是独立验证误差。
- `duration_pending` / `duration_discarded` / `duration_unmatched`：监控未完成及无法匹配样本；普通完整训练 unmatched 应接近零。
- D-human critic 日志含加权辅助项；判断性能仍看协议 makespan，同时观察原 MetricHuman / MetricReward。

CPU 测试覆盖零初始化等价、实际 T0 迁移、C mask/replay/批内人员上下文、教师冻结、辅助梯度、完成事件、终止/恢复边界及 checkpoint 恢复。未启动 Isaac 仿真、正式训练或协议评测；仿真兼容性及性能收益仍需上述冒烟和正式实验确认。
