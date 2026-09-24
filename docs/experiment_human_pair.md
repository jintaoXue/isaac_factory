# D-human 配对残差网络（可切换）

只增强人员分配 D-human；原 QNetwork、encode_D 和机器人头保留。配置默认 `human_pair_head=false`，当前 E5-human 运行不受影响。

| 训练入口 | 网络 | 人因 reward | 教师 / AR / ORU |
|---|---|---|---|
| E5-human | 原网络 | 开 | T0 / 开 / 关 |
| E5-human-pair | 配对残差 | 开 | T0 / 开 / 关 |
| E5-pair | 配对残差 | 关 | T0 / 开 / 关 |

## 网络

`Q(s,t,h) = Q_old(s,t,h) + MLP([encode_D(s,t), pair(t,h)])`。

每个人员候选共用一个 MLP（隐藏层 64，ReLU，输出 1）。pair 的 8 个特征是 fatigue、η、当前任务 skill、η×skill、疲劳增长率、恢复率、可用性、存在标记。skill 按当前 C 头任务取值，避免使用上一任务的 skill_effective。不存在的槽位残差为零，动作仍使用原可行性 mask。

末层权重和偏置初始化为零；加载旧 T0 时，新旧 Q 输出一致，然后通过原 TD loss 学习残差。旧编码器和原网络参数仍按原训练方式更新。教师保留冻结，沿用原教师探索机制。

原 checkpoint 可以加载到新网络；含配对参数的新 checkpoint 必须通过 pair 入口评测，误用旧入口会报错而非丢弃参数。直接 Hydra 开关是 `agent.params.config.human_pair_head=true`。

## 命令（训练机 / isaac-lab）

选择空闲 GPU；重复实验改 tag，脚本拒绝覆盖同名训练目录。

```bash
conda activate isaac-lab
cd /home/xue/work/isaac_factory

# 仅打印配置
HC_HUMAN_RUN_TAG=pair-smoke-v1 bash run_2026_journal_experiments.sh E5-human-pair cuda:0 --dry-run

# 5 局冒烟；独立于正式训练
HC_HUMAN_RUN_TAG=pair-smoke-v1 HC_MAX_TRAIN_EPISODES=5 bash run_2026_journal_experiments.sh E5-human-pair cuda:0

# 正式训练：同一 T0 初始化，60 局
HC_HUMAN_RUN_TAG=pair-formal-v1 HC_MAX_TRAIN_EPISODES=60 bash run_2026_journal_experiments.sh E5-human-pair cuda:0

# 预先固定 300000 步；不要用协议评测 seeds 选 checkpoint
HC_LOAD_DIR=logs/rl_games/HcFactory/hier_E5-human-pair-pair-formal-v1 HC_LOAD_STEP=300000 bash run_2026_journal_experiments.sh eval-E5-human-pair cuda:0
```

协议维持 N10/K10/T40000、seeds43–52×1、epsilon=0，指标 `MetricFullorderCore/09_mean_makespan`。评测关闭 shaping、保留人因指标日志。独立 variant、目录和 W&B 名，不覆盖原扫参。

## 验证与比较

CPU 测试覆盖当前任务配对、mask、梯度/TD 更新、replay、target、冻结教师、checkpoint 保存/恢复，并用本机真实 T0 权重验证零残差等价。原人因奖励测试也通过。尚未运行 Isaac 仿真冒烟或验证 makespan 收益。

先比较 E5-human 与 E5-human-pair；需要拆开贡献时再跑 E5-pair，对照 E5-no-oru。保持训练种子、预算、教师、AR 与评测协议一致；性能收益需实验确认。
