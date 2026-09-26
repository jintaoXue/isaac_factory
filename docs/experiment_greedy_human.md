# G0-greedy：固定人分配规则，其余层从零训练

## 为什么选 D

| 层 | 第一版选择 | 理由 |
|---|---|---|
| A：产品释放 | 保持 G0 | 受订单类型和释放可行性约束，不是本次人因诊断对象 |
| B：WIP 优先级 | 保持 G0 | 涉及后续资源竞争，简单局部规则可能改变全局调度 |
| C：任务选择 | 保持 G0 | 工艺顺序与共享资源有耦合，先不同时改动 |
| D：人分配 | 固定贪心，不训练人 Q 头 | 当前效率和任务技能提供可解释的即时评分 |
| D：机器人分配 | 保持 G0 | 本次只检验人因匹配的收益 |

在当前步内信息池给出的合法工人集合中选取：

`h* = argmax_h efficiency[h] × clip(skill_task[h, selected_task])`

效率 η 已包含当前疲劳影响；技能 clip 使用当前 profile。相同分数选最小编号，无合法候选或 none task 返回零动作。不放宽 mask，不重复分配已占用工人。它是即时效率代理，不含子任务技能、路程、未来疲劳和下游资源需求，不宣称精确最短完工。

## 一键训练（无需教师、无需 checkpoint）

本机 `isaac-lab` 环境：

```bash
conda activate isaac-lab
cd /home/xue/work/isaac_factory
bash run_2026_journal_experiments.sh G0-greedy
```

只打印命令：

```bash
bash run_2026_journal_experiments.sh G0-greedy cuda:0 --dry-run
```

默认 cuda:0；gap、N10、K10、T35000、100 episodes，与 G0 使用相同训练超参。A/B/C、机器人 D 和共享 encoder 沿用 G0 的学习流程。人 D 始终使用规则，不受 ε 探索控制，不进入人回放，不计算人 TD loss，参数冻结；保留原六文件 checkpoint 接口。无 ORU、教师、AR、人因 shaping、match 网络。

目录 `logs/rl_games/HcFactory/hier_G0-greedy`，已存在时自动添加 `-v1` 等后缀；W&B 项目 `HcFactory_TPA`，名称 `G0-greedy-N10-S42`（重复运行带后缀）。不会覆盖 G0 或 G0-human-match。

## 训练后协议评测

```bash
bash run_2026_journal_experiments.sh eval-G0-greedy
```

自动寻找最新 G0-greedy 训练目录及 checkpoint。人 D 保持同一个贪心规则；N10、gap、ε=0、seeds 43–52×1，沿用现有 G horizon。用独立开发集选好 checkpoint 后可设置 `HC_LOAD_DIR=... HC_G_LOAD_STEP=...` 固定评测，不用协议 seeds 选模型。checkpoint 内标记固定人规则，误用普通 RL 人头加载时会报错。

## 查看结果

- W&B config：`train_cfg.params.config.human_policy = greedy`。
- `MetricPolicy/human_greedy = 1`；`MetricTrain/08_buffer_D_human = 0`。人头无 TD 更新是预期行为，不代表训练故障。
- 检查 A/B/C/机器人 D 的 loss；正式对比使用 `MetricFullorderCore/09_mean_makespan`、`/03_success_rate`、`/episode`（10 局）。
- 与 G0、G0-human-match 比较同样的评测 seeds 和交互预算。G0-greedy 的人分配不随机，早期训练曲线更好不自动证明学习加速；需要独立 ε=0 验证曲线。

## 可选：同 checkpoint 替换诊断

之前的成对诊断改名为 `diag-G0-greedy`，避免与训练后的评测混淆：

```bash
bash run_2026_journal_experiments.sh diag-G0-greedy
```

它需要已有 **G0** checkpoint，依次运行原策略 / 替换人 D，固定同一路径与 step，使用独立诊断 seeds 9001–9010×1，各 10 局。W&B 后缀 `diag-RL` / `diag-greedy-human`，项目 `HcFactory_TPA_Eval`。开关 `greedy_human_eval` 仅用于这一评测替换；新训练使用 `human_policy=greedy`。

贪心胜出说明可利用的人因即时收益存在；未胜出不代表 RL 已达到最优。训练与评测可以和现有任务并行，但同 GPU 会共享算力。
