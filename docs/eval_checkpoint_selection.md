# 评测 Checkpoint 选择（训练最优，非 latest）

> 更新：2026-09-22。与 T0 教师选点同一精神：**训练曲线 makespan 最低点 → 最近 `save_interval` 保存步**。  
> **不要**用评测 seeds 43–52 来挑 ckpt（协议：测试集不参与选择）。

## 选择规则（固定）

1. 指标：`MetricFullorderCore/05_makespan`（越低越好）。  
2. 过滤：丢弃 `makespan ≥ 35000`（截断失败）。  
3. 取该 run 上 **makespan 最小** 的 episode；并列时优先更高 `03_success_rate`，再优先更晚 episode。  
4. 读取该 episode 结束时的 `Train/step`（累计 env step）。  
5. 对齐磁盘保存：`save_interval = 5000`（`hier.yaml`），再加训练结束时的 final save。  
   - 取距 `Train/step` **最近**的 `5000` 倍数（并列取偏小侧，避免用到「该 ep 结束之后」才写出的权重）。  
   - 例：T0 `1290374` → **`1290000`**（与协议已锁定教师一致）。  
6. 评测时显式：`HC_LOAD_STEP=<上一步>`（禁止依赖默认 latest）。

上机加载前必须确认文件存在：

```bash
ls "$HC_LOAD_DIR/nn/state_encoder_step_${HC_LOAD_STEP}.pth"
```

若缺文件，改为同目录下**不大于** `Train/step` 的最大已存 step。

## 跑满 10 个 seed（重要）

历史上 5090 单进程连跑 10 局会在 **seed 52（第 10 局）前崩溃**，W&B 只剩 9 条。  
`batch_train.sh` job 29 现默认 **`HC_EVAL_SEED_CHUNK=5`**：把 43–52 拆成两段 **进程**（`43–47` / `48–52`），中间退出 Isaac 再开下一段。

- **同一条 W&B 记录**：两段共用同一个 `HC_WANDB_RUN_ID` + 相同 `wandb_name`，第二段 `resume=allow` 续写；`MetricFullorderCore/episode` 连续为 1…10。  
- 本地 `episodes.jsonl` 也写到同一目录并 append。  
- 若坚持单进程：`HC_EVAL_SEED_CHUNK=0`（不推荐在 5090 长评测上用）。

## 5090 已有权重


| 实验 | W&B run | log 目录 | best ms @ ep | Train/step | **HC_LOAD_STEP** |
| --- | --- | --- | --- | --- | ---: |
| E0 / T0 | `zynalxhz` | `hier_2026-08-27_23-17-41` | 14600 @ 70 | 1290374 | **1290000** |
| E1 | `4zo7fjs3` | `hier_2026-09-13_15-20-57` | 15425 @ 60 | 1081850 | **1080000** |
| E2 | `b4jokwbb` | `hier_2026-09-15_18-39-11` | 15202 @ 8 | 143341 | **145000** |
| E2.5 | `zvjalw62` | `hier_2026-09-11_19-46-22` | 16029 @ 46 | 852112 | **850000** |
| E6 | `efzuah0r` | `hier_2026-09-17_10-05-15` | 16126 @ 23 | 414820 | **415000** |

路径前缀：`/home/sci/work/isaac_factory_tpa/logs/rl_games/HcFactory/`

### 5090 一条龙

```bash
cd ~/work/isaac_factory_tpa && git pull origin master
./run_2026_journal_experiments.sh eval-5090 cuda:0
```

## 工位全部 → 5090（`eval-desk`）

工位训完需同步的最优 step（只传 6 个 `*_step_STEP.pth`）：


| 实验 | log 目录 | **HC_LOAD_STEP** | 备注 |
| --- | --- | ---: | --- |
| E1.5 | `hier_2026-09-09_15-05-17` | 755000 | |
| E3 | `hier_2026-09-15_14-37-23` | 715000 | |
| E3-no-oru | `hier_2026-09-14_01-26-32` | 595000 | |
| E3.5 | `hier_2026-09-10_19-58-56` | 595000 | |
| E4 | `hier_2026-09-12_10-20-18` | 645000 | |
| E5 | `hier_2026-09-17_06-43-17` | 750000 | |
| E5-no-oru | `hier_2026-09-18_21-14-57` | 360000 | |
| E6-no-oru | `hier_2026-09-21_15-42-18` | 415000 | 若仍在训则为 **best-so-far**，训完后可重算再评 |

工位根：`/home/xue/work/isaac_factory/logs/rl_games/HcFactory/`  
5090 根：`/home/sci/work/isaac_factory_tpa/logs/rl_games/HcFactory/`

```bash
# 5090 拉代码后一条龙（顺序：E4→E5-no-oru→E5→…）
cd ~/work/isaac_factory_tpa && git pull origin master
./run_2026_journal_experiments.sh eval-desk cuda:0 --dry-run
./run_2026_journal_experiments.sh eval-desk cuda:0
```

每个实验仍是 **一条** W&B run（chunk 续写）；缺 ckpt 会停在该实验。

## E5：工位 → 5090（只传选中 step）

| 项 | 值 |
| --- | --- |
| 实验 | **E5**（`c3ces8gi`） |
| 工位目录 | `/home/xue/work/isaac_factory/logs/rl_games/HcFactory/hier_2026-09-17_06-43-17` |
| 5090 目录 | `/home/sci/work/isaac_factory_tpa/logs/rl_games/HcFactory/hier_2026-09-17_06-43-17` |
| best | makespan **15780** @ ep40，`Train/step≈748232` |
| **HC_LOAD_STEP** | **750000** |
| 文件（仅这 6 个） | `state_encoder` / `agent_A` / `agent_B` / `agent_C` / `agent_D_human` / `agent_D_robot` 的 `*_step_750000.pth`（合计约 2.6MB） |

### 1) 在工位执行 rsync（只传 step 750000）

```bash
# 在工位 xue@sci / home/xue/work/isaac_factory
REMOTE=sci@10.68.14.234          # 按你的 5090 登录改
REMOTE_REPO=/home/sci/work/isaac_factory_tpa
SRC=logs/rl_games/HcFactory/hier_2026-09-17_06-43-17/nn
STEP=750000

ssh "${REMOTE}" "mkdir -p ${REMOTE_REPO}/${SRC}"

rsync -avz --progress \
  "${SRC}/state_encoder_step_${STEP}.pth" \
  "${SRC}/agent_A_step_${STEP}.pth" \
  "${SRC}/agent_B_step_${STEP}.pth" \
  "${SRC}/agent_C_step_${STEP}.pth" \
  "${SRC}/agent_D_human_step_${STEP}.pth" \
  "${SRC}/agent_D_robot_step_${STEP}.pth" \
  "${REMOTE}:${REMOTE_REPO}/${SRC}/"
```

### 2) 5090 拉代码并评测 E5

```bash
cd ~/work/isaac_factory_tpa
git pull origin master

# 确认 6 个文件
ls -lh logs/rl_games/HcFactory/hier_2026-09-17_06-43-17/nn/*_step_750000.pth

# 预览
./run_2026_journal_experiments.sh eval-E5 cuda:0 --dry-run

# 正式（默认 chunk=5，但仍是**一条** W&B run：Hier4TPA-E5-…-eval）
./run_2026_journal_experiments.sh eval-E5 cuda:0
```

等价手动命令：

```bash
BASE=logs/rl_games/HcFactory/hier_2026-09-17_06-43-17
export HC_TEST_SEEDS=43,44,45,46,47,48,49,50,51,52 HC_TEST_TIMES=1 HC_EVAL_SEED_CHUNK=5
HC_LOAD_DIR=$BASE HC_LOAD_STEP=750000 HC_EVAL_VARIANT=E5 \
HC_WANDB_NAME=Hier4TPA-E5-N10-S42-step750000-eval \
  ./run_2026_journal_experiments.sh hier-eval-n10 cuda:0
```

## 4090 工位其余（未同步则需同样 rsync 单 step）


| 实验 | run | log 目录 | best ms @ ep | **HC_LOAD_STEP** |
| --- | --- | --- | --- | ---: |
| E1.5 | `m3nz6opg` | （见该 run `full_experiment_name`） | 15872 @ 42 | 755000 |
| E3 | `opk2sqyy` | （同上） | 16628 @ 38 | 715000 |
| E3-no-oru | `2orsgsw7` | （同上） | 16118 @ 32 | 595000 |
| E3.5 | `31rt1h7i` | （同上） | 15197 @ 33 | 595000 |
| E4 | `zgtz84wp` | （同上） | 15570 @ 35 | 645000 |
| **E5** | `c3ces8gi` | **`hier_2026-09-17_06-43-17`** | **15780 @ 40** | **750000** |
| E5-no-oru | `a2n538na` | `hier_2026-09-18_21-14-57` | 14550 @ 21 | 360000 |
| E6-no-oru | `kok6twua` | （训完后重算） | — | 待定 |

本机根路径：`/home/xue/work/isaac_factory/logs/rl_games/HcFactory/`。

## 注意

- **训练最优 ≠ 评测最优**：E2/E6 最优点偏早，latest 会差一截；正式表必须报所选 step。  
- 评测表汇总时看单条 `Hier4TPA-*-eval` run 的 10 局（chunk 续写到同一记录）。  
- 重算可用 W&B history：`Train/step` + `MetricFullorderCore/05_makespan`，再按 §规则对齐 5000。
