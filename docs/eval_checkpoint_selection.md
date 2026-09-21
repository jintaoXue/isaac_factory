# 评测 Checkpoint 选择（训练最优，非 latest）

> 更新：2026-09-21。与 T0 教师选点同一精神：**训练曲线 makespan 最低点 → 最近 `save_interval` 保存步**。  
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

## 5090 可评（本地有权重）


| 实验 | W&B run | log 目录 | best ms @ ep | Train/step | **HC_LOAD_STEP** | vs last ms |
| --- | --- | --- | --- | --- | ---: | --- |
| E0 / T0 | `zynalxhz` | `hier_2026-08-27_23-17-41` | 14600 @ 70 | 1290374 | **1290000** | last 15558 |
| E1 | `4zo7fjs3` | `hier_2026-09-13_15-20-57` | 15425 @ 60 | 1081850 | **1080000** | = final（最优在末） |
| E2 | `b4jokwbb` | `hier_2026-09-15_18-39-11` | 15202 @ 8 | 143341 | **145000** | last 17280 |
| E2.5 | `zvjalw62` | `hier_2026-09-11_19-46-22` | 16029 @ 46 | 852112 | **850000** | last 18116 |
| E6 | `efzuah0r` | `hier_2026-09-17_10-05-15` | 16126 @ 23 | 414820 | **415000** | last 20815 |

路径前缀：`/home/sci/work/isaac_factory_tpa/logs/rl_games/HcFactory/`  
（E2.5 短跑 `2mq6zow7` 不参与横比。）

### 5090 命令（N=10，seeds 43–52）

```bash
cd ~/work/isaac_factory_tpa && git pull
BASE=logs/rl_games/HcFactory
export HC_TEST_SEEDS=43,44,45,46,47,48,49,50,51,52 HC_TEST_TIMES=1

# E0（入口已写死 1290000）
./run_2026_journal_experiments.sh E0 cuda:0

HC_LOAD_DIR=$BASE/hier_2026-09-13_15-20-57 HC_LOAD_STEP=1080000 HC_EVAL_VARIANT=E1 \
  ./run_2026_journal_experiments.sh hier-eval-n10 cuda:0

HC_LOAD_DIR=$BASE/hier_2026-09-15_18-39-11 HC_LOAD_STEP=145000 HC_EVAL_VARIANT=E2 \
  ./run_2026_journal_experiments.sh hier-eval-n10 cuda:0

HC_LOAD_DIR=$BASE/hier_2026-09-11_19-46-22 HC_LOAD_STEP=850000 HC_EVAL_VARIANT=E2.5 \
  ./run_2026_journal_experiments.sh hier-eval-n10 cuda:0

HC_LOAD_DIR=$BASE/hier_2026-09-17_10-05-15 HC_LOAD_STEP=415000 HC_EVAL_VARIANT=E6 \
  ./run_2026_journal_experiments.sh hier-eval-n10 cuda:0
```

## 4090 工位（拷权重或本机评）


| 实验 | run | best ms @ ep | **HC_LOAD_STEP** |
| --- | --- | --- | ---: |
| E1.5 | `m3nz6opg` | 15872 @ 42 | 755000 |
| E3 | `opk2sqyy` | 16628 @ 38 | 715000 |
| E3-no-oru | `2orsgsw7` | 16118 @ 32 | 595000 |
| E3.5 | `31rt1h7i` | 15197 @ 33 | 595000 |
| E4 | `zgtz84wp` | 15570 @ 35 | 645000 |
| E5 | `c3ces8gi` | 15780 @ 40 | 750000 |
| E5-no-oru | `a2n538na` | 14550 @ 21 | 360000 |
| E6-no-oru | `kok6twua` | （抓取时几乎未训完） | 待跑满后重算 |

log 目录从各 run 的 `full_experiment_name` 读取（本机 `/home/xue/work/isaac_factory/logs/...`）。

## 注意

- **训练最优 ≠ 评测最优**：E2/E6 最优点偏早，latest 会差一截；正式表必须报所选 step。  
- 早期单点最优方差大；若审稿质疑，可加稳健规则（如「后半程最优」）作附录，主表仍用本文件规则并保持全方法一致。  
- 重算可用 W&B history：`Train/step` + `MetricFullorderCore/05_makespan`，再按 §规则对齐 5000。
