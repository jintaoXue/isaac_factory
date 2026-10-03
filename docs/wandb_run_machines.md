# W&B 训练 / 评测跑机对照

> 依据 `rl-driving/HcFactory_TPA` 与 `HcFactory_TPA_Eval` 各 run 的 `wandb-metadata.json`（GPU 名 + 仓库路径）整理。  
> 抓取日期：2026-09-21。hostname 多台都叫 `sci`，**不要用 hostname 区分机器**，用下表指纹。

## 机器指纹


| 简称 | GPU | 典型路径 / 环境 | 账号痕迹 |
| --- | --- | --- | --- |
| **4090 工位** | RTX 4090 | `/home/xue/work/isaac_factory`，`isaaclab` | `username=xue`；Tailscale `sci-1` / `100.92.127.35` |
| **5090** | RTX 5090 | `/home/sci/work/isaac_factory_tpa`，`env_isaaclab` | `sci`；Tailscale `sci-2` / `100.85.45.124` |
| **4090 服务器** | RTX 4090 | `/root/work/isaac_factory_master` | `root` 用户目录 |

**判定规则（优先顺序）：** GPU 型号 → `root`/`program` 路径前缀 → username。hostname 多台都叫 `sci`，不可靠。

### 5090 连接（2026-10-02 实测）

| 项 | 内容 |
| --- | --- |
| 路径 | `/home/sci/work/isaac_factory_tpa` |
| SSH（推荐） | `ssh sci@100.85.45.124` 或 `ssh sci@sci-2`（Tailscale；稳定） |
| SSH（校园网） | `ssh sci@10.68.134.197`（Wi‑Fi DHCP，会变；勿写死进脚本） |
| Tailscale | 节点名 `sci-2`；IPv4 `100.85.45.124`；账号 `thecaichicken@` |
| 网卡 | Wi‑Fi `wlo1` UP；有线 `enp4s0` DOWN（NO-CARRIER） |
| MAC(Wi‑Fi) | `bc:f1:05:d7:49:a2`（申请 DHCP 保留用） |
| MAC(有线) | `30:56:0f:25:3a:cd` |
| 网关 / DNS | `10.68.0.1` / HKU `147.8.235.74`、`147.8.235.73` |
| AnyDesk | `1932745231`（active） |
| GPU / 驱动 | RTX 5090 · `580.178.04` · 约 32 GB |
| conda | 主用 `env_isaaclab`（另有 `isaaclab`、`bn_pdformer`） |

工位本机 Tailscale：`sci-1` / `100.92.127.35`。跨机传文件优先 Tailscale SSH/rsync，避免追 DHCP。

---

## 1. Hier4TPA 主线（E 系列训练）

项目：`HcFactory_TPA`，命名原为 `Hier4TPA-*`；**新跑统一无前缀**：`{实验名}-N10-S42`。下表仍记历史 run 名。

### 4090 工位（`/home/xue/work/isaac_factory`）


| 实验 | wandb 名 | run id | 状态 | 创建日 (UTC) |
| --- | --- | --- | --- | --- |
| E1.5 | `Hier4TPA-E1.5-N10-S42` | `m3nz6opg` | finished | 2026-09-09 |
| E3.5 | `Hier4TPA-E3.5-N10-S42` | `31rt1h7i` | finished | 2026-09-10 |
| E4 | `Hier4TPA-E4-N10-S42` | `zgtz84wp` | finished | 2026-09-12 |
| E3-no-oru | `Hier4TPA-E3-no-oru-N10-S42` | `2orsgsw7` | finished | 2026-09-13 |
| E3 | `Hier4TPA-E3-N10-S42` | `opk2sqyy` | finished | 2026-09-15 |
| E5 | `Hier4TPA-E5-N10-S42` | `c3ces8gi` | finished | 2026-09-16 |
| E5-no-oru | `Hier4TPA-E5-no-oru-N10-S42` | `a2n538na` | finished | 2026-09-18 |
| E6-no-oru | `Hier4TPA-E6-no-oru-N10-S42` | `kok6twua` | running* | 2026-09-21 |

\*抓取时为 running，以 W&B 实时状态为准。

### 5090（`/home/sci/work/isaac_factory_tpa`）


| 实验 | wandb 名 | run id | 状态 | 创建日 (UTC) |
| --- | --- | --- | --- | --- |
| E2.5（短跑 ~30ep） | `Hier4TPA-E2.5-N10-S42` | `2mq6zow7` | finished | 2026-09-09 |
| E2.5（60ep 重跑） | `Hier4TPA-E2.5-N10-S42` | `zvjalw62` | finished | 2026-09-11 |
| E1 | `Hier4TPA-E1-N10-S42` | `4zo7fjs3` | finished | 2026-09-13 |
| E2 | `Hier4TPA-E2-N10-S42` | `b4jokwbb` | finished | 2026-09-15 |
| E6 | `Hier4TPA-E6-N10-S42` | `efzuah0r` | finished | 2026-09-17 |

### 4090 服务器（`/root/work/isaac_factory_master`）

**当前没有 `Hier4TPA-E*` 主线训练。** 该机上的相关 run 见 §2 / §3（旧协议 T 系列、rule 等）。

### 一览（仅 E 训练）


| 机器 | 实验 |
| --- | --- |
| 4090 工位 | E1.5, E3, E3.5, E3-no-oru, E4, E5, E5-no-oru, E6-no-oru |
| 5090 | E1, E2, E2.5×2, E6 |
| 4090 服务器 | （无 E 系列） |

入口已接但抓取时 **未见** 正式 W&B 训练 run：`E4-no-oru`。

---

## 2. 教师 / 旧协议 hard 训练（对照）


| 机器 | wandb 名 | run id | 备注 |
| --- | --- | --- | --- |
| **5090** | `hier_hard_K10_N10_T40000` | `zynalxhz` | **T0 教师源**（协议 step 1290000）；权重目录在 5090 `…/hier_2026-08-27_23-17-41` |
| **5090** | `hier_T1R_…__T1_random_ep20` | `78tdddig` | crashed |
| **5090** | `hier_hard_K10_N10_T40000` | （另一次） | crashed（2026-08-27） |
| **4090 工位** | `hier_hard_ORU_…__legacy` | `jmy3yhun` / `p469o8sx` | 旧 ORU；后者 finished |
| **4090 工位** | `hier_curriculum_K10_T40000` | 多条 | 工位 curriculum |
| **4090 服务器** | `hier_T0_…__T1_random_ep20` | — | crashed |
| **4090 服务器** | `hier_T1RH_…__T1_random_ep20` | — | crashed |

---

## 3. 基线 / 采库 / 评测（按机）

### 4090 工位

- 训练侧：`explore_N10_T40000`；多条 `hier_curriculum_*`
- 评测侧（`HcFactory_TPA_Eval`）：
  - **E0**：`Hier4TPA-E0-N10-S42-step1290000-eval`（有成功 finished）
  - `rule_K*_N16_*`、`random_K10_N10_*`、`hier_eval_N10/N16_step2500000_*` 等

### 5090

- `rule_K1_single_N10_T40000_10ep`（finished）
- 若干 eval crashed：`hier_eval_hard_N10/N16_*`、`random_K10_N16_*`

### 4090 服务器（`isaac_factory_master`）

- `rule_K10_multi_N10_T40000_10ep`（finished）
- `random_K10_N10_T40000_100ep`（crashed）
- `rule_K1_N10_T40000_5seed_x2`（crashed，Eval 项目）
- 旧 T0/T1RH 尝试（crashed，见上）

### 其他（非三台目标机）

部分 Eval 跑在 **Windows + RTX 4070**（`E:\Files\github\isaac_factory`）：rule/random 的 `10seed_x1` 等。不计入上述三台。

---

## 4. 速查：E 系列「在哪台训的」


```text
E0 eval ………… 4090 工位
E1 …………… 5090
E1.5 ………… 4090 工位
E2 …………… 5090
E2.5 ………… 5090（含短跑 + 60ep）
E3 …………… 4090 工位
E3-no-oru …… 4090 工位
E3.5 ………… 4090 工位
E4 …………… 4090 工位
E5 …………… 4090 工位
E5-no-oru …… 4090 工位
E6 …………… 5090
E6-no-oru …… 4090 工位
T0 教师 ……… 5090（zynalxhz）
```

论文横比以 **统一 eval（seeds 43–52）** 为准；训练机器不同不改变协议，但复现 ckpt 时需到对应机器的 `logs/` 或先拷权重。

---

## 5. 如何自己复查

```bash
python - <<'PY'
import wandb, json
api = wandb.Api()
r = api.run("rl-driving/HcFactory_TPA/<run_id>")
# 或按名：
# r = next(api.runs("rl-driving/HcFactory_TPA", filters={"display_name": "Hier4TPA-E4-N10-S42"}))
m = json.load(open(r.file("wandb-metadata.json").download(replace=True).name))
print(m.get("gpu"), m.get("root"), m.get("username"), m.get("host"))
PY
```
