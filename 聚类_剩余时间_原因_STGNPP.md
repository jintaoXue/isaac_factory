# 聚类 / 剩余时间 / 原因 / STGNPP

> 日期：2026-09-06  
> 对照 ckpt：`dense_i1_a1_prefix8`（正式保留，Test report F1=0.854）  
> 主任务数字仍以 [`F1提升实验历程.md`](./F1提升实验历程.md) 为准。本文只回答三件附录事。

---

## 1. 聚类有没有用？

**笔记第 2 节那套两层 k-means，当前基本没用。** 真正可能沾一点边的，是另一套「规则簇嵌入 + 对比学习」。消融打的是后者，不是 k-means。

### 1.1 仓库里其实有三种「聚类」

| 东西 | 是什么 | `prefix8` 里开没开 | 对 A.1 的作用 |
| --- | --- | --- | --- |
| 离线 **k-means**（工位 `cluster_id` / 整线 `window_cluster`） | 笔记 §2，npz 里存着 | **没进损失、没进解码** | 无。`w_cluster=0`，对比学习也不吃 `window_cluster` |
| **规则簇** `seed_cluster_ids` → `cluster_emb` | 按缺料 / 延误 / 排队等规则把工位分成 7 态，加到事件头 | 开（嵌入从 0 训起来） | 可能有一点；`recall_lift_threshold=0`，解码不靠它抬分 |
| **对比学习** `w_contrast=0.05` | 按「这窗堵不堵 × 扰动维 × 工位类型」拉表示 | 开 | 可能有一点；**不是** k-means |
| PDFormer `n_cluster=16` pattern keys | 原论文 delay-aware 图案 | 架构自带 | 不是你加的工况聚类 |
| 无监督 `cluster_head` | 预报未来窗的簇 id | **关**（`w_cluster=0`） | 日志 `cluster_acc≈0.006` 是没训的头，不当成果 |

所以：k-means 算过、存过，模型没拿它当监督。要「去掉聚类」做消融，只能关 **规则簇融合 + 对比学习**，否则两边数字会一样。

### 1.2 消融怎么跑

- **有聚类（现行）**：`prefix8`，`fuse_hist_cluster=true`，`w_contrast=0.05`
- **无聚类**：`factory_bn/configs/FactoryBN_dense_nocluster.json`（`fuse_hist_cluster=false`，`w_contrast=0`），从 prefix8 微调 8 epoch

```bash
python -m factory_bn.train \
  --config factory_bn/configs/FactoryBN_dense_nocluster.json \
  --data_dir raw_data/dense_i1 \
  --save_dir libcity/cache/model_cache/dense_i1_a1_prefix8_nocluster \
  --device cuda --no_wandb
```

已跑完（epoch 7）。Test **P=0.842 / R=0.862 / F1=0.852**（`up_r=0.500`），相对 prefix8 的 0.852 / 0.856 / **0.854** 只差 **0.002**。规则簇 + 对比学习对主分数可有可无。

---

## 2. 另外两个输出现在怎么样？

截图上除了谁/何时/多久，还有 **订单剩余时间** 和 **过程原因**。都不抢 `report_f1`。

### 2.1 订单剩余时间（`remain_len`）

| | Val | Test |
| --- | ---: | ---: |
| `remain_len_mae` | ≈11.6–12.8 分钟 | **10.5 分钟** |
| 损失权重 | `w_remain_len=0.35`（小） | |

含义：还要多少个 60 秒窗，整局 `jobs_remaining` 才到 0。一局约 2 小时、10 根水管，MAE 10 分钟是「能用、不算准」。头已经看到 `jobs/total`，不是瞎猜，但也没认真拟合（权重只有事件头的约 1/7）。

已跑完 remain 臂：`FactoryBN_dense_prefix8_remain.json`，从 prefix8 微调 8 epoch（best=7），`w_remain_len=2.0`，`remain_len = head + rate × jobs`。

| | prefix8 | remain 臂 |
| --- | ---: | ---: |
| Test `remain_len_mae` | 10.5 | **9.9** |
| Val `remain_len_mae` | ≈12.0 | 11.3 |
| Test report P / R / F1 | **0.852 / 0.856 / 0.854** | 0.843 / 0.844 / 0.843 |
| Test `up_r` / `on_r` | 0.455 / **0.911** | **0.482** / 0.893 |

剩余时间只好了 **0.6 分钟**，主 F1 掉 0.011。不换正式模型。再抠 remain 不如改任务（按工位/在制品），不要再加 `w_remain_len`。

### 2.2 过程原因（六类）× 原因簇消融

规则簇 `seed_cluster_ids` 聚的就是过程原因态。要看「聚类有没有帮上原因」，对照是 **prefix8（融簇）vs nocluster（不融簇、无对比）**，标签、划分、初始化都一样。

Test（有效样本 `cause_n=579`，多数类瞎猜 0.41；`blocked_downstream` / `high_utilization` 两边都无 support）：

| 类 | prefix8（有簇） | nocluster（无簇） |
| --- | ---: | ---: |
| 到货延误 | 0.956 | **0.971** |
| 缺料 | 0.988 | 0.988 |
| 上游饿 | 0.949 | 0.949 |
| 排队 | 0.992 | **1.000** |
| overall acc | 0.965 | **0.971** |
| macro recall | 0.971 | **0.977** |
| Val macro | 0.774 | **0.778** |
| Test report F1 | **0.854** | 0.852 |

**融原因簇没有提高过程原因准确率**，nocluster 还略高。原因头吃的是整窗 X（缺料、延误、排队等已经在特征里），`cluster_emb` 只是同一套规则的离散版，加进去是重复信息。高 acc 是在复述规则，不是「会解释瓶颈」。

Val 的 macro 只有约 **0.77**，test 0.97 是划分碰巧简单，不要写成「原因已经完美」。

**有提升空间，但是换问题，不是再堆 acc：**

1. **按工位出原因**（现在是整窗一个 `cause`），对齐「这台为什么堵」。
2. 补 `blocked_downstream` / `high_utilization` 样本，否则这两类不能下结论。
3. 若要「扰动类型」而不是过程启发式，标签得改，不能继续用现在这 6 类规则。

---

## 3. STGNPP 怎么用？会不会比现在好？

现行 **关着**：`use_stgnpp=false`，`w_will=w_mark=w_tts=w_event=0`。日志里的 `will_f1=0`、`nll=0` 是这个关着的 180s 辅头，不是 15 分钟 `will_block`。

STGNPP（Jin et al., AAAI'23）吃的是 **历史上已经发生的事件序列**（时刻、时长），用强度函数预报 **下一次** 何时发生、持续多久。它擅长「下一次还要等几分钟」，不擅长「38 个工位里谁会堵」。

### 3.1 应该怎么挂，才不毁掉 A.1

不要替换 `will / start / dur / prefix`。只当辅头：

1. `use_stgnpp=true`，`w_tts` / `w_event` 先设 **0.1–0.3**（比主事件头小一个数量级）。
2. 历史事件用占用段或 TPM，不要用 L2 扰动名。
3. 解码：冷工位把强度预报的 `tau` 当成 `start_min` 的先验（只在 `start≤2` 的近地平线用）；`will` 仍由事件头决定报不报。
4. 进行中继续走 prefix（`start=0` + 剩余），不要让点过程抢。
5. val 仍用 `report_f1`、P≥0.80 存盘。NLL 下降不算赢。

### 3.2 会不会比 prefix8 更好？

**主 F1 大概率不会明显更好，有可能略差。** prefix8 的 0.85 已经主要靠进行中；STGNPP 帮的是尚未发生的「何时」。test `up_r` 只有 0.455，这里理论上有空间，但：

- 很多即将发生的堵在 30 分钟历史上 **没有事件可询**，点过程退回空历史；
- 以前分数回归、前兆头都没把 `up_r` 换出更高的总 F1；
- NLL 和 A.1 抢梯度，和监督 `bottleneck_score` 那次一样。

已跑完：`FactoryBN_dense_prefix8_stgnpp.json`，从 prefix8 微调 8 epoch（best=6）。无监督损失已接上 NLL。冷工位用 `tau` 当 `start≤2` 先验。

| | prefix8 | STGNPP |
| --- | ---: | ---: |
| Val P / R / F1 | **0.883 / 0.881 / 0.882** | 0.863 / 0.899 / 0.881 |
| Test P / R / F1 | **0.852 / 0.856 / 0.854** | 0.862 / 0.837 / 0.849 |
| Test `up_r` / `on_r` | **0.455 / 0.911** | 0.455 / 0.889 |
| Test `remain_len_mae` | 10.5 | 10.2 |
| Test NLL | 0（关着） | 5.66（在训，没换成 F1） |

`up_r` **完全没动**（仍 0.455），进行中召回掉 0.022，总 F1 掉 0.005。和 `uphist` 一样 **不采用**。点过程 NLL 在训，但没有帮到 15 分钟事件匹配。

---

## 4. 产物

| 文件 | 含义 |
| --- | --- |
| `dense_i1_a1_prefix8_nocluster/last_metrics.json` | 无规则簇 / 无对比；原因略好、A.1 几乎一样 |
| `dense_i1_a1_prefix8_sup/last_metrics.json` | prefix8 契约上的监督臂 |
| `dense_i1_a1_prefix8_remain/last_metrics.json` | 剩余时间加速率残差 |
| `dense_i1_a1_prefix8_stgnpp/last_metrics.json` | STGNPP 辅头 + start 先验 |
| [`无监督vs监督bottleneck_score对照.md`](./无监督vs监督bottleneck_score对照.md) §7 | 监督对照数字 |

正式模型仍是 **prefix8**，这四组都不替换。
