# 两篇论文分别借鉴什么 + 代码怎么改

> 配套实现：`PDFormer/factory_bn/`  
> 详细结构见：`06.瓶颈预测模型_PDFormer与点过程适配.md`

## 1. 分别借鉴什么

### A. [PDFormer](https://ojs.aaai.org/index.php/AAAI/article/view/25556) → **稠密动态瓶颈分数预报**

对标你们的 `window_feature_table` / `bottleneck_score_s`。

| 借鉴点 | 工厂用法 |
|--------|----------|
| 动态空间自注意力 | 资源节点间依赖随时间变（拥堵/缺料传播） |
| Geo mask（短程 hop） | 工艺链 + buffer + 物流邻接 |
| Sem mask（长程相似） | 行为相似机台（相关距离 / DTW） |
| **DFT 延迟感知**（pattern → 加到 Geo Key） | 上游积压形态延迟传到下游 |
| 时间自注意力 + 多步回归 | 用过去 12 个 60s 窗预报下一窗 score |

**不借鉴**：城市日历日/周周期（仿真用逻辑时间即可）。

### B. [STGNPP](https://ojs.aaai.org/index.php/AAAI/article/view/26669) → **稀疏瓶颈事件：下一次何时开始、持续多久**

对标你们的 `bottleneck_event.csv`（`start_s`, `duration_s`）。

| 借鉴点 | 工厂用法 |
|--------|----------|
| Spatio-temporal Inquirer | 按历史事件发生的窗口索引抽取 PDFormer 隐状态 |
| Continuous GRU（flow + discrete） | 无瓶颈时缓变、事件时刻突变（BLOCKED/死锁突发） |
| 周期门控强度 \(\lambda\) | 用 episode 相位代理峰时（订单推进阶段） |
| **NLL（累积强度）+ duration MAE** | 真正点过程目标，而非只做 will 二分类 |
| 同时输出发生间隔与持续时长 | `next_tau`, `next_dur` |

**窗口级 `will_bottleneck`** 仅作辅助（数据极少时）。事件多了以后以 STGNPP NLL 为主。

## 2. 组合架构（已实现）

```text
运营特征 X (B,T,N,F)
        │
        ▼
 PDFormer Encoder（Geo/Sem/Temporal + DFT pattern）
        │
        ├──────────────► score_pred          【PDFormer 主任务】
        │
        └─ enc (B,T,N,D)
              │
              ├ Inquirer(历史事件) → Continuous GRU → h_event
              │         │
              │         └ PeriodicGatedIntensity → NLL(τ) + duration  【STGNPP 主任务】
              │
              └ 辅助 will / mark / tts（稀疏数据兜底）
```

## 3. 怎么跑

```bash
cd source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer

python -m factory_bn.export_dataset \
  --run_dir ../output/bottleneck_dataset/18_materials --window_size 60

python -m factory_bn.train \
  --config factory_bn/configs/FactoryBN.json --max_epoch 50
```

新增/改动文件：

| 文件 | 作用 |
|------|------|
| `factory_bn/stgnpp.py` | Inquirer / ContGRU / 周期门控强度 + NLL |
| `factory_bn/model.py` | 双路径 BNPDFormer |
| `factory_bn/export_dataset.py` | 导出 `event_*` 序列 |
| `factory_bn/dataset.py` | 历史事件窗 + next_tau/dur |
| `factory_bn/configs/FactoryBN.json` | `use_stgnpp`, `w_event`, `n_flow_layers` |

## 4. 数据现实

`18_materials` 事件仍很少（ep0≈2）。此时 **score 头可正常训**；STGNPP NLL 只在 `next_mask>0` 的节点上生效，事件增多后会真正主导。继续采扰动/多 episode 数据后再看 NLL / MAE-t / MAE-d。
