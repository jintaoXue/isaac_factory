# Hier4TPA 算法可继续优化点

> 对照实现：`source/algo/hierarchical/hc_factory/`  
> 论文叙事：domain-structured HRL（A→B→C→D + information pool），非自动 option discovery。  
> 参考：Pateria et al., *ACM Comput. Surv.* 2021（HRL survey）；本仓库 `docs/experiment_protocol.md`（T0–T4 / ORU）。

## 0. 现状一句话

| 模块 | 现状 |
|------|------|
| 决策栈 | A/B/C/D + intra-step information pool + `dispatch_list`（K 并行） |
| 后端 | Masked DQN（CTCE）；各层共享 Transformer obs encoder |
| 长时域 | Catalog / progress key；ORU / curriculum 部分落地或推进中（见 protocol） |
| 人因 | Fatigue monitor（旧作延续） |

---

## 1. 层级与信用分配（HRL 向）

| ID | 方向 | 说明 | 优先级 |
|----|------|------|--------|
| H1 | **层间信用分配** | 全局 makespan 稀疏时，A/B 难学；可试分层奖励、延迟回报回传、或只在有效决策步更新 A/B | 高 |
| H2 | **B 层可学习打分** | 现状常偏 FIFO/规则；改为 RL 学 priority score，与 C/D 联合或交替训 | 高 |
| H3 | **自下而上训程** | 先稳 C/D，再开 A/B（survey：bottom-up vs end-to-end） | 中 |
| H4 | **层内终止/持续** | 显式「本步是否继续服务下一 in-process job」vs 固定扫完 K | 低 |
| H5 | **勿硬套 option discovery** | 四层来自生产语义；论文写 handcrafted hierarchy；发现类方法仅作 future | 叙事 |

## 2. 值函数 / 策略后端

| ID | 方向 | 说明 | 优先级 |
|----|------|------|--------|
| R1 | **Rainbow 组件** | Double DQN、prioritized replay、dueling、（可选）noisy / distributional | 高 |
| R2 | **Prioritized replay** | 按 TD-error 或「稀有 progress key / 近完成」加权 | 高 |
| R3 | **PPO / SAC / TD3** | 离散 mask 主线仍偏 DQN；连续头（速度、路径参数）再考虑 actor–critic | 低–中 |
| R4 | **Safe / constrained** | 延续 JMS：疲劳约束、成本函数 vs mask 硬约束 | 中 |
| R5 | **目标条件化** | universal / goal-conditioned 头：条件于 `n_finished`、剩余订单、stage | 中 |

## 3. 长时域与数据（与 ORU / curriculum 对齐）

| ID | 方向 | 说明 | 优先级 |
|----|------|------|--------|
| L1 | **ORU 完整闭环** | explore 存 transition → 训练期 offline update；T1/T2 与 protocol 一致 | 最高 |
| L2 | **Progress-key 去冗** | 硬去重 → soft cosine / 桶内多样性，避免 catalog 同质化 | 高 |
| L3 | **Reverse curriculum** | 固定 target=10，start_nfin 8→0；与 warm-up 顺序写清 | 高 |
| L4 | **HER 式 relabel** | 失败轨迹按「已达到的 nfin / key」重标（稀疏回报） | 中 |
| L5 | **Policy-guided catalog** | T3/T4：采库用当前策略而非纯 random | 中 |
| L6 | **Stagnation 与 ORU 解耦** | 仿真 reset 仅作工程；论文贡献强调 offline update | 叙事 |

## 4. 信息池与并行派工

| ID | 方向 | 说明 | 优先级 |
|----|------|------|--------|
| P1 | **Ledger 可微 / 可学习** | 占用回写规则固定；可试可学习占用偏好或冲突惩罚 | 低 |
| P2 | **动态 K** | `max_parallel_cd_dispatch` 随 WIP / 空闲资源自适应 | 中 |
| P3 | **A 与 WIP cap 联动** | 准入与 CONWIP 式上限联合约束（生产控制接口） | 中 |
| P4 | **冲突诊断指标** | 同一步 not_joined / mask 拒绝率进 wandb（已有 MetricCatalog 可扩） | 低 |

## 5. 表征与网络

| ID | 方向 | 说明 | 优先级 |
|----|------|------|--------|
| N1 | **共享 vs 分塔 encoder** | 消融：共享 Transformer vs A–D 独立 | 实验 |
| N2 | **实体相对位置 / 图** | human–robot–machine 关系用 GNN 或 relation bias | 中 |
| N3 | **条件化于疲劳/技能** | obs 或 adapter 注入人因状态 | 中 |
| N4 | **容量扫描** | 深度/宽度 vs 长 horizon 过拟合 | 实验 |

## 6. 多智能体与部署

| ID | 方向 | 说明 | 优先级 |
|----|------|------|--------|
| M1 | **CTCE → 部分分散** | 部署时 D 层本地执行、A/B 仍集中 | 中长期 |
| M2 | **通信 / 对手建模** | survey 的 MA-HRL；仅当 CTCE 不够用再开 | 低 |
| M3 | **Sim-to-real / 海创** | 感知噪声、人因偏差、延迟 | 中长期 |

## 7. 建议落地顺序（工程）

1. **L1 ORU 闭环** + MetricCatalog 对齐论文 T0–T2  
2. **R1/R2** Double DQN + prioritized replay（改动面小）  
3. **H2 B-score RL** + **H1** 稀疏回报下的 A/B 更新频率  
4. **L3 curriculum** 与 hard train 消融表  
5. 网络消融 **N1**、并行 **P2/P3**  

---

## 8. 与论文写作的边界

- **可写进 Method 的**：四层 + pool、masked DQN、ORU、curriculum（实现到哪写到哪）。  
- **适合 Ablation / Future**：Rainbow 全套、HER、自动 subtask discovery、分散执行。  
- **Related work 引用**：Pateria survey 支撑「长 horizon → 层级抽象」；明确我们是 **handcrafted production hierarchy**，不是 option discovery。
