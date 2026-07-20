# BSTAN 与当前主模型方案对比说明

## 1. 文档目的

本文档用于整理当前瓶颈预测模型路线选择的讨论结果，重点回答：

- 为什么 BSTAN 适合作为制造瓶颈预测 baseline；
- 为什么当前项目不建议直接以原始 BSTAN 作为最终主模型；
- 当前主方案相比 BSTAN 增强了哪些能力；
- 多智能体调度在预测模型中的意义是什么；
- 为什么最终选择 `PDFormer-inspired Dynamic Graph Transformer + Multi-task Event Head` 作为主模型路线。

## 2. 项目任务背景

当前项目面向水喉制造工艺的瓶颈预测。当前代码中已有 Isaac Sim / Isaac Lab 工厂仿真环境，并通过 A/B/C/D 四层调度 agent 推进生产过程：

```text
Agent A：产品排序 / 产品释放
Agent B：选择当前推进的产品
Agent C：选择下一道工艺或物流任务
Agent D：分配 human / robot / machine 等资源
```

水喉工艺路线为：

```text
钢管
  -> 切管下料
  -> 管口压槽
  -> 配料点焊
  -> 氩弧焊底焊
  -> MIG焊面焊
  -> 涂漆防锈
  -> 成品
```

预测目标不是只判断当前哪里堵，而是预测未来一段时间内的瓶颈事件：

```text
will_bottleneck
future_bottleneck_object_id
future_bottleneck_type
time_to_start
duration
severity
```

## 3. BSTAN 的思路

BSTAN 是制造瓶颈预测领域直接相关的方法。其核心思想是：

```text
将生产系统表示为 station / buffer / material flow 图；
使用 GAT 建模 station 之间的空间关系；
使用 GRU 建模状态随时间的变化；
预测未来 blockage / starvation；
再识别未来 bottleneck station。
```

如果迁移到当前水喉工艺，BSTAN-style 图可以表示为：

```text
切管机
  -> 切管后 buffer
  -> 压槽机
  -> 压槽后 buffer
  -> 点焊台
  -> 点焊后 buffer
  -> 氩弧焊机器人
  -> 氩弧焊后 buffer
  -> MIG焊机
  -> MIG后 buffer
  -> 涂漆工位
```

节点特征可以来自窗口聚合表：

```text
queue_length
avg_waiting_time
occupancy_ratio
active_pct
blocked_ratio
starved_ratio
output_rate
```

因此，BSTAN 非常适合作为当前项目的制造图 baseline。

## 4. BSTAN 的价值

BSTAN 的价值主要体现在：

```text
1. 与制造瓶颈预测问题高度同源；
2. station / buffer / material flow 图与水喉工艺天然匹配；
3. blockage / starvation 与当前瓶颈维度调研一致；
4. GAT + GRU 结构实现复杂度相对可控；
5. 能作为证明“图结构有用”的强 baseline。
```

在实验中，BSTAN-style baseline 可以回答：

```text
相比 XGBoost / LSTM / TCN，只加入静态制造图是否能提升瓶颈预测效果？
```

这对论文实验非常重要。

## 5. 原始 BSTAN 的局限

原始 BSTAN 不完全适合作为当前项目最终主模型，主要有以下限制。

### 5.1 图结构偏静态

BSTAN 更擅长建模固定工艺拓扑：

```text
切管 -> 压槽 -> 点焊 -> 氩弧焊 -> MIG -> 涂漆
```

但当前仿真中，很多瓶颈来自动态调度关系：

```text
human_0 当前被分配给 MIG 物流；
robot_1 当前被分配给点焊前物流；
gantry_0 当前被多个物流任务竞争；
某个产品被 Agent B/C 连续推向同一工位。
```

这些关系不是固定工艺图能完整表达的。

### 5.2 输出偏瓶颈位置

BSTAN 更自然的任务是预测：

```text
future bottleneck location
```

而当前项目希望预测完整事件：

```text
是否发生
位置
类型
开始时间
持续时间
严重程度
```

因此，需要 multi-task event head。

### 5.3 对长距离和多资源耦合表达有限

水喉工艺中的瓶颈可能不是相邻工序直接导致的。例如：

```text
human 长时间被点焊占用
  -> 涂漆工位未来等待人工
  -> 成品输出下降
```

或者：

```text
gantry / robot 被 MIG 物流连续占用
  -> 多个前后工位等待运输
  -> transport bottleneck 扩散
```

这类依赖跨越多个资源、多个时间窗口，不只是相邻 station 的局部传播。

### 5.4 未显式建模传播延迟

制造瓶颈具有明显时间延迟：

```text
某工位处理变慢
  -> 若干秒后前置 buffer 堆积
  -> 再若干秒后上游工位 blocked
  -> 再若干秒后下游工位 starved
  -> 最后系统 throughput 下降
```

BSTAN 的 GRU 可以隐式学习部分时间关系，但没有显式建模这种传播延迟。

## 6. 当前主模型方案

当前推荐主方案为：

```text
PDFormer-inspired Dynamic Graph Transformer
+ Delay-aware attention
+ Multi-task Bottleneck Event Head
```

更完整地说：

```text
Dispatch-aware Delay-aware Dynamic Graph Transformer
for Manufacturing Bottleneck Event Prediction
```

整体结构：

```text
Graph sequence G_{t-T:t}
  -> Node / Edge / Global Feature Encoder
  -> Prior-guided Dynamic Graph Construction
  -> Delay-aware Dynamic Spatio-Temporal Transformer Encoder
  -> Multi-task Event Head
  -> Future bottleneck event prediction
```

## 7. 多源图结构定义

当前主模型不只使用一张静态工艺图，而是融合多类关系：

```text
A_final(t) =
    alpha * A_process
  + beta  * A_transport(t)
  + gamma * A_material(t)
  + delta * A_agent(t)
  + eta   * A_learned(t)
```

### 7.1 A_process

`A_process` 表示固定工艺流程关系。

水喉工艺中：

```text
切管 -> 压槽 -> 点焊 -> 氩弧焊 -> MIG -> 涂漆
```

这是 BSTAN 最主要使用的图关系。

### 7.2 A_transport(t)

`A_transport(t)` 表示当前窗口内实际发生的物流关系。

例如：

```text
product_03 正从氩弧焊后 buffer 运往 MIG焊机；
robot_1 / gantry_0 正在执行 logistic_for_MIG_welding_surface。
```

则当前窗口形成：

```text
氩弧焊后 buffer -> robot_1 / gantry_0 -> MIG焊机
```

该图随时间变化。

### 7.3 A_material(t)

`A_material(t)` 表示物料供应关系。

例如点焊需要：

```text
钢管
法兰
弯头
```

则形成：

```text
pipe material -> 点焊台
flange material -> 点焊台
elbow material -> 点焊台
```

如果某类物料短缺，该边的状态或边特征会发生变化。

### 7.4 A_agent(t)

`A_agent(t)` 是体现多智能体调度的关键。

它表示当前窗口内由调度决策形成的临时资源绑定关系。

例如：

```text
Agent B 选择 product_03；
Agent C 选择 logistic_for_MIG_welding_surface；
Agent D 分配 human_0 和 robot_1。
```

则形成：

```text
product_03 -> logistic_for_MIG_welding_surface
logistic_for_MIG_welding_surface -> human_0
logistic_for_MIG_welding_surface -> robot_1
logistic_for_MIG_welding_surface -> MIG焊机
```

这不是“智能体主观意图”，而是现实工厂中也存在的：

```text
任务派发历史
资源分配历史
物流执行历史
工件路由历史
```

### 7.5 A_learned(t)

`A_learned(t)` 表示模型从数据中学习出的隐式依赖。

例如模型可能发现：

```text
MIG 焊机持续高负载
与
涂漆工位未来 starved
之间存在稳定关系。
```

即使人工定义的图没有完全描述这条关系，模型也可以通过 learned graph 捕捉。

## 8. BSTAN、动态图 BSTAN 和当前主模型的关系

三者不是互斥关系，而是逐步增强关系。

### 8.1 BSTAN-style baseline

```text
输入图：
  A_process
  A_material / buffer flow

模型：
  GAT + GRU

输出：
  bottleneck_node
```

作用：

```text
验证静态制造图是否对瓶颈预测有帮助。
```

### 8.2 Dynamic BSTAN

```text
输入图：
  A_process
  A_material
  A_transport(t)
  A_agent(t)

模型：
  GAT + GRU

输出：
  bottleneck_node
  bottleneck_type
  或 multi-task event labels
```

作用：

```text
验证加入动态物流和调度关系是否提升预测效果。
```

### 8.3 当前主模型

```text
输入图：
  A_process
  A_transport(t)
  A_material(t)
  A_agent(t)
  A_learned(t)

模型：
  Dynamic Graph Transformer
  Delay-aware attention
  Multi-task Event Head

输出：
  will_bottleneck
  future_bottleneck_object_id
  future_bottleneck_type
  time_to_start
  duration
  severity
```

作用：

```text
建模多资源、多时间窗口、长距离依赖和延迟传播，并预测完整瓶颈事件。
```

## 9. 为什么不是只用 BSTAN + Multi-head

BSTAN 也可以接 multi-head：

```text
BSTAN encoder
  -> will_bottleneck head
  -> location head
  -> type head
  -> time_to_start head
  -> duration head
  -> severity head
```

因此，multi-head event prediction 并不是 Transformer 独有能力。

但是，BSTAN 的 encoder 仍然更偏：

```text
邻居节点聚合
时间递推记忆
```

而当前水喉制造系统中的瓶颈常常来自：

```text
多工序并行推进
human / robot / gantry 共享资源
物流任务竞争
物料供应变化
agent 连续调度造成的资源绑定趋势
跨多个窗口的延迟传播
```

这些更适合用 Dynamic Graph Transformer 建模，因为 Transformer 更擅长：

```text
长距离依赖
跨时间窗口依赖
多节点之间的全局注意力
动态关系融合
```

## 10. 多智能体调度的意义

在当前项目中，多智能体调度首先是数据生成机制，但它不只是数据生成器。

真实工厂中同样存在：

```text
MES 派工
APS 排产
班组长安排
AGV 调度系统
人工资源分配
设备任务队列
```

仿真中的 A/B/C/D agent 用来模拟这些调度决策。

因此，模型不应学习“agent 怎么想”，而应学习：

```text
已经发生的生产决策和资源分配记录如何影响未来瓶颈。
```

这些记录包括：

```text
哪个产品被推进
哪个任务被派发
哪个 human 被占用
哪个 robot / gantry 被占用
哪个 machine 被绑定
哪些物流任务正在执行
哪些任务在等待资源
```

这些信息能提前暴露未来瓶颈趋势。

例如：

```text
过去 4 个窗口中，多个产品连续被推进到 MIG 前；
human_0 和 robot_1 持续服务 MIG 相关物流；
gantry_0 长时间被 MIG 区域占用；
MIG 前 buffer 开始上升；
涂漆工位开始出现 starved。
```

这说明未来可能发生：

```text
MIG / transport 复合瓶颈
或 human 资源瓶颈
```

这就是 `A_agent(t)` 和 `A_transport(t)` 作为预测变量的意义。

## 11. 当前主模型的创新点

当前主模型的创新点不应表述为“直接套用 Transformer”，而应表述为：

```text
面向多智能体制造仿真瓶颈预测的动态图构建、事件标签设计和延迟感知模型适配。
```

具体包括：

### 11.1 任务建模创新

从传统的：

```text
bottleneck location prediction
```

扩展为：

```text
bottleneck event prediction
```

即同时预测：

```text
是否发生
位置
类型
开始时间
持续时间
严重程度
```

### 11.2 图结构创新

从静态 station graph 扩展为：

```text
process graph
transport graph
material graph
agent dispatch graph
learned graph
```

### 11.3 多智能体调度适配

把调度系统的历史决策转化为动态图关系：

```text
dispatching decisions
resource assignment records
task allocation history
transport assignment history
```

### 11.4 延迟传播建模

把 PDFormer 中的 propagation delay 思想迁移到制造场景：

```text
工艺加工时间
buffer 等待时间
物流搬运时间
人工操作时间
物料补给延迟
```

瓶颈影响不是瞬时传播，而是在多个窗口内逐步扩散。

## 12. 推荐实验路线

建议实验按以下顺序推进：

```text
1. 非图 baseline
   XGBoost / LSTM / TCN

2. BSTAN-style static manufacturing graph baseline
   A_process + A_material
   GAT + GRU

3. Dynamic BSTAN
   A_process + A_material + A_transport(t) + A_agent(t)
   GAT + GRU + optional multi-head

4. 当前主模型
   A_process + A_transport(t) + A_material(t) + A_agent(t) + A_learned(t)
   Dynamic Graph Transformer + Delay-aware attention + Multi-task Event Head
```

这样设计实验可以逐步回答：

```text
图结构是否有用？
动态物流和调度边是否有用？
Transformer 是否优于 GAT-GRU？
delay-aware 模块是否有用？
multi-task event head 是否优于只预测位置？
```

## 13. 推荐默认时间尺度

第一版建议：

```text
window_size = 30s
stride = 30s
input_length T = 120s
prediction_horizon H = 120s
min_bottleneck_duration = 60s
```

含义：

```text
每 30 秒生成一个制造系统图快照；
模型看过去 4 个窗口；
预测未来 4 个窗口内的瓶颈事件。
```

该设置能较好地体现：

```text
动态调度趋势
长距离资源依赖
瓶颈传播延迟
```

## 14. 结论

BSTAN 应作为当前项目的重要制造图 baseline，因为它与生产系统瓶颈预测问题高度同源，能够验证静态制造图对瓶颈预测的价值。

但最终主模型更适合采用：

```text
PDFormer-inspired Dynamic Graph Transformer
+ Delay-aware attention
+ Multi-task Bottleneck Event Head
```

原因是当前项目的瓶颈来源不仅包括固定工艺流程，还包括：

```text
多产品并行
human / robot / gantry / machine 共享资源
物流任务竞争
物料供应变化
调度决策形成的动态资源绑定
跨时间窗口的延迟传播
```

因此，当前主方案相比 BSTAN 更能体现多智能体制造仿真场景下的动态瓶颈预测特征。
