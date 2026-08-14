1. statedic在我当前的模型中是如何编码的？（原始数据表都是异质数据）

> 结论先说：**没有单独的 state Embedding / hetero 边类型消息传递。**
> 异质表里的离散状态 → 统一成 `norm_state` 字符串 → 按时间窗聚合成连续数值特征 → 拼上资源类型 one-hot → 进模型用 `Linear(F→64)` 编码。

---

### 总流程（对照代码）

```text
仿真 raw state（machine/human/... 各说各的）
        │  bottleneck_data.py :: map_*_state
        ▼
resource_event_log.to_state = IDLE/PROCESSING/BLOCKED/...
        │  bn_agg :: build_timelines + 窗口求交
        ▼
window_feature_table 一行 = 13 维运营连续特征
        │  export_dataset.py :: FEATURE_COLS + type_onehot
        ▼
X[t, n, :] ∈ R^{18}   （13 ops + 5 type）
        │  dataset.py z-score
        ▼
DataEmbedding: Linear(18→64) + 时间PE + LapPE
```

---

### Step A：异质 raw state → 统一 `norm_state`

文件：`src/bottleneck_data.py`

机台/龙门架用一套字典式映射：

```python
# map_machine_state
"free"              → IDLE
"invalid"           → STOP
"working_*"         → PROCESSING
"waiting_*"         → BLOCKED
"materialReadyFor_*"→ WAITING
```

人/机器人另有一套（还看 subtask）：

```python
# map_human_robot_state
"free"              → IDLE
working + wait      → STARVED
working + walk/操作 → PROCESSING
```

写入 `resource_event_log` 的是 **`to_state = norm_state`**，不是原始字符串。  
这就是所谓「state dic」：**按资源类型选不同映射表，落到同一套状态词表**。

---

### Step B：离散状态 → 连续时间轴 → 窗口统计（不是 one-hot）

文件：`tools/bn_agg/`

1. `build_timelines`：事件 `to_state` 展开成区间 `[t_i, t_{i+1}) → state`
2. 窗口内与状态集合求交时长：

```python
ACTIVE_STATES  = {"PROCESSING"}           → active_pct_s, current_active_duration_s
BLOCKED_STATES = {"BLOCKED"}              → blocked_time_s
STARVED_STATES = {"STARVED", "WAITING"}   → starved_time_s
```

3. 其它异质表同步聚到**同一行**：
   - `job_trace` → `queue_length_s`, `avg_waiting_time_s`
   - `buffer_event_log` → `occupancy_ratio_s`, `queue_growth_rate_s`
   - `route_transport_task` → `route_delay_s`
   - `material_inventory_log` → `material_shortage_propagation_s`

输出：`window_feature_table.csv`，每资源 × 每窗口一行。  
**此时状态已经不是类别 id，而是 float 占比/时长。**

---

### Step C：拼成模型输入向量 X

文件：`PDFormer/factory_bn/export_dataset.py` + `graph.py`

```python
FEATURE_COLS = [  # 13 维
    "queue_length_s", "avg_waiting_time_s", "occupancy_ratio_s",
    "queue_growth_rate_s", "active_pct_s", "current_active_duration_s",
    "blocked_time_s", "starved_time_s", "inter_departure_var_s",
    "upstream_blocked_ratio_s", "downstream_starved_ratio_s",
    "route_delay_s", "material_shortage_propagation_s",
]

RESOURCE_TYPES = ("machine", "gantry", "human", "transport_robot", "buffer")  # 5 维 one-hot

feat = [row[c] for c in FEATURE_COLS] + type_onehot(resource_type)
# → features[t, n, :] shape = (T, N, 18)
```

- 异质资源共用**同一套 13 维 schema**（没有的表就填 0，如 machine 的 occupancy 常为 0）
- 类型差异靠 **5 维 one-hot** 告诉模型「这是机台还是人」
- **不含** `bottleneck_score_s`（标签，防泄漏）

---

### Step D：模型里怎么 embed（真正的「编码」）

文件：`factory_bn/backbone.py` + `model.py`

```python
# BNPDFormer 关掉了日内/星期 Embedding
DataEmbedding(..., add_time_in_day=False, add_day_in_week=False)

# forward:
x = Linear(18 → embed_dim=64)(X)   # TokenEmbedding，所有节点共用一套权重
x += 时间位置编码
x += LaplacianPE(adj_mx)           # 工艺/物流图结构位置
```

之后进 PDFormer 的 Temporal / Geo / Sem 注意力。  
**没有** `nn.Embedding(num_states, …)` 对 IDLE/PROCESSING 做查表。

---

### 和「异质数据」的关系（容易误解的点）

| 你可能以为的 | 当前实际做法 |
|--------------|--------------|
| 每张表/每类资源各自一套 encoder | 先映射+聚合到统一 float 向量 |
| 离散 state 用 Embedding | 变成占比/时长再 Linear |
| HeteroGNN（按边类型 message） | 单一 `adj_mx` + 类型 one-hot；边类型未分开传 |

---

2.
