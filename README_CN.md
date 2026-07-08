# Isaac Factory — 海创工厂生产仿真环境

基于 [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) 与 [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) 构建的**人机协同工厂生产调度仿真环境**，用于模拟水喉（`ProductWaterPipe`）等产品的多工序制造流程，并支持四层实时决策智能体的训练与评估。

> English version: [README.md](README.md)

---

## 目录

- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [数据资产](#数据资产)
- [快速运行](#快速运行)
- [命令行参数](#命令行参数)
- [远程可视化（Livestream）](#远程可视化livestream)
- [框架架构（hc_factory）](#框架架构hc_factory)
- [项目结构](#项目结构)
- [工厂与产品说明](#工厂与产品说明)  
- [Human Subtask 感知训练](#human-subtask-感知训练)
- [相关文档](#相关文档)

## 环境要求

| 组件 | 版本 / 说明 |
|------|-------------|
| NVIDIA Isaac Sim | **4.5.0**（Workstation 安装） |
| NVIDIA Isaac Lab | 与本仓库 `source/` 目录内嵌版本一致 |
| Python | 3.10（conda 环境 `isaaclab`） |
| GPU | 支持 CUDA 的 NVIDIA GPU |
| 操作系统 | Linux（推荐 Ubuntu 20.04 / 22.04） |

---

## 安装步骤

### 1. 安装 Isaac Sim 4.5.0

按 [Isaac Sim 官方安装文档](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html) 完成 Workstation 版安装。

### 2. 单独安装 Isaac Lab（建议 v2.0.1） https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html

克隆 [Isaac Lab 官方仓库](https://github.com/isaac-sim/IsaacLab) 到 `v2.0.1`，创建 `_isaac_sim` 符号链接，并初始化 conda 环境。

> 说明：如未配置 GitHub SSH key，请使用 HTTPS 地址克隆（更省心）。

```bash
git clone --branch v2.0.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 将 Isaac Sim 链接到仓库根目录
ln -s /path/to/isaac-sim _isaac_sim

# 创建 conda 环境 isaaclab
./isaaclab.sh --conda isaaclab

# 激活环境（Isaac Sim 4.5.0 对应 Python 3.10）
conda activate isaaclab

# 安装 Isaac Lab 全部扩展
```

### 3. 安装本仓库（isaac_factory）

在已激活 `isaaclab` 环境的前提下，克隆本仓库：

```bash
git clone git@github.com:jintaoXue/isaac_factory.git
cd isaac_factory
```

---

## 数据资产

仿真场景依赖外部 USD 资产与地图数据，需放置于以下路径：

| 资源 | 路径 | 说明 |
|------|------|------|
| 工厂 USD 场景 | `~/work/Dataset/HC_data/final_for_isaac/HC_import.usd` | 含机器、人员、机器人、物料等全部 3D 资产 |
| 地图路由数据 | `~/work/Dataset/HC_data/map_data/` | `map_routes_human.json`、`map_routes_robot.json` 等 |

> 路径可在 `source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/cfg_hc_env.py` 的 `asset_path` 字段中修改。

本仓库 `map_data/` 目录提供了地图生成与坐标转换工具，用于维护人机共用的路网点数据。

---

## 快速运行

在项目根目录、已激活 `isaaclab` 环境的前提下执行：

```bash
# 带 GUI 运行（4 个并行环境，使用 cuda:1）
python train.py --task HRTPaHC-v1 --algo rule_based --num_envs 4 --device cuda:1

# 无头模式（服务器 / 批量训练）
python train.py --task HRTPaHC-v1 --algo rule_based --num_envs 4 --device cuda:1 --headless
```

当前注册的 Gym 环境 ID 为 **`HRTPaHC-v1`**（Human-Robot Task Planning and Allocation for HC Factory），默认算法为 **`rule_based`**（基于规则的四层多智能体决策）。

运行日志保存在 `logs/rl_games/HcFactory/` 目录下。

---

## 命令行参数

`train.py` 基于 Hydra 配置系统，常用参数如下：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | Gym 环境 ID | `HRTPaHC-v1` |
| `--algo` | 算法配置名 | `rule_based` |
| `--num_envs` | 并行仿真环境数量 | 3 |
| `--device` | CUDA 设备 | `cuda:0` |
| `--headless` | 无 GUI 模式 | 关闭 |
| `--seed` | 随机种子 | 42 |
| `--test` | 测试模式（加载 checkpoint） | 关闭 |
| `--wandb_activate` | 启用 Weights & Biases 日志 | 关闭 |
| `--video` | 录制仿真视频 | 关闭 |
| `--active_livestream` | 启用 Livestream 推流 | 关闭 |
| `--livestream_public_ip` | Livestream 公网 IP | — |
| `--livestream_port` | Livestream 端口 | 49100 |

完整参数列表见 `train.py` 源码。

---

## 远程可视化（Livestream）

在无显示器的服务器上运行仿真，并在本地电脑远程查看画面：

**1. 服务器端（启动仿真 + 推流）**

```bash
python train.py \
  --task HRTPaHC-v1 \
  --algo rule_based \
  --num_envs 1 \
  --active_livestream \
  --livestream_public_ip <服务器公网IP> \
  --livestream_port 49100 \
  --device cuda:1 \
  --headless
```

**2. 客户端（远程电脑）**

按 [Isaac Sim Livestream 客户端文档](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/manual_livestream_clients.html) 安装并运行 Livestream Client，连接上述 IP 与端口即可。

---

## 框架架构（hc_factory）

### 总体设计

`hc_factory` 模块实现了一个**向量化多环境工厂仿真器**，对齐工业实时制造运营的四层决策栈：

```
┌─────────────────────────────────────────────────────────┐
│  Agent A — Product Sequencing（产品排序）               │
│  根据当前生产订单，决定优先生产哪种产品                    │
├─────────────────────────────────────────────────────────┤
│  Agent B — Product Selection（产品选择）                │
│  从待产批次中选择下一个待加工的产品实例                    │
├─────────────────────────────────────────────────────────┤
│  Agent C — Process Task Planning（工序任务规划）         │
│  为选中产品规划下一个关键工序任务（加工 / 物流）           │
├─────────────────────────────────────────────────────────┤
│  Agent D — Human-Robot Allocation（人机资源分配）       │
│  将规划任务分配给最合适的人工、机器人或机器资源             │
└─────────────────────────────────────────────────────────┘
         ↓ action dict
┌─────────────────────────────────────────────────────────┐
│  HcVectorEnv（向量化环境）                               │
│  ├── HcSingleEnv × N（单环境逻辑实例）                   │
│  │   ├── MachineManager      机器状态与工位管理           │
│  │   ├── ProductMaterialManager  物料 / 在制品管理        │
│  │   ├── HumanManager        人工资源与路径              │
│  │   ├── RobotManager        机器人资源与路径            │
│  │   ├── StorageManager      仓储区域管理                │
│  │   ├── TaskManager         任务进度与工序解码          │
│  │   └── AlgoMultiAgentMasker  动作合法性掩码            │
│  └── RouteManagerVectorEnv   跨环境共享路径规划           │
└─────────────────────────────────────────────────────────┘
         ↓ apply_data_to_sim()
┌─────────────────────────────────────────────────────────┐
│  Isaac Sim 物理引擎（USD 场景 + Articulation / RigidBody）│
└─────────────────────────────────────────────────────────┘
```

### 单步仿真流程

每个仿真步（`step`）分为两阶段：

1. **逻辑步（`step_env_logic`）**：各 SingleEnv 接收四层 Agent 的动作字典，更新任务记录、物料状态、人机分配，并计算动作掩码（mask）。
2. **物理步（`step_env_physics`）**：将所有环境的关节位置、刚体位姿写入仿真器，执行 `sim.step()` 推进物理时间。

### 状态 / 动作接口

每个环境实例维护一个 `env_state_action_dict`，包含：

- `machine` / `material` / `human` / `robot` / `storage`：各资源管理器的状态
- `progress`：生产进度（订单、在产、已完成、进行中任务记录）
- `agent_action_mask`：四层 Agent 及各资源的动作合法性掩码
- `action`：当前步接收的动作
- `articulations` / `rigid_prims`：待写入仿真的物理对象数据

### 算法模块

| 文件 | 说明 |
|------|------|
| `source/algo/multiagent/hc_factory/rule_based.py` | 规则基线，串联 A→B→C→D 四层 Agent |
| `source/algo/multiagent/hc_factory/agent_A_product_sequencer.py` | 产品排序 Agent |
| `source/algo/multiagent/hc_factory/agent_B_product_selector.py` | 产品选择 Agent |
| `source/algo/multiagent/hc_factory/agent_C_process_task_planner.py` | 工序任务规划 Agent |
| `source/algo/multiagent/hc_factory/agent_D_human_robot_allocator.py` | 人机资源分配 Agent |
| `source/isaaclab_tasks/.../algo_cfg/rule_based.yaml` | 规则基线 Hydra 配置 |
| `source/isaaclab_tasks/.../algo_cfg/rl_filter.yaml` | RL 过滤器配置（预留） |

---

## 项目结构

```
isaac_factory/
├── train.py                          # 训练 / 仿真入口脚本
├── isaaclab.sh                       # Isaac Lab 环境管理脚本
├── map_data/                         # 地图数据与生成工具
├── source/
│   ├── algo/multiagent/hc_factory/   # 多智能体决策算法
│   ├── isaaclab/                     # Isaac Lab 核心库
│   ├── isaaclab_assets/              # 资产定义
│   ├── isaaclab_rl/                  # RL 框架集成（RL-Games 封装）
│   └── isaaclab_tasks/
│       └── isaaclab_tasks/direct/hc_factory/
│           ├── __init__.py           # Gym 环境注册（HRTPaHC-v1）
│           ├── hc_vector_env.py      # 向量化环境入口
│           ├── hc_vector_env_base.py # 向量化环境基类（场景加载、物理步）
│           ├── hc_single_env.py      # 单环境逻辑
│           ├── hc_single_env_base.py # 单环境基类（Manager 注册与 reset/step）
│           ├── env_asset_cfg/        # 环境资产配置
│           │   ├── cfg_hc_env.py             # 环境全局配置
│           │   ├── cfg_material_product.py   # 产品与物料定义
│           │   ├── cfg_process_task_gallery.py  # 工序任务库
│           │   ├── cfg_process_subtask_gallery.py # 子任务库
│           │   ├── cfg_machine.py            # 机器配置
│           │   ├── cfg_human.py              # 人工配置
│           │   ├── cfg_robot.py              # 机器人配置
│           │   ├── cfg_storage.py            # 仓储配置
│           │   └── cfg_route/                # 路径规划配置与地图点
│           ├── src/                  # 运行时 Manager 实现
│           │   ├── machine.py
│           │   ├── material.py
│           │   ├── human.py
│           │   ├── robot.py
│           │   ├── storage.py
│           │   ├── route.py
│           │   ├── task_progress_manager.py
│           │   └── algo_multiagent_masker.py
│           └── algo_cfg/             # 算法 Hydra 配置
└── logs/                             # 运行日志与参数快照
```

---

## 工厂与产品说明

### 工厂场景

仿真场景为**海创（HC）工厂**，包含多台数控机床、焊接机器人、龙门吊、工作台等生产设备，以及人工操作员与 AGV 机器人。工厂布局与坐标系基于真实工厂地图数据构建，人机共用路网点（部分节点对机器人有 mask 限制）。

### 当前产品：水喉（ProductWaterPipe）

默认生产订单为 5 件水喉，每件经历 6 道加工工序及对应的物流任务：

| 序号 | 工序 | 执行设备 |
|------|------|----------|
| 1 | 管材切割（`pipe_cutting`） | 滚床 CNC 切管机 |
| 2 | 管材开槽（`pipe_grooving`） | 大型开槽机 |
| 3 | 批量点焊（`batch_spot_welding`） | 工作台 |
| 4 | 氩弧焊底焊（`arc_welding_root`） | 焊接机器人 |
| 5 | MIG 面焊（`MIG_welding_surface`） | 旋转管自动焊机 |
| 6 | 防锈漆喷涂（`paint_rust_proof`） | 工作台 |

每道工序前均有龙门吊执行的**物流任务**（将物料/在制品运送至目标机器工位）。物料状态随工序推进从 `pipe → flange/elbow → semi → product` 逐步演化。

工序与任务定义详见 `env_asset_cfg/cfg_process_task_gallery.py` 与 `cfg_process_subtask_gallery.py`。

---

## Human Subtask 感知训练

`perception` 模块用于从工厂多相机图像（及可选结构化信号）估计**每个正在作业的 human 当前 subtask**。流程分两步：仿真内 **collect** 采集数据，再离线 **train** 训练。

配置文件：`env_asset_cfg/cfg_perception.py`（采集与训练超参）、`env_asset_cfg/cfg_camera.py`（相机位姿）。实现代码：`src/perception.py`。

### 实验设计

**目标：** 每个仿真步，对所有 `state != free` 的 human 预测：

| 输出 | 类型 | 说明 |
|------|------|------|
| `subtask_name` | 9 类分类 | human 列 subtask，如 `go_to_material`、`control_gantry`、`wait` |
| `subtask_done` | 二分类 | 当前 human subtask 是否已完成 |

**模型输入：**

| 模态 | 形状 / 格式 | 来源 |
|------|-------------|------|
| 多相机 RGB | `(N_cam, 3, 224, 224)` | 固定工厂相机（`cfg_camera.py`）；每视角 ResNet18 编码 |
| Agent signals（可选） | 8 维向量 | 特权 task-record 特征：`subtask_index`、`num_subtasks`、四列 `finished`、两列 `wait` 标志 |

**Ground Truth（写入 `meta.jsonl`）：**

- 来自仿真 `subtasks_dict`：`subtask_name = ongoing[human]`，`subtask_done = finished[human]`
- 每步同时记录：`task`、`subtask_index`、`area_id`、全场景 `human_labels`、`agent_signals`、`task_records`
- `state == free` 的 human 仅作场景上下文，**不作为预测目标**
- 标签词表见 `cfg_perception.py` 中的 `HumanSubtaskVocab`、`TaskVocab`（与工艺 gallery 同步）

**损失函数：** `CrossEntropy(subtask_name) + BCE(subtask_done)`

**数据集目录结构**（默认 `output/perception_dataset/`）：

```
perception_dataset/
├── manifest.json
└── env_00_episode_000000/
    ├── meta.jsonl              # 每个保存步一行 JSON
    └── cameras/step_000123/    # 该步多相机 JPG
```

### 数据采集

1. 在 `env_asset_cfg/cfg_perception.py` 中开启采集：

```python
CfgPerception = {
    "enabled": True,
    "mode": "collect",   # collect | infer | off
    ...
}
```

2. 确认相机已注册（`cfg_camera.py` 中 `CfgCameraRegistrationInfos`）。

3. 运行仿真（建议 `num_envs=1`，便于整理数据集）：

```bash
python train.py \
  --task HRTPaHC-v1 \
  --algo rule_based \
  --num_envs 1 \
  --device cuda:0 \
  --headless
```

数据默认写入：

`source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_dataset/`

可通过 `CfgPerception` 中的 `save_interval`、`max_episodes`、`max_steps_per_episode` 控制采集量。

### 离线训练

在项目根目录、已激活 `isaaclab` 环境下执行：

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/src/perception.py train \
  --dataset_dir source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_dataset \
  --output_dir source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_runs \
  --run_name subtask_baseline \
  --epochs 20 \
  --batch_size 16 \
  --device cuda:0
```

Checkpoint 保存在 `output/perception_runs/subtask_baseline/`（`best.pt`、`last.pt`、`history.json`）。

**评估：**

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/src/perception.py eval \
  --dataset_dir source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_dataset \
  --checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_runs/subtask_baseline/best.pt \
  --device cuda:0
```

**仿真内推理：** 将 `CfgPerception["mode"]` 设为 `"infer"`，并配置 `checkpoint_path` 指向训练好的 `best.pt`。

---

## 相关文档

- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/main/index.html)
- [Isaac Sim 4.5.0 文档](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [Isaac Sim Livestream 客户端](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/manual_livestream_clients.html)
- 开发笔记：`coding_note.md`
