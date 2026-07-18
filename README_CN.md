# Isaac Factory — 海创（HC）工厂人机协同生产仿真

基于 [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) 与 [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) 的**工厂级生产调度仿真环境**：在三维场景中复现海创（HC）工厂的机器、人员、机器人与物流，模拟水喉（`ProductWaterPipe`）等多工序制造流程，并支持四层实时决策智能体的训练、评估与视觉感知数据采集。

> English: [README.md](README.md)

**推荐软件栈（`master`）：** Isaac Sim **5.1.0** + Isaac Lab **2.3.2** · Ubuntu 22.04 · RTX 5090

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

### 本仓库已验证环境

以下组合已在实际训练中验证通过；其他硬件/系统组合也可能可用，但尚未在本项目中测试：

| Isaac Sim | Isaac Lab | 操作系统 | GPU | 说明 |
|-----------|-----------|----------|-----|------|
| **4.5.0** | **2.0.1** | Ubuntu 20.04 | RTX 4090 | 早期稳定栈 |
| **5.1.0** | **2.3.2** | Ubuntu 22.04 | RTX 5090 | 当前推荐栈（`master` 分支） |

> 版本需配套：4.x 对应 Python 3.10，5.x 对应 Python 3.11。须先按对应标签安装上游 [Isaac Lab](https://github.com/isaac-sim/IsaacLab)，再克隆本仓库。

### 通用要求

RAM、显存、驱动等硬件要求请参考 [Isaac Sim 系统要求（System Requirements）](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)。

---

## 安装步骤

整体顺序（**三个独立步骤，不可跳过**）：

1. 安装并验证 **Isaac Sim**
2. **单独**克隆、配置官方 **Isaac Lab** 仓库（创建 conda 环境、安装扩展、跑通官方示例）
3. 再克隆本仓库 **isaac_factory**，复用已配置好的 conda 环境

推荐目录布局：

```
~/work/
├── IsaacLab/          # 步骤 2：官方 Isaac Lab 仓库（conda 环境在此创建）
└── isaac_factory/     # 步骤 3：本项目（hc_factory 环境与 train.py）
```

详细流程以 [Isaac Lab 本地安装总览](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 与 [预编译 Isaac Sim + 源码 Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html) 为准。

### 1. 安装并验证 Isaac Sim

按所选版本安装 Workstation 预编译包：

| 版本 | 官方安装文档 |
|------|----------------|
| 4.5.0 | [Isaac Sim 4.5.0 安装](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html) |
| 5.1.0 | [Isaac Sim 5.1.0 安装](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html) |

解压后建议设置环境变量（路径按实际安装位置修改）：

```bash
export ISAACSIM_PATH="${HOME}/isaacsim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

**验证 Isaac Sim 能否正常启动：**

```bash
# 启动仿真器（可加 --help 查看参数）
${ISAACSIM_PATH}/isaac-sim.sh

# 验证 Python 与独立脚本
${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.api/add_cubes.py
```

若从旧版本升级，首次启动建议执行：`${ISAACSIM_PATH}/isaac-sim.sh --reset-user`。

### 2. 单独配置 Isaac Lab 仓库

> **重要：** 必须先完成本步骤。Isaac Lab 与 `isaac_factory` 是**两个独立的 Git 仓库**；conda 环境、`./isaaclab.sh --install` 均在 **IsaacLab** 目录下执行。

按 [Isaac Lab 官方安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html) 操作：

```bash
cd ~/work

# 5.1.0 栈（推荐）
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 4.5.0 栈
# git clone --branch v2.0.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git
# cd IsaacLab

ln -sfn ${ISAACSIM_PATH} _isaac_sim

./isaaclab.sh --conda isaaclab
conda activate isaaclab

./isaaclab.sh --install
```

**在 IsaacLab 目录下验证（必须通过后再进行步骤 3）：**

```bash
conda activate isaaclab
cd ~/work/IsaacLab
python scripts/tutorials/00_sim/create_empty.py
```

版本对齐文档：

- Sim 4.5.0 + Lab 2.0.1 → [v2.0.1 安装说明](https://isaac-sim.github.io/IsaacLab/v2.0.1/source/setup/installation/binaries_installation.html)
- Sim 5.1.0 + Lab 2.3.2 → [v2.3.2 安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/binaries_installation.html)

### 3. 克隆 isaac_factory 并试跑

Isaac Lab 仓库配置完成且 `create_empty.py` 跑通后，**另起目录**克隆本项目：

```bash
cd ~/work
git clone https://github.com/jintaoXue/isaac_factory.git
cd isaac_factory

# 本仓库运行 train.py 时同样需要 _isaac_sim 链接
ln -sfn ${ISAACSIM_PATH} _isaac_sim

# 复用步骤 2 中已创建的 conda 环境，无需重新 --conda / --install
conda activate isaaclab
```

按 [数据资产](#数据资产) 放置 USD 与地图文件后试跑：

```bash
python train.py --task HRTPaHC-v1 --algo rule_based --num_envs 1 --device cuda:0 --headless
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

# 无头 + 相机 / 感知采集（需启用渲染 kit）
python train.py --task HRTPaHC-v1 --algo rule_based --num_envs 1 --device cuda:0 --headless --enable_cameras
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
| `--enable_cameras` | 启用相机与离屏 RTX 渲染（headless 下采集图像必需） | 关闭 |
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
│           │   └── route/                    # 路径规划配置与地图点
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

`perception` 模块用工厂多相机图像做两个任务：**(1) 各视角画面中的 human id**；**(2) working human 的当前 subtask / done**。流程：仿真内 **collect** → 离线 **train**。

配置：`env_asset_cfg/perception/cfg_perception.py`、`cfg_camera.py`（含手动 `ground_footprint_xy`）。实现：`src/perception.py`。

### 实验设计

**任务 A — human id recognition（每相机多标签）**

| 项 | 说明 |
|----|------|
| GT | `rigid_prims` XY 落在该相机 `ground_footprint_xy` 多边形内 → `human_ids` |
| 跳过 | `detect_human_id=False` 的高空相机（仍存图） |
| 训练 | 单目 ResNet18 + BCE multi-hot（`HumanIdVocab`） |

**任务 B — human subtask recognition（working human）**

| 输出 | 说明 |
|------|------|
| `human_subtask` | 9 类（`HumanSubtaskVocab`） |
| `human_subtask_done` | 二分类 |
| 输入 | 多相机 RGB + `human_task_id` |

**`meta.jsonl`：** 每个保存步 **一行**，结构对齐 `PerceptionSampleTemplate`（`human_id_recognition` / `human_subtask_recognition` / 精简 `env_state_action_dict`）。

```
perception_dataset/
├── manifest.json
└── env_00_episode_000000/
    ├── meta.jsonl
    └── cameras/step_000123/
```

### 数据采集

1. `CfgPerception["mode"] = "collect"`，`enabled=True`
2. 确认 `cfg_camera.py` 已注册相机并标定 footprint
3. 运行：

```bash
python train.py \
  --task HRTPaHC-v1 \
  --algo rule_based \
  --num_envs 1 \
  --device cuda:0 \
  --headless \
  --enable_cameras
```

数据目录：`.../hc_factory/output/perception_dataset/`

### 离线训练

```bash
# 任务 A：human id
python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/src/perception.py train \
  --task id \
  --dataset_dir source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_dataset \
  --run_name perception_baseline \
  --epochs 20 --batch_size 32 --device cuda:0

# 任务 B：subtask
python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/src/perception.py train \
  --task subtask \
  --dataset_dir source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_dataset \
  --run_name perception_baseline \
  --epochs 20 --batch_size 32 --device cuda:0
```

Checkpoint：`output/perception_runs/perception_baseline_{id|subtask}/`

```bash
python .../perception.py eval --task subtask \
  --dataset_dir .../perception_dataset \
  --checkpoint .../perception_baseline_subtask/best.pt \
  --device cuda:0
```

---

## 相关文档

- [Isaac Lab 安装总览](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [Isaac Lab 预编译 Sim + 源码 Lab 安装](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html)
- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/main/index.html)
- [Isaac Sim 系统要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [Isaac Sim 5.1.0 文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
- [Isaac Sim 4.5.0 文档](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [Isaac Sim Livestream 客户端](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/manual_livestream_clients.html)
- 开发笔记：`coding_note.md`
