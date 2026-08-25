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
- [批量训练（batch_train.sh）](#批量训练batch_trainsh)
- [日志与 Weights & Biases](#日志与-weights--biases)
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

仿真依赖外部 USD 与路网数据。代码默认路径（可用 `~` 展开为当前用户 home）：

| 资源 | 路径 | 说明 |
|------|------|------|
| 工厂 USD | `~/work/Dataset/HC_data/final_for_isaac/HC_import.usd` | 主场景（`cfg_hc_env.py` → `asset_path`） |
| 人/机路网预计算 | `~/work/Dataset/HC_data/map_data/map_routes_human.json` | human 路网图 |
| AGV 路网预计算 | `~/work/Dataset/HC_data/map_data/map_routes_robot.json` | robot 路网图 |
| 路网点（随仓库） | `.../env_asset_cfg/route/map_points_human.json` 等 | 点号 / occupancy，已在 git 内 |

多机同步时**最少**需拷贝：`HC_import.usd`、`map_points_human.json`（若远端 clone 不完整）、`map_routes_robot.json`；人走完整路径时还需 `map_routes_human.json`。

> USD 路径可在 `env_asset_cfg/cfg_hc_env.py` 的 `asset_path` 修改；路网路径见 `env_asset_cfg/route/cfg_route.py`。

仓库根目录 `map_data/` 另有地图生成与坐标转换工具（维护用）。

### 人因动力学（HeterogeneousHuman）

`cfg_human.py` / `human.py` 中：工人疲劳 `fatigue` → 效率 `η` → 拉长子任务工时与减速行走；熟练度按 task×subtask 异质化。Makespan 仍是唯一 RL 目标；疲劳后工时变长，因此默认 **`T_MAX_ANCHOR=64000`（N=16）/ N=10 → `T_max=40000`**（相对无疲劳旧值约 4×）。可用环境变量 `HC_T_MAX_ANCHOR` 覆盖。

---

## 快速运行

在项目根目录、已激活 conda 环境（如 `isaaclab` / `env_isaaclab`）的前提下：

```bash
# 带 GUI
python train.py --task HRTPaHC-v1 --algo rule_based --num_envs 1 --device cuda:0

# 无头
python train.py --task HRTPaHC-v1 --algo rule_based --num_envs 1 --device cuda:0 --headless

# 无头 + 相机 / 感知采集
python train.py --task HRTPaHC-v1 --algo rule_based --num_envs 1 --device cuda:0 --headless --enable_cameras
```

Gym 环境 ID：**`HRTPaHC-v1`**。常用算法：

| `--algo` | 说明 |
|----------|------|
| `rule_based` | 规则基线：A 准入 → B FIFO → C/D 派工 |
| `hier` | Hierarchical Masked DQN（A→B→C→D） |
| `flat` | Flat 联合动作（代码保留，**非主对比**） |

实验目录：`logs/rl_games/HcFactory/<algo>_<timestamp>/`（含 `nn/`、`params/`、`metrics.jsonl`）。

---

## 批量训练（batch_train.sh）

推荐用 `./batch_train.sh` 跑 Hier4TPA 流水线（编号 22–32）：

```bash
./batch_train.sh              # 查看帮助
./batch_train.sh 24 cuda:0    # rule 单产品 N=10
./batch_train.sh C cuda:0     # N=10 主路径：22→24→25→26→27
```

| 序号 | 内容 | 默认 N / T_max |
|------|------|----------------|
| 22 | explore 采 catalog | 10 / 40000 |
| 23 | explore debug（可视化 + warmstart） | 10 / 40000 |
| 24 / 25 | rule K=1 / K=10 | 10 / 40000 |
| 26 | hier random 基线（ε=1） | 10 / 40000 |
| 27 | hier 倒序 curriculum → target 10 | 段预算 ΔN×per_T |
| 28 / 29 | hier 硬训 / 评测 | 16 / 64000 |
| 30–32 | rule / random N=16 基线 | 16 / 64000 |

常用环境变量：`HC_RULE_EPISODES`、`HC_MULTI_K`、`HC_T_MAX_ANCHOR`、`HC_WARMSTART`、`HC_LOAD_DIR`、`HC_WANDB_MODE`。

---

## 日志与 Weights & Biases

每次训练（无论是否开 wandb）都会把与 `wandb.log` 同结构的指标写入：

```
logs/rl_games/HcFactory/<exp>/metrics.jsonl
logs/rl_games/HcFactory/<exp>/metrics_summary.json   # 结束时汇总
```

`batch_train.sh` 中 22–32 默认带 `--wandb_activate`，**默认 `WANDB_MODE=online`**（边训边上云），同时写本地 jsonl。网络不稳：`HC_WANDB_MODE=offline ./batch_train.sh ...`。

### 共享机器：只用自己的 wandb，不影响别人

**不要**在共享机上执行 `wandb logout`（会改全局 `~/.netrc`）。在本仓库放私有文件（已 gitignore）：

```bash
cd /path/to/isaac_factory   # 或远端 isaac_factory_tpa 等
cp .wandb_local.env.example .wandb_local.env
# 编辑 .wandb_local.env：
#   HC_WANDB_API_KEY=<你的 API key>
#   HC_WANDB_ENTITY=<你的用户名或团队>
#   HC_WANDB_MODE=online
```

再跑 `./batch_train.sh ...`。脚本只在当前进程注入 `WANDB_API_KEY` / `WANDB_ENTITY`，**不改系统全局登录**。

也可临时一行：

```bash
HC_WANDB_API_KEY=xxx HC_WANDB_ENTITY=your_name ./batch_train.sh 24 cuda:0
```

启动时应看到：`[wandb] loaded local env: .wandb_local.env` 以及 `entity=your_name`。

---

## 命令行参数

`train.py` 基于 Hydra，常用参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | Gym 环境 ID | `HRTPaHC-v1` |
| `--algo` | `rule_based` / `hier` / `flat` | `rule_based` |
| `--num_envs` | 并行环境数 | 3 |
| `--device` | CUDA 设备 | `cuda:0` |
| `--headless` | 无 GUI | 关闭 |
| `--enable_cameras` | 相机与离屏渲染 | 关闭 |
| `--seed` | 随机种子 | 42 |
| `--test` | 评测模式 | 关闭 |
| `--test_times` / `--test_seeds` | 评测 episode / seeds | 见 yaml |
| `--train_n_products` | rule 训练订单件数 | yaml（常用 10） |
| `--explore` / `--curriculum` | hier 采库 / 倒序课程 | 关闭 |
| `--wandb_activate` | 启用 W&B | 关闭 |
| `--wandb_project` / `--wandb_name` | W&B 项目与 run 名 | 见 yaml |
| `--video` | 录视频 | 关闭 |
| `--active_livestream` 等 | Livestream | 关闭 |

完整列表见 `train.py`。

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

`hc_factory` 模块实现了一个**向量化多环境工厂仿真器**，对齐工业实时制造运营的四层决策栈（**Hierarchical TPA**）。算法侧经 **信息池** 在同一步内完成 A→B→C/D×K，再写入 env：

```
┌─────────────────────────────────────────────────────────┐
│  Agent A — Product Sequencing（产品准入）                 │
│  每决策步最多准入 1 个新产品类型到候选槽                     │
├─────────────────────────────────────────────────────────┤
│  Agent B — Product Priority（在制优先级）                 │
│  对 eligible 槽排序（规则：在制 FIFO，staging 靠后）        │
├─────────────────────────────────────────────────────────┤
│  Agent C/D × K — Process + Allocation（并行派工）         │
│  按优先级循环：规划工序任务 → 分配 human/AGV；K 可调         │
│  （信息池更新 human/robot/gantry/机位 capacity mask）      │
└─────────────────────────────────────────────────────────┘
         ↓ action: sequencing + dispatch_list（+ 兼容字段）
┌─────────────────────────────────────────────────────────┐
│  HcVectorEnv（向量化环境）                               │
│  ├── HcSingleEnv × N（单环境逻辑实例）                   │
│  │   ├── MachineManager / ProductMaterialManager / …   │
│  │   ├── TaskManager         批量解码 dispatch_list；   │
│  │   │                      5.2b 派工成功即入 producing │
│  │   └── AlgoHierarchicalMasker  动作合法性掩码          │
│  └── RouteManagerVectorEnv   跨环境共享路径规划           │
└─────────────────────────────────────────────────────────┘
```

并行度由配置项 **`max_parallel_cd_dispatch`** 控制（`rule_based.yaml` / `hier.yaml`，默认 `1`）。

### 单步仿真流程

每个仿真步（`step`）分为两阶段：

1. **逻辑步（`step_env_logic`）**：各 SingleEnv 接收动作字典（含 `product_sequencing` 与可选 `dispatch_list`），更新任务记录、物料状态、人机分配，并计算动作掩码（mask）。
2. **物理步（`step_env_physics`）**：将所有环境的关节位置、刚体位姿写入仿真器，执行 `sim.step()` 推进物理时间。

### 状态 / 动作接口

每个环境实例维护一个 `env_state_action_dict`，包含：

- `machine` / `material` / `human` / `robot` / `storage`：各资源管理器的状态
- `progress`：生产进度（订单、在产、已完成、进行中任务记录）
- `agent_action_mask`：四层 Agent 及各资源的动作合法性掩码
- `action`：当前步动作，主要字段：
  - `product_sequencing`：A 准入
  - `dispatch_list`：0…K 条 `{slot_index, process_task_planning, human_robot_allocation}`
  - 兼容字段：`product_selection` / `process_task_planning` / `human_robot_allocation`（= 首条 dispatch）
- `articulations` / `rigid_prims`：待写入仿真的物理对象数据

### 算法模块

| 文件 | 说明 |
|------|------|
| `source/algo/hierarchical/hc_factory/rule_based.py` | 规则基线（信息池 + 并行 CD） |
| `source/algo/hierarchical/hc_factory/hierarchical_tpa.py` | Hier Masked DQN 训练 / 评测 |
| `source/algo/hierarchical/hc_factory/hierarchical_dispatch.py` | A→B→C/D×K 动作构建 |
| `source/algo/hierarchical/hc_factory/tpa_info_pool.py` | 步内信息池 / 资源 ledger |
| `source/algo/hierarchical/hc_factory/agent_A_product_sequencer.py` | 产品准入 Agent |
| `source/algo/hierarchical/hc_factory/agent_B_product_priority.py` | 在制优先级 Agent |
| `source/algo/hierarchical/hc_factory/agent_C_process_task_planner.py` | 工序任务规划 Agent |
| `source/algo/hierarchical/hc_factory/agent_D_human_robot_allocator.py` | 人机资源分配 Agent |
| `source/isaaclab_tasks/.../algo_cfg/rule_based.yaml` | 规则基线配置（含 `max_parallel_cd_dispatch`） |
| `source/isaaclab_tasks/.../algo_cfg/hier.yaml` | Hierarchical RL 配置 |

---

## 项目结构

```
isaac_factory/
├── train.py                          # 训练 / 仿真入口
├── batch_train.sh                    # Hier4TPA 批量任务（22–32）
├── .wandb_local.env.example          # 共享机私有 wandb 模板（复制为 .wandb_local.env）
├── isaaclab.sh
├── map_data/                         # 地图工具（维护用）
├── source/
│   ├── algo/hierarchical/hc_factory/  # 分层 TPA（含 wandb_metrics 本地 jsonl）
│   ├── isaaclab/
│   ├── isaaclab_assets/
│   ├── isaaclab_rl/
│   └── isaaclab_tasks/
│       └── isaaclab_tasks/direct/hc_factory/
│           ├── env_asset_cfg/        # 含 cfg_human 人因、route/ 路网点
│           ├── src/                  # Managers（含 fatigue 工时）
│           └── algo_cfg/             # rule_based.yaml / hier.yaml（t_max_anchor）
└── logs/                             # 实验目录：nn / params / metrics.jsonl
```

---

## 工厂与产品说明

### 工厂场景

仿真场景为**海创（HC）工厂**，包含多台数控机床、焊接机器人、龙门吊、工作台等生产设备，以及人工操作员与 AGV 机器人。工厂布局与坐标系基于真实工厂地图数据构建，人机共用路网点（部分节点对机器人有 mask 限制）。

### 当前产品：水喉（ProductWaterPipe）

默认生产订单全量 **16** 件（idx `00`–`15`）；训练/基线常用 **N=10**。同时在制上限 WIP=`single_env_parallel_producing_limit=10`。人因开启后默认时间预算约 **N=10 → 40000 step，N=16 → 64000 step**（`curriculum.T_MAX_ANCHOR`）。每件 6 道加工工序及对应物流：

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
- 论文 / 实验笔记：`2026_Journal_Paper.md`
