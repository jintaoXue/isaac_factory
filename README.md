# Isaac Factory — HC Factory Production Simulation Environment

A **human-robot collaborative factory production scheduling simulation** built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) and [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html). It simulates multi-step manufacturing workflows for products such as water pipes (`ProductWaterPipe`), and supports training and evaluation of a four-layer real-time decision-making agent stack.

> 中文版: [README_CN.md](README_CN.md)

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Assets](#data-assets)
- [Quick Start](#quick-start)
- [Command-Line Arguments](#command-line-arguments)
- [Remote Visualization (Livestream)](#remote-visualization-livestream)
- [Framework Architecture (hc_factory)](#framework-architecture-hc_factory)
- [Project Structure](#project-structure)
- [Factory and Product Description](#factory-and-product-description)
- [Human Subtask Perception Training](#human-subtask-perception-training)
- [Related Documentation](#related-documentation)

## Requirements

| Component | Version / Notes |
|-----------|-----------------|
| NVIDIA Isaac Sim | **4.5.0** (Workstation installation) |
| NVIDIA Isaac Lab | Bundled version under `source/` in this repo |
| Python | 3.10 (conda environment `isaaclab`) |
| GPU | NVIDIA GPU with CUDA support |
| OS | Linux (Ubuntu 20.04 / 22.04 recommended) |

---

## Installation

### 1. Install Isaac Sim 4.5.0

Follow the [Isaac Sim installation guide](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html) to install the Workstation edition.

### 2. Install Isaac Lab Separately

Clone the [official Isaac Lab repository](https://github.com/isaac-sim/IsaacLab), create the `_isaac_sim` symlink, and set up the conda environment:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Link Isaac Sim into the repo root (choose one)
# Option A: Symbolic link (recommended)
ln -s /path/to/isaac-sim _isaac_sim
# Option B: Install via pip (isaacsim package; configure ISAAC_PATH manually)

# Create conda environment isaaclab
./isaaclab.sh --conda isaaclab

# Activate the environment
conda activate isaaclab

# Install all Isaac Lab extensions
isaaclab -i
```

### 3. Install This Repository (isaac_factory)

With the `isaaclab` environment activated, clone this repo:

```bash
git clone git@github.com:jintaoXue/isaac_factory.git
cd isaac_factory
```

---

## Data Assets

The simulation depends on external USD assets and map data placed at the following paths:

| Resource | Path | Description |
|----------|------|-------------|
| Factory USD scene | `~/work/Dataset/HC_data/final_for_isaac/HC_import.usd` | All 3D assets: machines, workers, robots, materials |
| Map route data | `~/work/Dataset/HC_data/map_data/` | `map_routes_human.json`, `map_routes_robot.json`, etc. |

> You can change the path in the `asset_path` field of `source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/cfg_hc_env.py`.

The `map_data/` directory in this repo provides map generation and coordinate conversion tools for maintaining shared human/robot route waypoints.

---

## Quick Start

From the project root with the `isaaclab` environment activated:

```bash
# Run with GUI (4 parallel environments, cuda:1)
python train.py --task HRTPaHC-v1 --algo rule_based --num_envs 4 --device cuda:1

# Headless mode (server / batch training)
python train.py --task HRTPaHC-v1 --algo rule_based --num_envs 4 --device cuda:1 --headless
```

The registered Gym environment ID is **`HRTPaHC-v1`** (Human-Robot Task Planning and Allocation for HC Factory). The default algorithm is **`rule_based`** (a four-layer rule-based multi-agent policy).

Run logs are saved under `logs/rl_games/HcFactory/`.

---

## Command-Line Arguments

`train.py` uses the Hydra configuration system. Common arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Gym environment ID | `HRTPaHC-v1` |
| `--algo` | Algorithm config name | `rule_based` |
| `--num_envs` | Number of parallel simulation environments | 3 |
| `--device` | CUDA device | `cuda:0` |
| `--headless` | Run without GUI | off |
| `--seed` | Random seed | 42 |
| `--test` | Test mode (load checkpoint) | off |
| `--wandb_activate` | Enable Weights & Biases logging | off |
| `--video` | Record simulation video | off |
| `--active_livestream` | Enable Livestream | off |
| `--livestream_public_ip` | Livestream public IP | — |
| `--livestream_port` | Livestream port | 49100 |

See `train.py` for the full argument list.

---

## Remote Visualization (Livestream)

Run simulation on a headless server and view it remotely on a local machine:

**1. Server (start simulation + stream)**

```bash
python train.py \
  --task HRTPaHC-v1 \
  --algo rule_based \
  --num_envs 1 \
  --active_livestream \
  --livestream_public_ip <SERVER_PUBLIC_IP> \
  --livestream_port 49100 \
  --device cuda:1 \
  --headless
```

**2. Client (remote machine)**

Install and run the Livestream Client following the [Isaac Sim Livestream documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/manual_livestream_clients.html), then connect to the IP and port above.

---

## Framework Architecture (hc_factory)

### Overview

The `hc_factory` module implements a **vectorized multi-environment factory simulator** aligned with a four-layer real-time manufacturing operations stack:

```
┌─────────────────────────────────────────────────────────┐
│  Agent A — Product Sequencing                           │
│  Determine which product type to prioritize next        │
├─────────────────────────────────────────────────────────┤
│  Agent B — Product Selection                            │
│  Select the next product instance from pending batches    │
├─────────────────────────────────────────────────────────┤
│  Agent C — Process Task Planning                        │
│  Plan the next key process task (processing / logistics)  │
├─────────────────────────────────────────────────────────┤
│  Agent D — Human-Robot Allocation                       │
│  Assign the planned task to human, robot, or machine      │
└─────────────────────────────────────────────────────────┘
         ↓ action dict
┌─────────────────────────────────────────────────────────┐
│  HcVectorEnv (vectorized environment)                   │
│  ├── HcSingleEnv × N (per-env logic instances)          │
│  │   ├── MachineManager         Machine & workstation state │
│  │   ├── ProductMaterialManager Material / WIP management │
│  │   ├── HumanManager           Human resources & routing │
│  │   ├── RobotManager           Robot resources & routing │
│  │   ├── StorageManager         Storage area management   │
│  │   ├── TaskManager            Task progress & decoding  │
│  │   └── AlgoMultiAgentMasker   Action validity masks     │
│  └── RouteManagerVectorEnv      Shared cross-env routing  │
└─────────────────────────────────────────────────────────┘
         ↓ apply_data_to_sim()
┌─────────────────────────────────────────────────────────┐
│  Isaac Sim physics (USD scene + Articulation / RigidBody)│
└─────────────────────────────────────────────────────────┘
```

### Simulation Step Flow

Each simulation step (`step`) has two phases:

1. **Logic step (`step_env_logic`)**: Each SingleEnv receives the four-layer agent action dict, updates task records, material state, human-robot assignments, and computes action masks.
2. **Physics step (`step_env_physics`)**: Writes joint positions and rigid-body poses for all environments into the simulator and advances time via `sim.step()`.

### State / Action Interface

Each environment instance maintains an `env_state_action_dict` containing:

- `machine` / `material` / `human` / `robot` / `storage`: State from each resource manager
- `progress`: Production progress (order, in-progress, finished, ongoing task records)
- `agent_action_mask`: Validity masks for the four agents and all resources
- `action`: Actions received at the current step
- `articulations` / `rigid_prims`: Physical object data to write into simulation

### Algorithm Modules

| File | Description |
|------|-------------|
| `source/algo/multiagent/hc_factory/rule_based.py` | Rule-based baseline chaining Agents A→B→C→D |
| `source/algo/multiagent/hc_factory/agent_A_product_sequencer.py` | Product sequencing agent |
| `source/algo/multiagent/hc_factory/agent_B_product_selector.py` | Product selection agent |
| `source/algo/multiagent/hc_factory/agent_C_process_task_planner.py` | Process task planning agent |
| `source/algo/multiagent/hc_factory/agent_D_human_robot_allocator.py` | Human-robot allocation agent |
| `source/isaaclab_tasks/.../algo_cfg/rule_based.yaml` | Rule-based Hydra config |
| `source/isaaclab_tasks/.../algo_cfg/rl_filter.yaml` | RL filter config (reserved) |

---

## Project Structure

```
isaac_factory/
├── train.py                          # Training / simulation entry point
├── isaaclab.sh                       # Isaac Lab environment management script
├── map_data/                         # Map data and generation tools
├── source/
│   ├── algo/multiagent/hc_factory/   # Multi-agent decision algorithms
│   ├── isaaclab/                     # Isaac Lab core library
│   ├── isaaclab_assets/              # Asset definitions
│   ├── isaaclab_rl/                  # RL framework integration (RL-Games wrapper)
│   └── isaaclab_tasks/
│       └── isaaclab_tasks/direct/hc_factory/
│           ├── __init__.py           # Gym env registration (HRTPaHC-v1)
│           ├── hc_vector_env.py      # Vectorized environment entry
│           ├── hc_vector_env_base.py # Vectorized env base (scene load, physics step)
│           ├── hc_single_env.py      # Single-env logic
│           ├── hc_single_env_base.py # Single-env base (manager registration, reset/step)
│           ├── env_asset_cfg/        # Environment asset configuration
│           │   ├── cfg_hc_env.py             # Global env config
│           │   ├── cfg_material_product.py   # Product & material definitions
│           │   ├── cfg_process_task_gallery.py  # Process task gallery
│           │   ├── cfg_process_subtask_gallery.py # Subtask gallery
│           │   ├── cfg_machine.py            # Machine config
│           │   ├── cfg_human.py              # Human config
│           │   ├── cfg_robot.py              # Robot config
│           │   ├── cfg_storage.py            # Storage config
│           │   └── cfg_route/                # Route planning config & map points
│           ├── src/                  # Runtime manager implementations
│           │   ├── machine.py
│           │   ├── material.py
│           │   ├── human.py
│           │   ├── robot.py
│           │   ├── storage.py
│           │   ├── route.py
│           │   ├── task_progress_manager.py
│           │   └── algo_multiagent_masker.py
│           └── algo_cfg/             # Algorithm Hydra configs
└── logs/                             # Run logs and config snapshots
```

---

## Factory and Product Description

### Factory Scene

The simulation models the **HC (Haichuang) factory**, including CNC machines, welding robots, gantry cranes, workbenches, human operators, and AGV robots. The factory layout and coordinate system are built from real factory map data. Humans and robots share route waypoints (some nodes are masked for robots).

### Current Product: Water Pipe (ProductWaterPipe)

The default production order is 5 water pipes. Each unit goes through 6 processing steps plus corresponding logistics tasks:

| Step | Process | Equipment |
|------|---------|-----------|
| 1 | Pipe cutting (`pipe_cutting`) | Roller-bed CNC pipe cutting machine |
| 2 | Pipe grooving (`pipe_grooving`) | Large grooving machine |
| 3 | Batch spot welding (`batch_spot_welding`) | Workbench |
| 4 | Argon arc welding root (`arc_welding_root`) | Welding robot |
| 5 | MIG surface welding (`MIG_welding_surface`) | Rotary pipe automatic welding machine |
| 6 | Rust-proof paint (`paint_rust_proof`) | Workbench |

Each processing step is preceded by a **logistics task** executed by the gantry crane (transporting materials/WIP to the target machine workstation). Material state evolves through the process: `pipe → flange/elbow → semi → product`.

Process and task definitions are in `env_asset_cfg/cfg_process_task_gallery.py` and `cfg_process_subtask_gallery.py`.

---

## Human Subtask Perception Training

The `perception` module learns to estimate **each working human's current subtask** from factory camera images (and optional structured signals). It runs in two phases: **collect** data inside simulation, then **train** offline.

Configuration: `env_asset_cfg/cfg_perception.py` (collection / training hyperparameters), `env_asset_cfg/cfg_camera.py` (camera poses). Implementation: `src/perception.py`.

### Experiment Design

**Goal:** At every simulation step, for every human with `state != free`, predict:

| Output | Type | Description |
|--------|------|-------------|
| `subtask_name` | 9-class classification | Human-column subtask, e.g. `go_to_material`, `control_gantry`, `wait` |
| `subtask_done` | binary | Whether the current human subtask is finished |

**Inputs (model):**

| Modality | Shape / format | Source |
|----------|----------------|--------|
| Multi-camera RGB | `(N_cam, 3, 224, 224)` | Fixed factory cameras (`cfg_camera.py`); ResNet18 per view |
| Agent signals (optional) | 8-dim vector | Privileged task-record features: `subtask_index`, `num_subtasks`, four `finished` flags, two `wait` flags |

**Ground truth (logged in `meta.jsonl`):**

- From simulation `subtasks_dict`: `subtask_name = ongoing[human]`, `subtask_done = finished[human]`
- Also logged per step: `task`, `subtask_index`, `area_id`, full-scene `human_labels`, `agent_signals`, `task_records`
- Humans with `state == free` are context only; they are **not** prediction targets
- Label vocab: `HumanSubtaskVocab` and `TaskVocab` in `cfg_perception.py` (synced with process galleries)

**Loss:** `CrossEntropy(subtask_name) + BCE(subtask_done)`

**Dataset layout** (default `output/perception_dataset/`):

```
perception_dataset/
├── manifest.json
└── env_00_episode_000000/
    ├── meta.jsonl              # one JSON object per saved step
    └── cameras/step_000123/    # multi-camera JPGs for that step
```

### Data Collection

1. In `env_asset_cfg/cfg_perception.py`, enable collection:

```python
CfgPerception = {
    "enabled": True,
    "mode": "collect",   # collect | infer | off
    ...
}
```

2. Ensure cameras are registered (`CfgCameraRegistrationInfos` in `cfg_camera.py`).

3. Run simulation (single env recommended for a clean dataset):

```bash
python train.py \
  --task HRTPaHC-v1 \
  --algo rule_based \
  --num_envs 1 \
  --device cuda:0 \
  --headless
```

Data is written under:

`source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_dataset/`

Tune `save_interval`, `max_episodes`, and `max_steps_per_episode` in `CfgPerception`.

### Offline Training

From the project root with the `isaaclab` environment activated:

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/src/perception.py train \
  --dataset_dir source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_dataset \
  --output_dir source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_runs \
  --run_name subtask_baseline \
  --epochs 20 \
  --batch_size 16 \
  --device cuda:0
```

Checkpoints are saved to `output/perception_runs/subtask_baseline/` (`best.pt`, `last.pt`, `history.json`).

**Evaluation:**

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/src/perception.py eval \
  --dataset_dir source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_dataset \
  --checkpoint source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/perception_runs/subtask_baseline/best.pt \
  --device cuda:0
```

**Inference in simulation:** set `CfgPerception["mode"] = "infer"` and `checkpoint_path` to a trained `best.pt`.

---

## Related Documentation

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/index.html)
- [Isaac Sim 4.5.0 Documentation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [Isaac Sim Livestream Client](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/manual_livestream_clients.html)
- Development notes: `coding_note.md`