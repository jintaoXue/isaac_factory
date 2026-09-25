# 纯理想仿真与 Isaac 可视化

训练和数值评测默认使用 `logic` 后端，不启动 Isaac Sim，不加载 PhysX、USD 或 Carbonite，不写场景位姿、不渲染。**每次 env.step 仍推进一个逻辑步**，没有事件跳跃、有效决策步筛选或时间加速。

| 场景 | 后端 | 逻辑更新 | Isaac / 场景 |
|---|---|---|---|
| 普通训练、`--test`、现有 headless 脚本 | logic（默认） | 原逐步更新 | 不启动 |
| `--visualize` | isaac | 相同逻辑管理器 | 原显示路径 |
| `--video` / `--enable_cameras` / livestream | isaac（自动选择） | 相同逻辑管理器 | 原相机/渲染路径 |
| `--sim_backend isaac` 或 `HC_SIM_BACKEND=isaac` | isaac（显式兼容开关） | 相同逻辑管理器 | 保留旧引擎路径 |

后端在进程启动时确定，不支持同一进程运行中切换。显式 `logic` 与可视化请求冲突时立即报错。当前纯逻辑入口支持 HRTPaHC-v1 的 hier / flat / rule_based；暂不支持 `--distributed`，可以分别在多台机器或 GPU 启动独立实验。

## 命令

已有命令无需增加前缀；环境里如果曾设置 `HC_SIM_BACKEND=isaac`，先取消该覆盖。

```bash
conda activate isaac-lab
cd /home/xue/work/isaac_factory

# 训练与数值评测：默认完全不启动 Isaac
bash run_2026_journal_experiments.sh E5-human-pair-c cuda:0
bash run_2026_journal_experiments.sh eval-E5-human-pair-c cuda:0

# 直接使用原 train.py；可使用 CPU 或 CUDA
python train.py --task HRTPaHC-v1 --algo hier --device cuda:0 --headless

# 可视化：仍需要原 Isaac 环境与 USD 资产；其余训练/加载参数照常填写
python train.py --task HRTPaHC-v1 --algo hier --device cuda:0 --visualize
```

既有 checkpoint、算法开关、人因奖励、教师、AR/ORU 和评测 seeds 不因后端选择而改变。现有训练进程不会被切换；新启动的进程才使用新后端。不要覆盖正在运行的实验目录。

启动日志应显示：`[Simulation] backend=logic; ... physics=OFF; render=OFF; one tick per step`。保存的 agent.yaml 和 W&B config 增加 `sim_backend`，便于区分旧结果。逻辑步数、epsilon/教师衰减、checkpoint 步数和 makespan 单位保持原定义。

## 代码检查与实现

- 原 `HcSingleEnvBase.step_env_logic` 的管理器顺序保留：storage → material → human → robot → route → machine → task → mask。疲劳/恢复、理想运动、机床/天车动画计时、路由避让、任务完成、reward 与自动重置仍运行。
- 新 `source/hc_logic_runtime.py` 提供轻量向量环境，复用这些管理器；相机 manager 不构建，`apply_data_to_sim` 在逻辑模式下不做操作。
- `train.py` 在任何 Isaac 导入前选后端，复用原参数处理、RL-Games 算法、训练循环、评测和日志。纯逻辑 Hydra 桥保留 `agent.params.config.*` 覆盖语法。
- 审计发现原代码有两处场景读数：货架的初始基准位姿，以及隐藏物料 reset 位姿。现在两种后端统一读取静态布局，不再让显示帧影响逻辑初始化。隐藏物料每次重置使用固定初始位姿，不继承上一局最后显示的位姿；仍置于地下 z=-100。
- 静态布局 `env_asset_cfg/static_layout.json` 从当前 HC_import.usd 离线导出 113 个 local pose，含源文件 SHA256；货架位置与已有 Isaac 状态转储逐项一致。布局转换沿用原公式，不修改生产参数。

纯逻辑运行不需要 USD/Isaac，但仍需要原 Python 依赖（PyTorch、NumPy、Hydra、RL-Games 等）以及 `cfg_route.py` 指向的地图/路由 JSON。换机器需要同步本次代码、静态布局 JSON、路由数据和模型权重。若更换工厂 USD 布局，必须更新静态布局，不能沿用旧文件。

可在安装了 pxr 的离线 Python 环境中重新导出，无需启动 Isaac：

```bash
python tools/export_static_layout.py /path/to/HC_import.usd
```

## 验证与边界

验证包含：纯逻辑初始化、货架静态布局对照、逐步超时与自动重置、任务完成/耗时事件、环境 checkpoint 恢复、完整小订单生产，以及进程内没有加载 isaaclab/isaacsim/omni/pxr/carb 模块。另用原 train.py 做了隔离 CPU 训练冒烟，并用真实 T0 checkpoint、非协议 seed9001 做短评测冒烟。

本次通过 7 项环境测试、2 项完整入口测试及 30 项人因算法回归测试。完整入口测试在临时目录运行，并在进程结束时断言没有导入引擎模块。

这些验证不代表新的调度性能结果，也不代表已测得加速倍数。未运行完整 N10 协议评测或 Isaac 可视化逐帧对照；后续比较耗时应固定硬件、逻辑步数和学习设置，区分启动时间与运行时间。
