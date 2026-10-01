# Windows 双系统部署备忘（4090 工位 / 备份盘）

> 用途：本机 Ubuntu 桌面与训练抢同一块 **无核显的 RTX 4090**，夜间易整机硬卡死。计划改用 **双系统里的 Windows** 跑 `isaac_factory` 长训。  
> 写盘位置：备份盘仓库 `docs/windows_dualboot_deploy.md`（便于进 Windows 后直接打开）。  
> 对齐原则：**软件栈与流程对照 Ubuntu**，算法代码同一仓库；不改 R 系列网络结构。

相关：

- Ubuntu 安装总览：[`README_CN.md`](../README_CN.md)
- R 系列训练：[`experiment_r_series.md`](experiment_r_series.md)
- 本机崩溃/卡死诊断：[`training_crash_diagnosis_2026-09-29.md`](training_crash_diagnosis_2026-09-29.md)

---

## 1. 目标与约束

| 项 | 说明 |
|----|------|
| 机器 | 4090 工位，**无核显**；显示与 CUDA 共用同一 GPU |
| Ubuntu 问题 | 桌面（Xorg/Cursor/Chrome 等）+ 训练 → 易 **整机硬卡死**；另有间歇 **SIGSEGV**（原生层，与 OS 不完全相关） |
| Windows 期望 | 减少桌面抢占导致的硬卡死；你家已有 Windows 跑 isaac_factory 的经验可复用 |
| 不能指望 | 换 Windows **不一定**消掉 Torch/CUDA SIGSEGV；桌面开着仍会占 GPU |
| 备份盘 | `/media/xue/Ubuntu 22.0/`（Ubuntu）↔ Windows 下通常为某盘符（如 `D:` / `E:`，以资源管理器为准） |

---

## 2. 进 Windows 后先做的系统设置（必做）

### 2.1 关闭/推迟自动更新（避免夜晚断训）

训练最怕：**更新完强制重启**。

**所有版本建议先做：**

1. **设置 → Windows Update → 高级选项 → 使用时段**  
   - 设成覆盖整晚训练窗口（例：`08:00–02:00`，按你作息拉长）。  
2. **Windows Update → 暂停更新**（Home 一般最长约数周；到期前记得再暂停或手动处理）。  
3. **设置 → 系统 → 电源**  
   - 接通电源时：**屏幕关闭**可设较久；**睡眠 = 从不**（睡眠等于断训）。

**若是专业版/企业版（更稳）：**

1. `Win + R` → `gpedit.msc`  
2. 计算机配置 → 管理模板 → Windows 组件 → **Windows 更新**  
3. 建议：  
   - **配置自动更新** → 已启用 → 选「通知下载并通知安装」或「自动下载但通知安装」（不要静默安装并自动重启）  
   - 查找并启用类似 **「对已登录用户不自动重启以安装更新」** 的策略  

Home 版没有组策略时：依赖「使用时段 + 暂停更新」；开训前看一眼是否「等待重启」。

### 2.2 中文语言与输入法

1. **设置 → 时间和语言 → 语言和区域**  
2. **添加语言** → **中文（简体，中国）** → 安装语言包  
3. 可选：设为 Windows **显示语言**（需注销/重启）  
4. 该语言 → 选项 → 键盘 → 确保有 **微软拼音**；没有则「添加键盘」  
5. 切换：任务栏输入法图标，或 **Win + 空格**

网络异常导致下不了语言包时：检查网络 / 可选更新里的语言相关包。

---

## 3. 需要安装的软件（清单）

按顺序安装；版本尽量与 Ubuntu 工位或你家已跑通的 Windows 机 **一致**。

### 3.1 系统与驱动

| 软件 | 要求 / 备注 |
|------|-------------|
| Windows 10/11 | 建议 64 位；记下是 **Home 还是 Pro**（影响组策略） |
| NVIDIA Game Ready / Studio 驱动 | 与 Isaac Sim 文档要求匹配；装完重启 |
| Visual C++ Redistributable | 常见依赖；缺 DLL 时再补 |
| Git for Windows | 克隆仓库；自带 Git Bash（可跑部分 `.sh`，但 R 入口优先用 `python`） |

### 3.2 Python / Isaac 栈（对照 Ubuntu）

本仓库 README 验证过的 Linux 组合（Windows 需自行对齐相近版本）：

| Isaac Sim | Isaac Lab | 说明 |
|-----------|-----------|------|
| **5.1.0** | **2.3.2** | 当前推荐（5090 Ubuntu 主栈） |
| **4.5.0** | **2.0.1** | 早期 4090 栈 |

安装顺序（与 Ubuntu 相同，**三步不可跳过**）：

1. 安装并验证 **Isaac Sim**（Windows Workstation 预编译包，见 NVIDIA 对应版本文档）  
2. 单独克隆配置官方 **Isaac Lab**，创建 conda 环境并 `isaaclab` 安装扩展  
3. 再放本仓库 **isaac_factory**，复用同一 conda 环境  

官方文档入口：

- Isaac Sim 安装（按版本选）：见 [`README_CN.md` 安装步骤](../README_CN.md)  
- Isaac Lab：[Binaries + source Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html)

**Conda：** Miniconda/Anaconda（环境名建议与 Ubuntu 一致，如 `isaaclab`）。

### 3.3 本仓库与数据

| 项 | 说明 |
|----|------|
| `isaac_factory` | 从备份盘拷贝，或 `git clone` 后把诊断修补一并带上 |
| `Dataset/HC_data` | 至少：`final_for_isaac/HC_import.usd`、`map_data/map_routes_*.json` 等（见 README「数据资产」） |
| W&B | `pip`/环境内已有 `wandb`；准备 API key；本地可放 `.wandb_local.env`（勿提交 git） |

### 3.4 可选但有用

| 软件 | 用途 |
|------|------|
| CUDA Toolkit（若文档要求独立安装） | 部分环境需要；以 Isaac/Torch 轮子说明为准 |
| VS Code / Cursor | 看日志；训练时尽量别重度占 GPU |
| 7-Zip | 解压 Isaac/数据 |
| Notepad++ | 看本备忘与日志 |

---

## 4. 目录布局建议（Windows）

对照 Ubuntu 的 `~/work/`：

```text
<盘符>:\work\
  IsaacLab\              # 官方 Lab + conda 环境在此创建
  isaac_factory\         # 本仓库（可由备份盘同步）
  Dataset\HC_data\       # USD + map_routes 等
```

环境变量示例（PowerShell，路径按实际改）：

```powershell
$env:ISAACSIM_PATH = "C:\isaacsim"   # 或你的安装目录
# 激活 conda 后再训
conda activate isaaclab
cd <盘符>:\work\isaac_factory
```

逻辑后端 R0 训练通常 **不需要** 开 Isaac 窗口；仍建议先按官方示例验证 Sim/Lab 能跑通。

---

## 5. R0 训练启动（对齐 Ubuntu）

Ubuntu 上等价入口：`python tools/run_r_series.py R0 cuda:0`。

Windows（在已 `conda activate` 的项目根目录）：

```powershell
$env:HC_R_RUN_TAG = "R0-debug"
$env:HC_R_SEED = "42"
$env:HC_MAX_HARD_EPISODES = "100"
$env:HC_R_WANDB = "1"
# 可选诊断（默认关）
# $env:HC_EMB_DEBUG = "1"
# $env:HC_CUDA_SYNC_ENCODE = "1"
# $env:CUDA_LAUNCH_BLOCKING = "1"

python tools\run_r_series.py R0 cuda:0
```

说明：

- W&B 显示名约为 **`R0-debug-N10`**（由 `HC_R_RUN_TAG` 生成）。  
- `run_r_series` 默认会清掉 `WANDB_RUN_ID`，每次多为 **新 run**；要「同一 run 续写」需另改启动器（见诊断文档讨论）。  
- 中断后续训：至少用已有 **checkpoint `load_dir` + `load_step` 热启动**；完整 resume（replay/步数）R0 尚未等价实现。

Git Bash 若要用 Ubuntu 同款：

```bash
export HC_R_RUN_TAG=R0-debug HC_R_SEED=42 HC_MAX_HARD_EPISODES=100 HC_R_WANDB=1
python tools/run_r_series.py R0 cuda:0
```

---

## 6. 与 Ubuntu 诊断修补的关系（重要）

本机 Ubuntu 上为 SIGSEGV / deepcopy 做的修改（`clone_env_subtree`、`pre=` 跳过重复 `_to_device`、连续特征 `clone` 再 `stack`、可选 `HC_EMB_DEBUG` 等）：

- **不改变 R 系列网络结构**（Embedding 大小、层宽、拼接维度不变）  
- **旧 checkpoint shape 兼容**  
- Windows 部署应 **带上同一份代码**（从备份盘同步整个 `isaac_factory` 即可）

详见：[`training_crash_diagnosis_2026-09-29.md`](training_crash_diagnosis_2026-09-29.md)。

---

## 7. 训练时降低 GPU 抢占（Windows 仍无核显）

1. 少开 Chrome / 多个 Electron 应用；IDE 可远程或降 GPU 加速。  
2. 电源：**从不睡眠**。  
3. 开训前确认 Windows Update 未卡在「等待重启」。  
4. 长训可用 `Start-Process` / 计划任务，避免关终端杀掉进程（或用 `tmux` 类方案在 Git Bash 里后台）。

---

## 8. 从备份盘同步到 Windows 的检查清单

部署当天按序勾选：

- [ ] 双系统进 Windows；确认备份盘盘符可读  
- [ ] **使用时段 + 暂停更新 + 从不睡眠**  
- [ ] （可选）中文语言包 + 微软拼音  
- [ ] NVIDIA 驱动正常（`nvidia-smi` 或 GeForce Experience）  
- [ ] Isaac Sim 能启动；Isaac Lab 官方空场景示例通过  
- [ ] conda 环境可 `import torch` 且 `torch.cuda.is_available()`  
- [ ] 拷贝/拉取 `isaac_factory`（含诊断修补）与 `Dataset/HC_data`  
- [ ] 配置 `.wandb_local.env`（API key，勿提交）  
- [ ] 短跑 R0（如 `HC_MAX_HARD_EPISODES=3`）确认无秒退  
- [ ] 再开长训；记录 W&B 名与本机日志目录  

---

## 9. 对照：Ubuntu 上已知现象（便于对比）

| 时间 | 现象 | 性质 |
|------|------|------|
| 2026-09-30 ~12:33 | 验证跑中整机卡死 | 主机硬卡死（非 SIGSEGV） |
| 2026-09-30 ~16:01 | R0-debug SIGSEGV @ ~58900 | 训练进程段错误 |
| 2026-09-30 ~22:38 | R0-debug-re1 跑到 ~600k/ep33 后整机卡死 | 再次硬卡死（非 SIGSEGV） |

若 Windows 长训仍 SIGSEGV：把 faulthandler 栈与 step 记到诊断文档，再考虑 Torch/驱动版本对齐 5090 机。

---

## 10. 快速命令备忘（PowerShell）

```powershell
# 驱动是否可见
nvidia-smi

# 激活环境后检查 CUDA
conda activate isaaclab
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 项目根目录短跑
cd <盘符>:\work\isaac_factory
$env:HC_MAX_HARD_EPISODES="3"; $env:HC_R_RUN_TAG="R0-debug-win"; $env:HC_R_WANDB="1"
python tools\run_r_series.py R0 cuda:0
```

文档维护：部署有出入时，在本节或文首更新日期与实际 Isaac/驱动版本号。
