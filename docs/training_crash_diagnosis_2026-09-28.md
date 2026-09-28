# R0 本机停训诊断：2026-09-28

## 结论与确定性

本次训练主进程发生原生层段错误（SIGSEGV，signal 11），不是正常完成训练。`resource_tracker` 的 6 个 semaphore 清理警告是异常退出后的伴随现象，不能据此认定 semaphore 是根因。已经确定崩溃方式，但缺少调用栈，尚不能确定具体出错的原生库或触发代码。

## 运行与时间线

- 本地运行目录：`logs/rl_games/HcFactory/hier_R0-S42`。
- W&B：<https://wandb.ai/rl-driving/HcFactory_TPA/runs/g4ewfy6w>，名称 `R0-S42-N10`。
- 本地 W&B 目录：`wandb/run-20260927_181639-g4ewfy6w`。
- 训练主进程 PID：109879；W&B 后台进程 PID：110022。
- 香港时间 2026-09-27 18:16:39 启动；2026-09-28 01:03:53 崩溃，约运行 6 小时 47 分钟。
- 最后记录 step=896600、episode=52，未达到配置的 100 局。

内核日志（`journalctl -k --since '2026-09-28 01:03:40' --until '2026-09-28 01:04:20'`）：

```text
Sep 28 01:03:53 sci kernel: HcFactory-hier-[109879]: segfault at 7ffc52d3d4b0 ip 00007ffc52d3d4b0 sp 00007ffc52d3d370 error 15
```

`/var/log/apport.log` 同时记录 PID 109879 收到 signal 11、core limit 0，执行文件为 conda 环境中的 `python3.10`；Apport 因可执行文件不属于系统软件包而忽略。没有找到可用的训练崩溃 core 文件。

W&B 的 `logs/debug-internal.log` 在 01:04:00 才记录：

```text
Internal process exiting, parent pid 109879 disappeared
```

因此 W&B 后台退出发生在主进程崩溃之后。API 将运行标为 `finished`，但这个状态不能证明训练正常结束，必须结合内核日志判断。

## 排查结果

1. 未找到此次时间段内 OOM killer、GPU NVRM Xid 或重启记录。W&B 系统历史抽样 120 条，临近退出的样本中进程 RSS 约 3029 MB、系统可用内存约 135967 MB、GPU 显存占用比例约 7.5%、温度约 51°C。这些证据不支持内存耗尽；抽样监控不能排除瞬时异常。
2. 最后记录的各 head loss 有限，critic mean 约 0.00349。Replay 容量 50000，A 为 528 条，其他 head 约 4192–4400 条，没有接近容量上限。不能把 replay 填满认定为原因。
3. `stalls=L1=34 L2=30 L3=0` 在最后相邻记录中不变；这些计数不能解释内核段错误。
4. 当前 Python 实际导入的 Torch 2.5.1+cu118 来自 IsaacSim 的 `omni.isaac.ml_archive/pip_prebundle`；NumPy 1.26.0 来自 IsaacSim 的 `omni.kit.pip_archive`；W&B 0.12.21 来自 conda site-packages。当前环境继承了多处 IsaacSim 的 PYTHONPATH/LD_LIBRARY_PATH。这是后续排查原生依赖兼容性的候选方向，尚无证据证明某个库导致崩溃。当前导入路径检查也不能替代崩溃时的栈。
5. 没有启用 PYTHONFAULTHANDLER，缺少故障时 Python 线程栈。此前短程 CPU 验证不能证明长程 CUDA 训练稳定。

## 为什么终端没有清楚显示段错误

`tools/run_r_series.py` 用 `subprocess.call` 启动训练，然后直接 `raise SystemExit(...)`。POSIX 下，子进程被 signal 11 终止时返回码为 -11；启动器没有显式解码并打印信号，因此终端可能只出现失败提示和资源清理警告。

参考：[Python 3.10 subprocess 返回码](https://docs.python.org/3.10/library/subprocess.html)。

## 下一次运行的诊断措施

优先补充故障可观测性，暂不把 reward、动作设计或某个依赖版本当作已确认根因：

1. 在启动命令前设置 `PYTHONFAULTHANDLER=1`，并将 stdout/stderr 同时持久化；通过环境继承覆盖训练子进程。Python 3.10 的 faulthandler 可在 SIGSEGV 时输出 Python 栈，有助于定位触发位置，但不会修复崩溃，也不等同于完整 C/CUDA 原生栈。
2. 启动器显式记录训练子进程返回码及信号名称；使用 `tee` 时启用 `set -o pipefail`，避免管道掩盖失败状态。
3. 如果仍然段错误，用调试器捕获原生栈，或配置确实能够保存 conda 进程的 core 收集方式。本机 Apport 已忽略该可执行文件，单独提高 `ulimit -c` 不能保证拿到转储。
4. 获得触发栈后，再做单因素隔离：纯净逻辑训练环境与当前 IsaacSim 混合环境、W&B 开关、CPU/CUDA 或线程配置。每次只改变一个因素；一次不再崩溃不足以证明修复。

参考：[Python 3.10 faulthandler](https://docs.python.org/3.10/library/faulthandler.html)。

## 检查点与复现实验注意事项

- `nn/` 中保留了 step 895000 的六个权重文件和 R replay schema 文件；与最后已记录步数相差 1600 步。这仅确认文件存在，不代表完整恢复训练状态。
- 当前检查点不包含 optimizer、replay、随机状态和 pending transition，不能声称从 895000 步精确续训。当前训练启动器也不会自动从该目录恢复。
- 崩溃运行使用 horizon 35000 / anchor 56000；当前启动器已变为 horizon 25000 / anchor 40000。复现实验需要明确配置差异。
- 本次诊断仅新增本文档，没有修改算法、启动器、训练配置，也没有重启训练。

## 后续实施：自动捕获（2026-09-28）

已新增 `tools/training_supervisor.py` 并接入 R 系列真实运行路径；dry-run 不启动监控。原命令保持不变：

```bash
bash run_2026_journal_experiments.sh R0 cuda:0
```

每次运行会打印独立证据目录 `outputs/train_monitor/run_<时间>_<唯一后缀>/`，保存：

- `console.log`：训练 stdout/stderr，自动启用 `PYTHONFAULTHANDLER=1` 与无缓冲 Python 输出；SIGSEGV 等故障发生时可包含 Python 线程栈。
- `status.json`：训练 PID、开始/结束时间、原始返回码、信号名称及 shell 退出码。例如 SIGSEGV 对应 -11 / SIGSEGV / 139。环境变量不会写入此文件。
- `monitor_*.jsonl`、`monitor_*.log`：复用现有监控，按训练 PID 采集资源情况，采样间隔设置为 10 秒；实际周期受查询耗时影响。
- `monitor.log`：资源监控自身的输出和异常。
- `kernel.log`：运行结束时查询本次运行期间的内核日志；无权限或查询失败会留下错误信息。

原旁路监控只匹配 Xorg 段错误，已改为匹配任意进程的 `segfault` 和 `general protection fault`。已经运行的旧监控进程不会自动加载改动；下一次启动才生效。R 启动器会自动启动并清理自己的资源监控，无需再手动开一份。

监控会转发 SIGINT/SIGTERM/SIGHUP 给训练进程组，不自动重启训练，不改变算法与超参数。即使 W&B 错误显示 finished，也可以根据本地退出证据判断。

验证：5 项测试覆盖正常退出、Python 异常、真实受控 SIGSEGV（禁止生成 core）、SIGTERM 和内核过滤；额外验证了资源监控及内核日志的集成运行。未启动真实长程训练。

限制：不能补回上一次丢失的调用栈；SIGKILL 无法触发进程内故障栈；整机掉电或 supervisor 自身被杀时，最终状态可能仍显示 running。Python 3.10 faulthandler 提供 Python 栈，若要准确追踪 C/CUDA 原生函数仍需 GDB 等调试器。监控改善取证，不代表已经修复段错误根因。
