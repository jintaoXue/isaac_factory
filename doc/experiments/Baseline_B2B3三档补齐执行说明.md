# B2/B3 三档补齐执行说明

> 2026-09-15 更新：B2–B5 十二组已全部完成并回传完整指标，最新结果见[十二组实验汇总](Baseline十二组对齐实验汇总_2026-09-15.md)。本文件保留当时的阶段说明；下方“待执行/待补齐”为历史状态，不据此重新运行已完成任务。

日期：2026-09-14。当前目标为补齐 B2（XGBoost）、B3（LSTM）各 Start≤5/10/15，共六组结果；upcoming 根因优化继续暂停。B4/B5 六组已经由用户返回正式核验结果，不重训、不重复收尾。本文件记录代码准备情况，**不代表服务器已经启动或通过预检**。

## 2026-09-15 首组核验错误与恢复

用户已返回服务器日志：三档预检通过，六组登记完成，B2 Start5 在 `verify_prediction_rows` 报 `KeyError: target_cause`。当前不得重复原启动命令，也不要直接 pull 更新训练 checkout。

B2 原 CSV 从 `model_sample_index.csv` 复制元数据，却没有显式导出 `target_cause`、`target_remain_len_windows`。代码执行顺序显示，训练器在进入核验前已完成保存；但仍需恢复入口实际检查模型、summary、指标、预测覆盖和来源后才能认定该组结果完整。测试已改用真实元数据列结构，经实际 CSV 序列化再送入核验器，补上这一回归缺口。

恢复入口为 `recover_remaining_baselines_matched.py`。通过 Git 对象加载修复后的核验/编排，服务器核心训练器仍使用原 `cd29f50`。旧 CSV 只在核验内存中按原数据 train/validation 的 sample_index 补齐两列真值，并独立重算原因混淆和全局剩余时间 MAE；不改写任何已有预测、模型、指标或阈值，不作新推理、拟合或 test 评估。未来新版本的 B2 导出器也已补齐两列，但本批不切换核心训练版本。

入口只接受原第一组这一个错误、原 pane dead/exit1、B4/B5 finisher dead/exit0、后五组均未启动以及干净的 `BSTAN_isaac_factory/dev_xwt@cd29f50`。先核验保存结果，保留失败计划和记录的完整副本，再把第一组登记为已完成、接续剩余五组。验证失败会保留现场停止，不自动重训。

恢复命令使用对话给出的修复提交号。只需 `git fetch origin dev_xwt`，再以系统 Python 从该提交读取恢复入口；不要使用普通 launcher 的 `--resume` 来覆盖这个已知失败。

恢复日志位于原数据根目录：`remaining_matched20260914_recovery20260915.log`。看到 `B2_START5_RECOVERED_WITHOUT_REFIT`、`REUSING_COMPLETED B2 5`，随后 `REMAINING_STAGE_START B3 5`，才表示核验成功且已经接续。最终仍以 `B2_B3_SIX_TASKS_VERIFIED` 为六组完成标志。

本次 49 项不同的本地检查通过；恢复尚未在服务器执行。后续报告应注明这次仅修正核验接口，不把它描述为训练失败重跑或 upcoming 改善。

## 代码与评估范围

实现提交：`33000410300c78f943086cbe3cc9deef4223bee1`，分支 `dev_xwt`。启动时将实际运行的完整提交号写入计划和结果；后续仅含文档的提交不改变训练实现。

- B2 补齐新版协议入口、三档标签、按历史末窗划分 ongoing/upcoming（包括历史冷状态而起点为 0 的 upcoming）、四类原因报告、中段加权剩余时间 MAE，以及验证集选阈值后固定用于 train 的导出。
- B3 复用共同训练器的新版协议，补齐命令行和自身前一档 best 的权重接续。保持原 LSTM 骨干，不增加 near 摘要投影或主模型专有组件。
- 六组共同使用现有 208 episodes、138 train / 30 validation / 40 test 分割，样本数 23859 / 5439 / 6886。只训练 train、评估 train/validation，不索引 test 样本或用 test 调参；加载现有共享文件不等于评估 test。
- 输入来源固定为同一份 30 窗 × 38 节点 × 27 通道历史，global 维度为 0。B2 保持自身表格摘要特征，B3 保持固定节点顺序的 LSTM 输入。信息来源一致不意味着模型内部输入表示相同。
- 评估参考仍为 dev_tyx `7b2ab39` 的 `FactoryBN_dense_12_3_start{5,10,15}_min8_opt.json`：Min8、20 个未来网格、will15 工位 F1、P≥0.80/R≥0.70 联合门槛、统一阈值候选、两种 upcoming 召回、四类原因、中段加权剩余时间 MAE。
- 复用已完成的共同协议预检及标签支持数，只增加 B2/B3 必需的模型预检；没有新的超参数搜索。

## 固定训练安排

顺序：B2 Start5 → B3 Start5 → B2 Start10 → B3 Start10 → B2 Start15 → B3 Start15，均 seed42。

| 模型 | 初始化和训练预算 | 保留的配置 |
| --- | --- | --- |
| B2 XGBoost | 各档独立拟合；每个非恒定预测头最多 500 轮 boosting，没有神经网络 best epoch 或跨档树权重迁移 | 原 7 个头；depth5、learning rate 0.03、subsample/colsample 0.8、min_child_weight3、lambda5、负采样比例4、hot正例权重4、event正例权重12、CPU n_jobs4 |
| B3 LSTM | Start5 从随机初始化训练；Start10/15 从本批自身前一档 best 接续，每档重建优化器，最多100 epoch、patience40、min_epochs12 | LSTM128×1层、node_hidden128、node_embedding32、dropout0.25、batch32、learning rate 0.0003、weight decay0.001 |

B3 使用本批共同神经训练协议的 event 重采样4、upcoming 正例权重9/10/11、误报权重2.5、remain_len损失权重0.5，以及按历史热状态划分的损失。输出范围统一为20网格；这是任务头范围适配，不是新增骨干组件。

旧 B2/B3 权重来自不同数据版本，不用于新批初始化。B4/B5 Start5 则曾使用自身既有 near 父权重；因此后续总表必须披露初始化和累计训练预算差异。B2 固定 boosting 预算与 B3 神经选模方式不同，不强行折算成相同 epoch，也不称为完全相同训练预算的架构因果比较。

## 复用目录与保护

唯一运行仓库：`/home/sci/work/BSTAN_isaac_factory`，分支 `dev_xwt`。不操作主实验仓库。

数据根目录为仓库内：

```text
source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3
```

复用其下两个既有目录：

```text
models/tuning/b2_v5_confirmation_v1/candidate_c5_event_w12/seed42
models/tuning/b3_v5_confirmation_v1/candidate_c0_incumbent/seed42
```

不创建目录。旧模型、指标和预测文件先写入同目录 ZIP，并逐文件校验 SHA 后才移走散装副本；原内容完整保留。各档结果另有独立登记，下档开始前归档上一档。B4/B5 和其他既有文件通过保护清单核对，不覆盖。目录不存在、旧任务仍运行、已存在本批记录或产物校验失败时停止，不删除记录后重试。

## 服务器启动命令

UU 独立终端不可用，因此由用户在服务器 SSH 终端执行。此命令先 fetch，在检查旧任务与正式结果后由启动器快进更新 checkout；不要先强制切分支或覆盖脏工作区。对话中的固定提交命令优先于本文件的通用分支入口。

```bash
cd /home/sci/work/BSTAN_isaac_factory
git fetch origin dev_xwt
/usr/bin/python3 -I -B - <<'PY'
import subprocess, sys
source = subprocess.check_output(['git', 'rev-parse', 'origin/dev_xwt'], text=True).strip()
path = 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/launch_remaining_baselines_matched.py'
sys.argv = [path, '--source_commit', source]
code = subprocess.check_output(['git', 'show', source + ':' + path])
exec(compile(code, path, 'exec'), {'__name__': '__main__', '__file__': path})
PY
```

启动器先检查 `baseline_dense_v6:0.0`、`baseline_dense_diag:0.0` 都已正常退出，B4/B5 最终 JSON 与 finisher state 的 SHA 一致，再复用原训练 tmux pane，不创建新 session 或强杀旧进程。使用既有 `/home/sci/repos/miniconda3/envs/env_isaaclab/bin/python`，保留 tmux 已验证的依赖路径并加入当前 tools，清除 PYTHONHOME；无需另行 conda activate。

服务器预检包含：现有目录/数据身份、原完成结果、XGBoost 分类和回归的微型合成数据运行，以及 B3 三档真实 train 样本的 CUDA 前后向。预检不作正式优化器更新，不重复共同标签统计。缺包或预检失败时保留日志反馈，不盲目安装或新建环境。

查看日志：

```bash
tail -n 40 /home/sci/work/BSTAN_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3/remaining_matched20260914.log
```

`REMAINING_STAGE_START B2 5` 等表示当前阶段；`REMAINING_STAGE_COMPLETE` 表示该阶段训练、导出及检查完成。六组最终完成标志为 `B2_B3_SIX_TASKS_VERIFIED`，只有看到队列已提交且 tmux 内日志开始更新，才视为成功启动。之后断开普通 SSH 连接不会终止该 tmux 任务。

最终文件位于数据根目录：`remaining_matched20260914_results.json`，成功状态为 `six_remaining_tasks_verified`。文件保留六组完整 summary、train/validation 指标、计数、阈值、来源和产物 SHA。B3 同时保存选中轮次及累计预算；B2 标注独立固定 boosting 训练，不伪造 best/总 epoch。

若中断，只允许检查后复用完整阶段；部分训练或部分导出不会自动重跑。`--resume` 为人工核对后的有限恢复入口，不能直接代替失败诊断。

## 本地验证与尚待确认项

本地 23 项新增内存检查及 38 项相关既有检查通过，覆盖新标签/历史划分、B2 完整指标导出与冻结阈值、B3 三档损失反传和前档权重传递、旧协议回归及启动保护。未创建测试目录或访问实验 test 数据。

本地没有 XGBoost 原生依赖，B2 全流程检查使用替代预测头；真实 XGBoost 运行及服务器 CUDA/数据/目录情况由服务器开跑前预检确认。目前没有这六组的新训练成绩，不把代码通过检查写成实验已完成。
