# Baseline 评估指标对齐说明

## 1. 对齐基线

本轮以 `origin/dev_tyx@52e8643` 的 `模型评估指标.md` 和
`PDFormer/factory_bn/train.py` 为唯一评估参考。`dev_tyx` 的当前主模型是
BNPDFormer：PDFormer 编码器加制造瓶颈 occupancy、event 和 cause 任务头。

本轮只改 B2-B5 的评估、阈值选择和 checkpoint 规则，不改变共享 raw、derived、
dataset、episode split 或标签，因此不需要重新采集或重建 dataset。B4 同时修正了
dense GCN 的过平滑问题；B5 补齐标准残差和 LayerNorm。两者仍分别保持 GCN-GRU 与
GAT-GRU 结构。

## 2. 主任务口径

- 输入 30 个 60 秒窗口，预测未来 15 个窗口。
- A.1 只监督 machine、workbench、gantry、AGV；human 和 buffer 只作上下文。
- 真事件是每个工位未来窗内最长的 hot 段，最短 8 窗口。
- `who` 命中要求工位正确；`report` 命中还要求开始时间误差不超过 3 窗口。
- `report_f1` 是正式主指标，occupancy hot 和 IoU event 只作附录。
- validation 扫描
  `0.55,0.60,0.62,0.65,0.68,0.70,0.72,0.75,0.78,0.80,0.82,0.85`。
- 在 `report_precision >= 0.80` 的候选中选择最高 `report_f1`；test 冻结所选阈值。
- checkpoint 可行门同时要求 `report_precision >= 0.80` 和
  `report_recall >= 0.35`。

B3-B5 扫描 `event_will_probability`。B2 没有独立神经事件头，因此逐阈值将未来
occupancy 概率恢复为 `(will, start, duration)`，再使用完全相同的 who/report 公式。

## 3. 必须输出

每个 validation/test split 在 `metrics.json` 中输出：

- `station_report`：who/report P/R/F1、start/duration MAE、ongoing/upcoming 指标、
  事件分母和 `report_threshold_used`；
- `remain`：hot P/R/F1/AP、真实/预测正例率、四类资源分项、`hot_type_hmean`、
  `remain_len_mae`；
- `occupancy_event`：IoU 0.5 的 event P/R/F1 和事件数；
- `cause`：六个过程原因的 per-class recall、`cause_macro_recall`、`cause_acc`、
  只用 train 标签确定的 `cause_majority_acc` 和 `cause_n`。

同名正式指标也提升到 split 顶层，便于和 BNPDFormer 的 `last_metrics.json` 直接聚合；
原有嵌套结构继续作为同一次运行的结构化输出，不引入另一套计算口径。

## 4. 产物与复现

训练产物会在 `config.json`、`best.pt`、`run_summary.json` 中记录 validation 选中的
`event_report_threshold`。`history.csv` 逐 epoch 记录该 epoch 的实际阈值和双门状态。

旧模型产物使用旧的固定阈值和旧 checkpoint 选择规则，不能与新结果混表。共享
`dataset.pt` 可复用，但 B2-B5 需要重新训练，或者在同一 checkpoint 上完整执行一次
validation 阈值选择后再冻结复评 test；正式结果采用重新训练版本。

## 5. 与主模型的边界

BNPDFormer 的分类型解码阈值、ongoing 强制、双 will head、recall lift 和网络 loss 是
主模型能力，不复制到 baseline。baseline 可以有自己的 validation 最优阈值，但标签、
有效 mask、最短段、时间容差和指标公式必须一致。

## 6. 分数与公平性

`report_precision >= 0.80`、`report_recall >= 0.35` 是统一的业务可行门，不是保证每个
baseline 都能达到的最低成绩。未过门时仍保存 validation `report_f1` 最优的诊断
checkpoint，并明确写 `checkpoint_constraint_met=false`；不能通过降低 test 门槛或加入
BNPDFormer 专属解码规则把 baseline 人为调到过线。

公平对照固定共享数据、标签、episode split、监督 mask、validation 阈值扫描、test 冻结
和指标公式。模型容量、学习率、正则化、batch 和训练时长可按结构分别配置，但必须在查看
新 test 结果之前冻结并完整记录。当前固定配置见《统一 Baseline 对照实验执行设计》的
`baseline_fair_v2` 表；后续配置搜索严格执行《Baseline 验证集调优协议》，候选运行不生成
test 指标。
