# Baseline 交接与远端执行指南

更新时间：2026-09-13。先读本文件，不需要把旧对话全部装入上下文。

## 1. 交接状态

**当前最新状态：**vector_gat两次训练、八份冻结诊断和整批独立核验全部完成，训练
pane472305/诊断pane496576均退出0、无相关Python；服务器979c680、tracked clean。
正式upcoming=3/145、2/145，对应父near=1/145、2/145；不采用为稳定提升。
last AP train=0.42392727/0.28651585，validation=0.00816133/0.00865408；两层动态
排序在全部upcoming目标上都有实际使用，但泛化差距仍然存在。整批1262573 bytes，
SHA e90919aeaae88f5bb924324bfc7d36451957f028f7dec0c554ba2e9b7f3285ae；独立终态
核验13191 bytes，SHA 4f5679cb13afedbb8885fa70b1d45ebdaee91fba3fafdc60ee0cc927097a879b。
六数据全量SHA、28当前文件、22旧joint归档成员及八诊断已复核。详见39.8。
真实故障计划/起点与训练支持审计及独立复核也已完成（40.1–40.3），diag pane513374
退出0。验证145个upcoming目标中80个在预测开始至起点窗末出现同工位真实runtime启动；
22个缺少训练节点×scenario独立起点支持，其中10个也有上述runtime启动。这不是不可预测证明。
168个train/validation episode中153个重采样计划：66个新旧函数均可还原、69个仅旧函数、
18个仅现版。区别是生成规则，不是CSV/张量格式；既有冻结数据和28当前模型文件不变。
独立复核7743 bytes，SHA 17bb7349324fe5747736a89c0d9bc84eac542206808b7cae53bba1643c0c1612。
不重跑任何已完成训练/诊断，未用test，原优化目标仍未达成。

用户最新质疑主0.5–0.6与baseline约0.01差距，优先核验主正式指标/权重来源（第41节）。
主文档prefix8 validation up_r=0.633、n_up=98；baseline为145。主文档将远历史前兆头
列在后续未采用的uphist，不能将当前主源码全部路径当作prefix8实际组件。
重新只读遍历主仓库仍无dense产物，主HEAD f322dbf；现有n10旧checkpoint不能替代。
历史状态分类输入审计尚未实现或运行；当前没有新的训练在跑，不能声称已改善。
新增第42节固定权重分层诊断已本地实现，6 tests/4 subtests通过；待服务器3测试及
两模型各8份诊断。只读当前B4 joint_onset/B5 vector_gat，保持原device/batch及阈值。
不得因新诊断而重训或改报警，尚无分层结果。
42.1更新：服务器3测试通过，两模型首份seed42 best validation输出已生成；driver
误将结构化误报字典做数字减法，pane545202/545207均退出1。只修复校验，保留并复用
首两份输出，原诊断源码仍878082e；本地7 tests/6 subtests通过，待服务器单项测试接续。

此前vector_gat B5 seed42训练及四份冻结诊断均已完成并独立核验，
diag pane489190退出0。两层实际动态排序在全部595/145个train/validation upcoming
目标上都有观察，但last AP train=0.42392727、validation=0.00816133；真例概率中位数
0.52487528/0.00154505。新增能力在用，泛化差距仍很大，尚不支持将低召回单独归因于
原标量GAT的邻居排序限制。正式best=3/145（父near=1/145），seed43训练仍在继续。
四诊断汇总536937 bytes，SHA 02c92f54bf9361efdf0a59ce1277c075fdbc1704f44c9d50658f5c7f4385b256；
独立复核4675 bytes，SHA ee18d021d94092f5d3c9c15de8f9f1d7e7b023a3d152d6d58023f0f144c5dc6f。
详见39.6–39.7；seed43已核验至epoch21，实际Python486258/训练pane472305。
自动接续现已启动：diag pane496576、Python496579，日志vectorgat20260913_finish.log，
正在等现有训练正常终态，不重启训练；随后自动做seed43四份诊断及八诊断/两seed/
六数据全量SHA汇总。服务器固定979c680，禁止pull或手动再启动seed43诊断。
接续7505-byte脚本来自f41d3e9；最终汇总如生成，仍须在退出后独立核验pane状态。
未改主仓库、未用test，原目标仍未完成。

此前vector_gat B5 seed42已正常结束，best8/total28，正式P/R/F1为
0.83555556/0.68675799/0.75388471，upcoming=3/145（父near=1/145）。seed43接续训练，
最近观察至epoch3；单seed的小幅变化尚不能认定方法有效。服务器仍固定979c680，禁止pull。
新增只读GAT换序观察0953fb3：两层、全部历史帧/头，比较两查询的共同有效邻居是否有
严格相反的首选。只读实际softmax，不重算或替换预测；零计数不排除低排名邻居换序。
本地13 tests/6 subtests、服务器4 tests及完整16×30×38×27 CUDA输出/RNG一致检查通过，
峰值1441.306640625 MiB。seed42四份冻结诊断已在baseline_dense_diag pane489190启动，
驱动cbcec6b，日志vectorgat20260913_b5s42_diag.log；尚无完成诊断结论。详见39.5。
旧joint和父near结果保留，B4不运行新变体，未用test，原目标尚未完成。

此前B5-only vector_gat修正预检已通过并启动两seed串行训练，服务器固定
979c680。训练pane472305/driver472308 live，首组seed42实际Python472349，已核验至
epoch4；实际vector_additive/near、参数285982、onset/F-beta关闭、test=false，旧joint
归档11成员及六关键文件通过，B4两组和待运行B5 seed43文件不变。日志
vectorgat20260913_train.log，运行中禁止pull或重启。详见39.4，尚无新候选最终指标。
预检17170 bytes，SHA e5f86076c68359f6887f8875507ca256db9199d09ebe12fe1882602aa27de811；
预检pane468293退出0、服务器9项测试通过，GPU峰值约1.62 GiB。失败旧日志已核验归档，
不重复旧训练/诊断，原目标仍未完成。

此前vector_gat已部署服务器dev_xwt@19a4a0c，8项服务器测试通过。
首次真实预检pane462149退出1，失败是旧JSON阈值列表与当前tuple直接比较，未开训。
已确认两seed配置数值完全相同；本地只修正预检表示转换及asdict写出，16 tests/22 subtests
通过，尚待部署/服务器9项测试/预检接续。重用日志前须核验归档，旧模型未动。详见39.3。

此前joint整批已结束，复核服务器7ff759d、两项pane退出0、无相关Python、
完整汇总SHA不变。新增具体架构发现：B5每头标量加法注意力不能按目标状态改变共同
邻居排序；主参考QK点积可以。实际层最小验证通过，但尚非208召回的因果结论。
已本地实现并登记B5-only vector_gat，父near，两seed、原60epoch预算，仅将激活移到
通道求和前，参数/初始化/RNG不变，旧标量分支兼容；15 tests/22 subtests通过。
标签vectorgat20260913，未部署/未开训，须先做真实208及GPU内存预检。详见第39节。
不要在B4启用这个variant，不重复已完成joint或旧搜索；原目标仍未完成。

此前joint_onset四次训练及16份冻结诊断全部正常结束并完成整批核验。
两项pane400178/439500均退出0，无该批训练/诊断Python；服务器7ff759d、tracked clean。
正式upcoming命中0/1/3/2（各145），父onset_aux为0/3/5/2，未取得稳定提升。
四颗last组合AP train为0.234417/0.383150/0.639215/0.598038，validation为
0.014917/0.019378/0.010652/0.011509；联合分支实际生效但泛化差距仍明显。
六数据文件全量SHA、28当前文件及44旧归档成员核验通过。完整汇总
baseline_dense_joint_onset_metrics_20260913.json，2836774 bytes，SHA
c35d4d46a2a45d40f48381662278ea1655c0a1e0a02b8ec053a7018767dcae19。
详见38.10–38.12。UU独立终端已恢复、同一SSH仍在；不重跑本轮或旧搜索，未用test。
当前没有新训练在运行，下一训练尚未登记；泛化差距的具体组件归因尚未解决，目标保持未达成。

此前joint_onset两颗B4的训练及全部8份冻结诊断均正常结束并独立核验。
末轮组合upcoming AP train/validation为0.234417/0.014917、0.383150/0.019378；
onset分支确实参与报警，但泛化差距仍明显，正式best命中仍0/145、1/145。
B5 seed42也已完成并核验：best7/total27，P/R/F1为0.82997763/0.67762557/0.74610357，
upcoming=3/145；seed43 Python424214继续训练，训练pane400178。
B5 seed42四份诊断已接续到pane430931，使用d5670bc Git对象stdin；服务器仍固定7ff759d，
禁止运行中pull或重跑已完成的B4诊断。新核验及组件解释见38.8–38.9，目标尚未达成。

此前joint_onset两颗B4均正常结束并核验，upcoming=0/145、1/145，
F1均值0.74186426；没有优于onset_aux父对照。B5 seed42 Python411056正在训练，
pane400178/driver400181仍live，服务器固定7ff759d。新增观察源码d5670bc已通过服务器
独立进程3项测试；首组best validation已完成，canonical分数/计数及保存阈值复现通过。
该权重onset胜出142/145 upcoming，但保存阈值下新增报警/命中/误报均为0。
后处理曾因report_threshold_used字段不在canonical报告而退出，已验证归档旧驱动/日志，
复用首份结果后接续剩余三份；当前diag pane416164、实际Python416194为best train。
详见38.7。诊断源码仅通过Git对象stdin执行，禁止训练中pull或重复首份诊断。

此前joint_onset首组B4 seed42已正常结束，best1/total11，P/R/F1为
0.85128806/0.66392694/0.74602360，upcoming=0/145，阈值0.65。完整训练快照及
当前七文件/旧F-beta归档11成员独立核验通过；B4 seed43 Python405436正在运行，
训练pane400178/driver400181仍live，服务器保持7ff759d。详见38.6。
新增联合分支只读观察：同一次forward比较原头、onset和组合分数，按保存阈值拆分
新增命中/误报；本地39 tests/9 subtests通过，尚待服务器内存测试与首组四份诊断。
不要在训练中pull；诊断新源码通过固定Git对象stdin执行，模型模块保持7ff759d。

此前joint_onset已在服务器dev_xwt@7ff759d启动四组串行运行，训练pane
400178 live、driver Python400181、首组B4 seed42 Python400222；已核验至epoch6，
实际config与预检完全匹配，旧F-beta归档11成员通过，其余三组仍是原F-beta。当前是
训练中间状态，尚无新候选最终指标。日志jointonset20260913_train.log。真实208修正版预检已经完整
通过，诊断pane396832退出0；62 tests/17 subtests通过。29,298样本中新旧mask gate
变化为零，所有595/145 upcoming均为observed cold；四组参数/RNG/初始化/主损失路径
与onset_aux对照通过。完整预检27178 bytes，SHA
8ba2f21b149e66962bf2421d54faad3d4af7135096ed4875fca678ae8b7562f0。
详见38.4–38.5；不得在运行中pull、不得重启已有进程或重复旧搜索，未评test。

此前joint_onset已部署服务器dev_xwt@9c741c5，服务器61 tests/17 subtests
通过；首次真实208预检pane382787已退出1，尚未开训。失败在类型掩码对比：旧metadata
保存节点索引列表，预检直接与0/1向量比较。须严格转换成相同索引表示，不能跳过检查。
另发现新历史门控仅在ops之后mask会让无效human影响有效machine；本地已改为反归一化
后、ops之前清零无效节点全部通道，契约升级v2，真实影响尚未量化。详见38.3。
本地90 tests/17 subtests通过，但误纳入旧tmp_path测试，自动创建系统临时目录；已告知
用户，不删除这些目录。后续不得整文件运行artifact_reuse，只选择不创建目录的具体测试。
旧预检脚本/日志原地保留，重新使用文件名前须先核验归档；未改主仓库或进行test调参。

event_fbeta四次训练、16份冻结诊断及整批完整核验均已结束，
两项pane均退出0，无对应Python，服务器HEAD1728ca6、tracked clean。四组upcoming
命中0/2/0/0（各145），近窗对照2/2/1/2；平均F1虽升至B4 0.75947、B5 0.76195，
提前召回未改善。五个冻结数据文件及episodes.npz全量SHA已重验，四份旧图归档
各11成员与原导出一致。完整汇总baseline_dense_fbeta_metrics_20260913.json，
2642311 bytes，SHA 3d375d44317934c1e6e0f3670bc435299e6216699ec419989d39a834d10abf28。
详见37.9；不重复任何已完成训练/诊断。后续joint_onset状态以上段为准。

新增第38节只读源码核查：主参考在训练时已组合continue/onset分数，并对组合值施加
主事件损失；baseline旧onset_aux及事后raw-max/独立阈值诊断不等同于这条联合训练
路径。38.1联合路径已通过57 tests/17 subtests；38.2现已登记joint_onset（父对照
onset_aux、两模型两seed、最多60epoch），新增配置/归档边界定向检查16+2 tests通过。
F-beta整批汇总现已完成；joint_onset首次预检失败后已修正、完整预检通过并启动。
注意legacy hist_last_hot有整局平滑前视问题，禁止新增为网络输入；草稿已改为从原30窗
X独立计算event_history_hot，有独立checkpoint契约。真实208预检通过不代表召回改善，不根据
本地测试宣称指标收益，也不将现有协议当作严格在线因果性已经验证。

history_graph_refine四次训练及16份诊断已全部正常结束并核验，
upcoming命中1/1/0/2（各145），未优于near的2/2/1/2，不采用这次新增图层。
完整汇总baseline_dense_postgru_metrics_20260913.json，2654014 bytes，SHA
db6a3e6a004426404820a774c428da3f862805bd94d4312c1db457f2a6271312。
该批次两项pane均曾退出0，完整收尾见36.6–36.7；其后运行版本及新会话以上段为准。
event_fbeta实现07398bb已通过本地62项及服务器独立进程6项检查，现已部署1728ca6并
完成真实预检后开训。不要重跑已完成搜索，不把中间upcoming日志当作最终改进证据。

### 最新用户决策：先固定 208 episode 优化 B4/B5

用户已授权沿用现有 baseline 208 episode，先查明并改善 upcoming 低召回。
主模型 204-episode 包暂不阻塞这一开发实验；正式共同重评仍保留该材料缺口。
首轮注册 `near_precursor`：仅使用原 30 分钟范围内的近窗摘要，B4/B5 各 seed42/43，
原骨干、损失、抽样、评分和选模规则不变，最多 60 epoch。详见长报告第 28 节和 JSON
`upcoming_208_resume`。不要把下方较早的等待主包记录当作新的用户停止指令。

近窗首轮四次训练已全部正常结束；B4 upcoming 均为 2/145，B5 为 1/145、2/145。
B4 best/last 各四份诊断均已结束且权重哈希通过；last train upcoming AP
0.0821/0.1799，validation 仍约 0.0095。独立事件支持诊断已完成：train/validation 的
595/145 个 upcoming 窗口目标对应 299/73 次去重起始事件；未来至起点才出现本地
扰动的目标分别为 300/82，不能据此推定不可预测或骨干上限。详见长报告 28.3。
用户随后要求主模型架构对照，源码及实际图边表核查见第 29 节：两跳未覆盖的目标节点
对为零，不能归因为看不到其他机器；主模型另有 onset 专门监督与占用预测补报路径。
STGNPP 在主参考配置中关闭。同一 best 权重的事件头/hot 头诊断已完成 8/8，
代码 `00d4c6f`，权重/源码/manifest/epoch/样本数核验通过，validation 原指标一致。
固定 hot 阈值额外覆盖的 upcoming 为 0/0/1/0，同时多出 33/16/11/21 个负例越阈值；
不采用直接补报。最新核验两项 tmux 均 dead=1、exit=0，无对应 Python 任务。

下一轮已登记 `onset_aux` / `onsetaux20260912`，以近窗方案为父对照，只加独立 onset
辅助头和按正负类分别取均值的 BCE（系数 1.0，ongoing 排除，缺类贡献零）。辅助头
不参与正式报警；骨干、历史信息、抽样、其他损失、阈值与选模不变。B4/B5 各 seed42/43，
最多 60 epoch，复用原目录。实现 `df9ee0e` 本地 29 tests / 11 subtests、服务器
11 tests 通过。B4 两颗 seed 已完成（best1/5、total11/15），正式 upcoming 为 0/145、
3/145，F1 均值 0.7447745；B4 best/last 的 train/validation 共 8 份诊断已全部核验。
辅助头 last train AP 为 0.282862/0.535091，validation 为 0.031005/0.032333，仍有明显
泛化差距，尚无正式提升。B5 seed42 已完成（best7/total27、upcoming 5/145），
其 best/last 四份诊断已全部完成核验；B5 seed43 也已完成（best5/total25、upcoming
2/145）。四次训练及 16 份 best/last train/validation 诊断均正常结束并核验通过，
整批完整导出为 `baseline_dense_onset_aux_metrics_20260913.json`。B5 seed43 的 last
辅助头 AP train/validation 为 0.658729/0.018469；当前证据仍支持明显泛化差距。
本轮onset训练源码为df9ee0e；当前新训练版本见下方最新状态。
不在训练中 pull 后续文档提交；实际状态必须重新查询，不能重复启动。

远历史真实构建及模型容量核验已通过：近窗/远历史的前 18 维逐元素相同，4 组初始
权重与 RNG 相同，仅后 5 维追加更早历史摘要。`far_precursor` / `farprec20260913`
已在核验完整 onset 导出、四组现有文件哈希及退出状态后启动，复用 baseline_dense_v6，
启动 pane PID=137283，日志 `farprec20260913_train.log`。这次以近窗为父对照，
onset 辅助头关闭。普通模型路径随批次转为 far_precursor；未开始的组仍为 onset_aux，
不能据路径名误认权重来源。每组开始前用 `model_before_farprec20260913.zip` 归档；
两组 B4 均已完成（best1/5、total11/15、upcoming 0/145、2/145），八份诊断已全部
核验并导出为 baseline_dense_far_b4_20260913.json。两颗 seed 的 last AP train 为
0.088649/0.202656，validation 为 0.010517/0.010266，仍有泛化差距。
B5 两组也已完成（best3/5、total23/25、upcoming 均为2/145），训练 pane 已退出0。
四组共16份 far best/last 诊断也已全部正常结束并核验，训练/诊断 pane 均退出0，
B5日志有 DIAG_BATCH_EXIT_CODE=0。完整导出 baseline_dense_far_metrics_20260913.json，
345240 bytes，SHA d3420cc6c4c23ccd4a3d07b514bdc35443e1f8e39e964c0b444ab7cb1994f020。
B5 last AP train为0.184805/0.308020，validation为0.008044/0.008621，仍有泛化差距。
两组 B4 旧 onset
归档各 11 个文件均已核验；新配置确认为 near_far、额外 30 窗、aux 关闭、test 关闭。
继续时须重新核验实时状态，详见第 31 节。

`--compare_onset_report` 已完成 B4 两颗 seed 的四份冻结检查：通过 git show 将
3b0e784 源码送入 stdin，服务器 HEAD/训练文件保持 df9ee0e；本地26项及服务器
内存3项测试通过。Seed42 原阈值下 upcoming 仍0/145、误报135→155；seed43
upcoming 3→52/145，但 precision 0.822→0.238、误报161→2613，不采用该组合。
完整核验与曲线存 baseline_onset_report_b4_20260913.json。该诊断外壳处于Z终态，
tmux未回收退出码；结果完整性已独立核验，不能写成正常退出0。B5四份检查也已正常
退出0并完整核验，整批8/8完成。原阈值下upcoming为5→12/145、2→9/145，但P降至
0.7124/0.7485，均低于0.8；不采用固定raw-max组合。B5完整导出
baseline_onset_report_b5_20260913.json，1171996 bytes，SHA
d1f2519da3448f19262d9a3caf0199f923671786b86bea47197b933ba4111e50。详见第32.5节，
不重复执行这些检查；独立onset校准尚未测试。

时间汇聚单项对照 temporal_attention / timeattn20260913已启动，父对照near，
只将GRU历史均值改为内容注意力加权均值，保留末状态及融合层，新增128参数。原参数、
RNG与首次输出一致；13项定向测试及28项相关检查（11子测试）通过，服务器四组实际
配置预检也通过。源码固定abc6713，日志timeattn20260913_train.log，pane PID188079、
首组B4 seed42已正常完成（best2/total12），P/R/F1为0.80876068/0.69132420/0.74544559，
upcoming仍2/145，与相同seed的near父对照相同；F1微升，不能与两seed均值混比。
B4 seed43也已正常完成（best5/total15），P/R/F1为0.81925134/0.69954338/0.75467980，
upcoming仍2/145。两组实际配置last_attention、near、aux关闭、test关闭，far归档均通过。
B5 seed42/43也已正常完成（best3/5、total23/25），upcoming均2/145，F1分别
0.74873865/0.74776564。四次训练均结束，训练pane退出0，日志末行为TRAIN_BATCH_EXIT_CODE=0；
四组far归档各11成员及实际配置已核验。这次轻量时间汇聚没有获得实质upcoming改善。

首组best/last × train/validation四份诊断已正常退出0，来源/权重/样本核验通过。
Best validation与正式CUDA值相差1个负例报告，数值审计已定位为CPU0.599993与CUDA0.600031
跨过0.60阈值；CUDA重现正式指标，观察hook对全部5439样本输出不变。正式指标仍用原
CUDA值，不声称CPU报告完全相同。审计文件baseline_timeattn_b4s42_numerical_audit20260913.json，
SHA 9172284d23cf648a7079c52faa45f8ab58bdce1c1c66f4ed72fb238ce7185759。详见33.5。

首组last upcoming AP train/validation为0.07918116/0.00968419，仍有泛化差距；best汇聚
接近均值，last upcoming表示改变量中位数约1.4%/1.7%，不是骨干上限证据。
B4 seed43四份诊断也已正常结束并核验；完整B4导出为baseline_dense_timeattn_b4_20260913.json，
257125 bytes，SHA 7030c7505836dbc935f5e8120c0f02cd1a028255077a2ac1ebb2e34956235abf。
Seed43 last AP train/validation为0.19218590/0.01024910。B5 seed42四份诊断已正常退出0并
核验，last AP为0.18110063/0.00801939，best validation正式指标复现通过。
B5 seed43最后四份诊断也已正常结束并核验，日志timeattn20260913_b5s43_diagnose.log，
pane PID220198退出0；last AP train/validation为0.31128931/0.00858045。16/16份全部核验，
完整导出baseline_dense_timeattn_metrics_20260913.json，489113 bytes，SHA
f3f66b75ccae759b36c5e52d9d56867b33d473d20e9f26b29e5f3be137abfe16。两项pane均退出0，
未发现相应baseline Python任务。新增--inspect_temporal_attention观察组件实际
权重和表示改变量，不改预测；本地19项/5子测试及服务器3项检查通过。诊断代码4b48e2b
通过Git对象经stdin执行，模型模块和HEAD保持abc6713。详见第33.2–33.8节，
不重复训练/诊断。训练结果不支持采用本次汇聚候选，但不能外推为全部时间注意力无效。

本次也复核了第21节旧134-episode/v5抽样实验：uniform_control、event4、upcoming4，
B4/B5各seed42/43共12次已完成，因precision/F1代价未采用加权抽样。它不是当前208包
对照，也不是从未尝试的新方向；未启动新抽样搜索。架构归因应区分已有未来起点头、
独立onset监督、时间编码、输入范围与报警融合。当前直接证据仍是判别与泛化不足，
不能宣称GCN/GAT-GRU在结构上无法预测upcoming。

冻结双头独立阈值范围诊断onsetfrontier20260913已完成8/8并逐份核验，外壳退出0。
源码f9cc9aa经Git对象stdin执行，本地25 tests/5 subtests及服务器4项检查通过；底层
模型模块仍abc6713。原归档/成员、分组、排序和完整canonical报告均与onset原结果一致。
Validation在P>=0.8下的最大upcoming命中为3/10/10/7（各/145）；四组P>=0.8/R>=0.7
均无可行点。它是使用真值的事后经验上界，不是部署阈值或正式分数，也不替代独立校准。
仅调现有权重的这两路标量阈值不能解决问题；不外推为骨干上限。完整导出
baseline_onset_frontier_metrics_20260913.json，1092127 bytes，SHA
22c883374b9bc1d2423ddc87a871890f2d417fc28cdfeaef866901a40591d478。两项pane均退出0，
无相应baseline Python，服务器tracked clean。未启动新训练或test，详见第34节。

后续可核查当前208包完整输入相同而未来目标冲突的情况；旧对称性审计只检查旧包
窗口内工位交换，不能代替这项检查。现已实现并登记baseline_input_identity20260913，
本地7项测试及4个前向子测试通过；同时检查全部_model_inputs加hist_last_hot，以及
event_context=false时实际事件分支所用字段。后者排除仅供其他任务头使用的全局/订单
字段及未使用的target_node_mask，避免这些字段掩盖事件分支的输入重复。未来和样本
身份不进入哈希。各口径基础输入若已唯一，可排除加上near/far后的精确重复。
该审计现已完成并核验：两口径train/validation并集29298个输入均唯一，跨split
重复/标签冲突为零；原目标数复现，五个数据文件全量SHA前后一致。无需重建摘要，
因为附加字段不能产生重复。产物baseline_input_identity20260913.json，10334 bytes，
SHA 63b2c5e46cdee806030165fa00c54ab0e6d79e0f2689445f365462454f27288b；源码d85212f，
服务器运行模块仍abc6713，诊断pane PID264887退出0。没有train/test运行。该结果
不排除近似模糊或表示压缩，不能据此宣称模型上限，详见长报告35.1。

下一项已登记history_graph_refine / postgru20260913：在近窗父对照的GRU历史汇聚
后增加一层同类型GCN/GAT残差交换，原初始化输出及RNG保持一致。B4/B5增加
16576/16704参数；仍用原输入、损失、抽样、阈值和选模，aux关闭。两颗seed、最多
60 epoch，复用目录并逐文件验证归档timeattention结果。实现和本地检查已通过；服务器5项测试和四组真实配置/首次输出预检也通过。
现已在371afb6启动四组串行训练，pane PID272151，首个Python272155（B4 seed42），
日志postgru20260913_train.log。首组旧timeattention归档11成员已核验，实际config
为near/last_mean/refine开启/aux关闭/test关闭；首次观察epoch1，不能判断最终收益。
B4/B5总参数289630/302686。服务器保持训练提交371afb6，训练中不pull；其他组
尚未开始，须核验实际进程和权重来源，详见36.1。不能重复已有时间注意力汇聚实验。

新增--inspect_history_graph观察一次原forward中的图残差及表示范数，按事件组
统计其实际更新量，原预测/参数/RNG不变。本地33项及23子测试通过；待每组权重
终态后按best/last × train/validation诊断。源码将经Git对象stdin执行，不pull
运行中的模型模块，不因观察到非零更新就宣称召回改善，详见36.2。

postgru首组B4 seed42已正常完成：best1/total11，P/R/F1为
0.82087912/0.68219178/0.74513716，阈值0.55，upcoming=1/145。四份best/last
train/validation诊断也已完成并核验，源码5cfce2d经stdin执行，诊断pane278553
退出0。Last AP train/validation为0.054607/0.010036；upcoming图残差相对原表示
范数中位数约0.476/0.488，说明路径活跃但尚无收益。完整导出
baseline_dense_postgru_b4s42_metrics_20260913.json，548096 bytes，SHA
034447da244ab18f376471a3c2dd18960dd284a8db76cdf640d91f14607e68b9。
B4 seed43实际Python276207仍在训练，B5两组尚未开始；训练pane272151仍live，
底层模型/HEAD保持371afb6。不要重复首组诊断，不在训练中pull，详见36.3。

postgru的B4两颗seed及八份诊断现已全部完成核验。Seed43 best7/total17、
P/R/F1=0.82637363/0.68675799/0.75012469，upcoming也是1/145，阈值0.70。
Last AP train/validation=0.198485/0.008768。两seed F1均值0.7476309227，
near父对照0.7510203841；upcoming均值0.0068965517，对照0.0137931034，不采用
B4该候选。完整B4导出baseline_dense_postgru_b4_metrics_20260913.json，1334176 bytes，
SHA e8c5f415bc16b823275077166cc6f933b3bd1997b20e354dd7369b145817118b。
诊断pane283768退出0，训练pane272151仍live；B5 seed42 Python282818在运行，
其配置和旧归档11成员已核验，seed43尚未开始。服务器模型仍371afb6，详见36.4。
不要重跑B4已完成训练/诊断，不在训练中pull。

B5 seed42现也已完成，best7/total27，P/R/F1为
0.8359375/0.6840182648/0.7523857358，upcoming为0/145、阈值0.60。四份冻结诊断
已正常退出0（pane295800），来源/权重/样本/分组及原正式指标核验通过，运行模块
仍371afb6。Last AP train/validation为0.422239/0.008441。完整导出
baseline_dense_postgru_b5s42_metrics_20260913.json，548987 bytes，SHA
bddea7fa14a9e665fa96529155c89addf158a1c3cc9d19116ea897c996479334。
B5 seed43实际Python294501和训练pane272151仍live；不要重复首组诊断，详见36.5。

下一项event_fbeta / fbeta20260913已在本地实现并登记，尚未部署训练。它以near为父
对照，只新增主参考配方的all/upcoming soft F-beta项，beta1.5、事件损失内系数0.8，
保留原GCN/GAT-GRU、BCE权重、均匀抽样及评分/选模，不叠加图refine或onset辅助头。
62 tests/10 subtests已通过，包括固定主源码块的数值/梯度对照。当前图批次四次训练、
16份诊断及汇总全部结束并核验后，才部署并做真实数据预检；服务器当前保持371afb6。
本项不是完整主配方对齐，不保证precision约束或召回改善。精确定义和执行顺序见37节。

F-beta实现提交07398bb已push，服务器仅fetch对象并在独立CPU进程载入新源码，
新增六项测试全部通过；HEAD和六个运行文件仍为371afb6，未部署、未加载真实数据。
核验文件baseline_fbeta_server_tests20260913.json，2056 bytes，SHA
8bf745a586d5495429b94a16e20e19f90a41b3d573aa2f27cad908cfcca5b88c。
该项内存测试不能替代批次收尾后的部署/真实数据预检，详见37.1。

### 2026-09-12 新对话续接：输入审计已完成

用户已授权继续优化；下方旧对话停止记录仅用于追溯，不是本轮停止指令。
本轮先通过 UU/Leo SSH 核实 BSTAN 干净 `dev_xwt@6b2b216`，两项 dense 会话均
dead=1、exit=0，未发现相应训练/诊断进程。随后只部署输入审计代码 `2d6bbd0`，
本地及服务器各 3 项边界测试通过。未启动模型训练、未改原 v6 数据或权重。

服务器现有 benchmark 中新增 `baseline_precursor_history_audit_20260912.json`：
train 23859/validation 5439 个样本全部通过样本身份、历史时间范围和前缀不变性检查；
有非零远历史摘要的样本分别为 23721/5409，有完整额外 30 窗历史的分别为 19719/4539。
未读取 test episode 特征数组、未使用标签。全部节点的统计不能当作 upcoming 指标。
详情见长报告 27.11 和 JSON 的 `resume_input_audit_20260912`。

主仓库当前仍为旧 `f322dbf`，其 `raw_data` 和 `libcity/cache/model_cache` 中无
`dense*` 目录。`origin/dev_tyx` 新 fetch 仍为 `20c40e2`；正式 204-episode 包、实际
split、`dense_i1_a1_prefix8/BNPDFormer_best.pt` 和运行配置仍缺现有路径。
已向用户询问这些路径；不要将旧 n10 权重冒充 dense 对照。下一项输入对照只有预案，
尚未开训，也没有宣称共同输入已经对齐。

### 旧对话收尾记录

用户要求完成当前运行、停止旧对话自动优化、整理资料，在新对话继续。
这不是“优化目标已完成”。不应在没有用户继续指令时启动新实验。

- 服务器训练会话 `baseline_dense_v6`：dead=1、exit=0。
- 服务器诊断会话 `baseline_dense_diag`：dead=1、exit=0。
- 收尾时未发现 `train_dense_baseline_control.py`、`train_b2/3/4/5_*`、
  `diagnose_baseline_events.py` 或 baseline 调参 Python 进程。
- 最后权重 train/validation 诊断 8/8 完成，数据、权重哈希、epoch 和样本数核验通过。
- 没有停止其他人的进程，没有清理目录，没有新建仓库或实验目录。
- 运行代码为 `79a5df6`；三分类训练代码为 `33628fc`。交接文档提交在其后，
  最新文档 commit 用 `git --no-pager log -1` 查询，不混作训练代码版本。

## 2. 背景与约束

研究任务是制造工厂瓶颈预测，主实验为同组同学的 BNPDFormer，baseline 为：
B2 XGBoost、B3 LSTM、B4 GCN-GRU、B5 BSTAN-style GAT-GRU。
当前重点提升 B4/B5 的提前预报，并保证主实验与 baseline 可公平比较。
不承诺每种模型达到指定高分，不通过修改事件定义、放宽容差或筛除困难正例造高分。

必须遵守：

1. 只在 `dev_xwt` 改代码；服务器只写 `/home/sci/work/BSTAN_isaac_factory`。
2. `dev_tyx` 和服务器 `/home/sci/work/BNPDFormer/_isaac_factory` 只读参考，不能改动。
3. 不再按 commit 新建目录或 worktree。复用已有目录，允许新增有意义的结果文件。
4. 旧结果用已有归档逻辑及 Git 保留；禁止覆盖未归档权重，禁止自动删除旧目录。
5. 不加无效旧版本兼容路径。二分类/三分类是有实验意义的对照，不是兼容兜底。
6. 调参只看 train/validation。当前 dense 搜索未评 test；不得用 test 决定候选。
7. 每次报告说明数据版本、split、seed、权重来源、阈值和统计方式。
8. 不改服务器其他人的训练、仿真、VPN、SSH 配置；不把登录密码写入文档或 Git。

用户曾询问旧 `BSTAN_v4_*`、`BSTAN_v5_*` 目录是否可迁移删除，但明确没有授权删除。
只读核查发现若干是 detached Git worktree；`BSTAN_baseline_dev_xwt_v5` 是独立仓库。
ignored 数据、依赖和软链接尚未完整核验，不能凭 tracked clean 判断可删除。

## 3. 工作目录

本地仓库：

```text
/Users/xuwantong/Desktop/hku courses/capstone/code/isaac_factory
```

服务器仓库：

```text
/home/sci/work/BSTAN_isaac_factory
```

当前 benchmark（目录名 v3 不等于张量版本）：

```text
/home/sci/work/BSTAN_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3
```

原始数据只读来源：

```text
/home/sci/work/BNPDFormer/_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset
```

同一服务器 `/home/sci/work/isaac_factory` 可能指向主实验仓库；不能误当 baseline 工作目录。

## 4. 数据与评估

| 项目 | 当前 baseline dense v6 |
|---|---|
| 数据版本 | `factory_baseline_dataset_v6` |
| 标签 | `factory_ops_event_30m_to_15m_dense_v2` |
| 评估契约 | `factory_dense_i1_eval_v1` |
| Raw 审计 | 217 个 episode，接纳 208，拒绝 9 |
| Episode split | train/validation/test = 138/30/40 |
| Sample split | 23859/5439/6886，总计 36184 |
| 名义输入/预测 | 30 分钟历史、15 分钟未来，每窗 1 分钟 |
| Train 目标 | ongoing 4191、upcoming 595、negative 297152 |
| Validation 目标 | ongoing 950、upcoming 145、negative 67331 |

目标计数是相关的窗口-工位目标，不是独立物理事件数。全样本共享导出对齐检查通过。
旧 134 个 episode 扩充至 208；不能与旧 v5 的 F1 约 0.3 直接比较来宣称模型提升。

当前 manifest SHA-256：

```text
e3d7b2008ad7c5d0844c10a4c0670ff36c5ba961382706695689daf7a050244f
```

当前计分规则：

- Upcoming：真实 start 为未来索引 1 或 2，持续至少 8 窗。
- Ongoing：真实 start=0，历史末窗 hot 且未来延续，剩余至少 1 窗。
- Report 命中：工位正确，起点误差不超过 3 窗；持续时间不是该命中的硬门槛。
- Precision=命中报告/预测报告；Recall=命中报告/真实事件目标；F1 为调和平均。
- Upcoming recall 单独按真实 start>0 的目标分组，总 P/R/F1 同时包含 ongoing/upcoming。
- 起点/持续时间 MAE 在成功匹配工位的目标上计算，单位分钟；无支持样本为 N/A。
- Remain MAE 在所有样本锚点计算到订单全部完成的剩余时间误差，单位分钟。
- Hot F1 是未来窗口-节点格子的指标；will AP 是全事件排序，不能替代 upcoming AP。
- Validation 阈值搜索先满足 P>=0.80 再比较 F1，否则按 F1；当前可行门为 P>=0.80、R>=0.70。
- 起点容差比 1--2 窗提前范围更宽；upcoming recall 不是严格的提前量准确率。

当前 v6 未评 test，但其中部分历史 episode 曾在旧版开发中被查看；不宣称全新未触碰测试集。

## 5. 当前结果与诊断结论

以下为 validation 的 seed42/43 均值，不是 test，也不是挑单个最大值。

| 模型/候选 | P | R | F1 | Upcoming recall |
|---|---:|---:|---:|---:|
| B4 原 history 控制组 | 0.8078 | 0.6968 | 0.7482 | 0.0103 |
| B5 原 history 控制组 | 0.8298 | 0.6826 | 0.7490 | 0.0138 |
| B4 三分类候选 | 0.8296 | 0.6872 | 0.7515 | 0.0034 |
| B5 三分类候选 | 0.8459 | 0.6785 | 0.7530 | 0.0138 |
| B4 近窗摘要候选 | 0.8221 | 0.6913 | 0.7510 | 0.0138 |
| B5 近窗摘要候选 | 0.8382 | 0.6758 | 0.7482 | 0.0103 |
| B4 onset 辅助监督候选 | 0.8331 | 0.6735 | 0.7448 | 0.0103 |
| B5 onset 辅助监督候选 | 0.8251 | 0.6877 | 0.7500 | 0.0241 |

B4/B5 onset 候选均已完成，表内为原正式事件头结果。独立分支训练时没有参与正式
报警，其诊断 AP 不能替代 report 指标；冻结双头报警的另行检查见上方最新状态。

已完成控制组、图上下文、upcoming 正例权重 4->12、三分类事件头四组对照，
每组两模型各 seed42/43。没有稳定的 upcoming 提升，三分类不提升为正式最优方案。
普通路径正按组从far转为时间汇聚候选，B4两组已完成、B5 seed42正在运行，B5 seed43仍为far。
旧近窗和onset分别保留在model_before_onsetaux20260912.zip和model_before_farprec20260913.zip，
far在每组时间汇聚训练前保存到model_before_timeattn20260913.zip。
必须核验 config 与归档来源，不能按路径猜候选。

新诊断最重要的证据：

| 最后一轮权重：upcoming 类概率 AP | B4 s42 | B4 s43 | B5 s42 | B5 s43 |
|---|---:|---:|---:|---:|
| Train | 0.226719 | 0.275094 | 0.446460 | 0.467045 |
| Validation | 0.035218 | 0.024133 | 0.020829 | 0.011249 |

这是 upcoming 对无事件负例的诊断 AP，不能与全事件 AP 混用。
较早 best.pt 在训练集也弱，但 last.pt 的训练集排序明显改善、验证集没有跟上。
因此不能断言“模型完全学不会”，也不能认为“再多训练几轮就好”。
不得把 last.pt 的训练单项分数拼入 best.pt 的正式结果。

完整 P/R/F1、who、hot、时间误差、各 seed 和负结果见长报告第 24--27 节及
`baseline_dense_v6_20260911.json`；旧 B2/B3 结果未完成当前 v6 公平重跑。

## 6. 主实验差异与下一步目标

2026-09-12 `git fetch origin dev_tyx` 成功，仍为 `20c40e230aedee6aef2429d352413fbcf0fa571a`。
主模型文档记录 `dense_i1_a1_prefix8`，204 episode，split 140/28/36。
其 validation P/R/F1 约 0.883/0.881/0.882、upcoming recall 约 0.633；test 约
0.852/0.856/0.854、upcoming recall 约 0.455。它是续训阶段单个 best checkpoint，
不是多 seed 均值；对应正式完整包/权重尚未找到并完成同口径重评。

新增确认的输入差异：主分支 `PDFormer/factory_bn/remain.py` 定义
`PRECURSOR_FAR_WINDOWS=30`；`dataset.py` 构造编码窗口之前最多 30 分钟摘要；
`model.py::_fuse_precursor` 实际进入事件头。baseline近窗父对照没有这些远历史摘要；
新增far对照已单独加入五维更早历史摘要并完成训练，未见稳定增益。
这是额外过去信息，不是未来泄漏；但“双方基础输入完全一致”尚不成立。
尚不能证明上述文档 checkpoint 使用了非零摘要权重，更不能把全部分差归因于此。

新对话继续时建议按顺序推进：

1. 核对主模型实际数据包、split、所用 checkpoint 与输入配置；204/208 差异不能忽略。
2. 显式约定共同历史信息范围：纯 30 分钟，或统一提供额外历史摘要。
   若扩展输入，应登记新特征契约，不能改数据后仍冒称原 v6 完全等价。
3. 在共同输入下做单因素对照，保留 GCN/GAT-GRU 骨干，不机械复制主模型专用解码器。
4. 同时报 train/validation 的 upcoming AP、recall、误报与 MAE，优先处理泛化差距，
   不再重复已无收益的权重/续训盲搜。
5. 获得稳定方案后再冻结多 seed 验证、共同数据和正式 test 协议，补齐 B2/B3。

还需共同讨论历史 hot 用完整 episode 平滑可能导致的前视信息问题；当前只记录，
未擅自改变共同标签。输入对齐不等于所有数据质量问题已经解决。

## 7. 结果在哪里

以下均相对第 3 节 benchmark，不创建新目录：

```text
models/tuning/b4_representation_v1/candidate_history/seed42/
models/tuning/b4_representation_v1/candidate_history/seed43/
models/tuning/b5_representation_v1/candidate_history/seed42/
models/tuning/b5_representation_v1/candidate_history/seed43/
```

每个目录中：

| 文件 | 内容 |
|---|---|
| `model_before_upcontext20260911.zip` | 原 history 控制组 |
| `model_before_upweight20260912.zip` | 图上下文候选 |
| `model_before_upclass20260912.zip` | upcoming 加权候选 |
| `model_before_nearprec20260912.zip` | 三分类候选 |
| `best.pt` / `last.pt` | 最新候选；当前开训前为近窗，后续以 config/运行记录为准 |

benchmark 根目录有四份完整 metrics 汇总：
`baseline_dense_control_metrics_20260911.json`、`baseline_dense_context_metrics_20260912.json`、
`baseline_dense_weighted_metrics_20260912.json`、`baseline_dense_three_class_metrics_20260912.json`，
另有 `baseline_dense_near_metrics_20260912.json`（四次近窗完整结果及文件哈希）。
`baseline_dense_onset_aux_b4_20260912.json` 保存已完成的 B4 两次完整结果、配置、文件
哈希及八份诊断摘要；整批四次与 16 份诊断已另存
`baseline_dense_onset_aux_metrics_20260913.json`，不能把较早 B4 导出当作四次汇总。
诊断文件模式 `b[45]_seed4[23]_{train|validation}_diagnostics_<tag>.json`，标签有
`upcoming20260911`、`upcontext20260912`、`upweight20260912`、`upclass20260912`、
`upclasslast20260912`。精确完整指标保存在服务器，Git 中为审计与摘要记录。

## 8. 通过 UU 操控远端的实际步骤

### 8.1 连接链路

```text
Codex 所在本机 -> UU远程 -> Leooo的MacBook Air 的终端 -> ssh my_sci -> sci@sci 服务器
```

Leo 端连港大 VPN。本机不能代替 Leo 直连服务器；不要反复用本机 SSH 证明不可达。
历史上试过端口映射，但当前已验证可用的是 UU 终端内 SSH，不依赖本机 2222。
SSH 认证由用户提供或已有会话完成；不在本文件保存密码。

### 8.2 新对话初始化 CUA

使用 `mcp__cua_repl.js`，首次或 reset 后只执行入口调用，先读返回的 API 文档：

```javascript
var uu = await cua.getApp('com.netease.uuremote');
```

再读实际终端画面：

```javascript
await uu.getAXStateAndScreenshot();
```

必须确认窗口为 `终端 - Leooo的MacBook Air`、AX 中有 `ID: TerminalWindow`。
SSH 登录后画面应为 `sci@sci`；若是 `leooo@...`，先在该终端执行 `ssh my_sci`。
如果显示的是 UU 首页/远控桌面，用最新 AX 定位“终端”入口；不要在错误窗口粘贴命令。
实测可从“窗口”菜单的“网易UU远程”回到设备页，确认Leooo设备后点击“终端”恢复
原SSH会话。远控顶部“+”用于创建虚拟显示屏，不是终端入口。本次误开的虚拟屏已
通过该标签右键“删除虚拟屏”撤销，原单显示屏状态已核验恢复。

可定义短命令助手：

```javascript
async function terminalCommand(command) {
  const state = await uu.getAXState({emit:false, disableDiffing:true});
  if (!state.includes('ID: TerminalWindow')) {
    throw new Error('UU terminal not selected; nothing sent');
  }
  await uu.paste(command, {format:'text'});
  await uu.pressKey('Return');
  await uu.getAXStateAndScreenshot();
}
```

终端输出主要通过截图读取。每次只发短命令；长 heredoc 曾发生粘贴不完整。
截图尚未显示执行结果时，先重新读画面，不要重复启动训练。
确认只是未提交完的只读命令时可用 `pressKey('ctrl+c')` 取消；不要对训练终端乱发中断。
API 键名使用小写 xdotool 风格，`Ctrl+C`/`Control+c` 不适用。
Git 查看用 `git --no-pager ...`；若进入 `(END)`，发 `q`，UU 终端有时还需 Return。
遇 `native pipe closed`，重新获取 app/状态，先确认命令是否执行；不要据此认定远端任务停止。

### 8.3 登录后的只读前置检查

在服务器终端逐条执行：

```bash
builtin cd "$HOME/work/BSTAN_isaac_factory"
git status --short --branch
git --no-pager log -1 --oneline
tmux display-message -p -t baseline_dense_v6 'dead=#{pane_dead} exit=#{pane_dead_status}'
tmux display-message -p -t baseline_dense_diag 'dead=#{pane_dead} exit=#{pane_dead_status}'
pgrep -af '[t]rain_dense_baseline_control.py|[d]iagnose_baseline_events.py|[t]rain_b[2345]_'
nvidia-smi
```

较早交接时两个会话均正常退出；最新运行状态见第 1 节。新对话仍需重新检查，不能依赖
旧 PID 或锁文件。若 pane_dead=1 但退出码为空，要结合 ps 状态、实际 Python 进程
与日志核实；本轮曾出现已结束的 Z 状态外壳尚未被 tmux 回收。不得仅凭空退出码认定
仍在运行或正常退出。新 B5 诊断另写 DIAG_BATCH_EXIT_CODE 到日志。
其他人的仿真可能占 GPU，不得终止。没有匹配进程时 `pgrep` 返回码 1 是正常情况。

### 8.4 Python 环境

服务器曾遇 Isaac Sim `PYTHONHOME/PYTHONPATH` 引起 SRE mismatch，以及清掉路径后
torch 不可导入。已经成功使用的环境恢复方式如下，仍需实际核验：

```bash
unset PYTHONHOME
export PYTHON=/home/sci/repos/miniconda3/envs/env_isaaclab/bin/python
export PYTHONPATH="$(tmux show-environment -g PYTHONPATH | cut -d= -f2-)"
export PYTHONDONTWRITEBYTECODE=1
export DENSE_DIR="$PWD/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/bottleneck_dataset/experiments/factory_pdformer_134_v3"
export TOOLS="$PWD/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools"
"$PYTHON" -c 'import sys,torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())'
```

曾验证 torch 2.7.0+cu128。若 tmux 没有保存 PYTHONPATH，先检查实际环境，不凭印象重装。
仅需 JSON/文件读取时用 `/usr/bin/python3 -I`，隔离污染路径。

### 8.5 修改、部署与实验

- 本地先确认 `dev_xwt`、读取 dirty diff，用 `apply_patch` 改动；保留用户修改。
- 测试后提交并 push `origin dev_xwt`。服务器确认无运行任务、tracked clean 后
  `git pull --ff-only origin dev_xwt`，核验 commit。不要在训练期间更新训练模块。
- 当前入口为 `batch_factory_baseline_dense.sh`；训练实现为
  `tools/train_dense_baseline_control.py`，诊断为 `tools/diagnose_baseline_events.py`。
- 用户已授权 208-episode B4/B5 受控优化。先查进程和已完成标签，再设置唯一
  `ARCHIVE_TAG`、`TRAIN_VARIANT`、`TRAIN_SEEDS`、`DEVICE` 并复用已有 tmux/模型目录。
- `ARCHIVE_TAG` 唯一且旧权重先归档；不创建新源码副本。不得重用已存在输出标签。
- 诊断 ZIP 可用 `--archive_member best.pt` 直接验证读取，不解压覆盖当前权重。
- 只对确认已退出的 pane 使用 respawn；正在运行时等待。SSH 断开不等于 tmux 任务停止。
- 避免终端一次打印完整大型 JSON；提取关键字段，完整结果文件留在服务器。

本地测试解释器为 `/tmp/factory-baseline-tests/bin/python`（临时路径，新对话要查是否仍存在）。
较早收尾验证 152 tests / 15 subtests；新 onset 实现的验证记录在第 30 节，不能混用计数。

## 9. 给新对话的起始提示

> 请先读 `doc/experiments/Baseline交接与远端执行指南.md` 和
> `doc/experiments/baseline_dense_v6_20260911.json`，按需查长报告第 27 节。
> 继续优化 B4/B5 upcoming，同时保持与 BNPDFormer 的公平对照。
> 先核验服务器现状和共同历史输入范围，不立即重复已完成搜索。
> 只在 dev_xwt、服务器 BSTAN_isaac_factory 内修改；通过 UU 的 Leo 终端 SSH 操作，
> 不新建目录、不删旧目录、不动主实验仓库、不用 test 调参。
> 近窗及 onset 辅助监督已完成，仍有训练/验证泛化差距；远历史输入对照正在推进。
> 先查第 1 节和服务器实时进程，避免重复启动已完成或正在运行的任务。
