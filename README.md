Isaac Production

---

# Isaac Production
Isaac Production is a training platform for human-robot task allocation in manufacting. Built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) and [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)

## after deploying isaac-lab and create conda env
in isaac-production folder, link the isaac-sim repository:
ln -s ${HOME}/isaacsim _isaac_sim
pip install heapdict


# env part description
Part1 上料输送轨道+激光除锈工位+下料输送轨道
Part2 法兰料架x2，左+右
Part3 固定物体
Part4 龙门架静态
Part5 运料框
Part6 固定物体 气罐
Part7 龙门架，下料抓手
Part8 龙门架中间放料区
Part9 上料抓手 
Part10 激光切料区
Part11 激光焊接区


# todo
gantt chart
argparse

self.max_episode_length 是否可以修改? (no)

wandb 

修复自动补全

self.scheduler lr_scheduler schedule_type (请参考~/Repos/miniconda3/envs/isaac-lab/lib/python3.10/site-packages/rl_games/common/a2c_common.py)

env reset fialed: 解决办法，在gym register 部分order_enforce=False

test setting的num robot character 不一定对

## human fatigue 建模

## 加一个movement time 把map_route 函数加一下

## 加一个predict task max fatigue

## 修正一下task的持续时间 以及机器控制

## 如何修改代码
1.rule-based只用fatigue model的predcit 函数给到predict的结果，的到worker的task mask分别用于high-level decison 和low-level decision
2.基于cost function的，这个cost function 会收集所有同质/异质worker的生产状态，或者基于/结合粒子滤波，cost function的输出呢还是high-level task space，不过会有多个worker
3.在做决策的时候用综合的mask，mask掉不安全的输出

### 网络设计

### 下一步要改的细节
    1.如果疲劳程度超过1, 工人不管在做什么subtask任务，都会变成休息（会使问题，难度更大
    2.cost function预测的具体是什么，是完成某一个task的fatigue增加量
    3.cost function的loss函数
        可以分为连续性监督和离散型监督，加上action，一共三个loss函数
    4.cost function的数据收集工作（可以先用仿真器训练一个）
        task_clearing 是收集结束状态， assign_task 是开始状态
    5. 如果是异质的工人，怎么进行训练
 6. code evaluation epoch的加入
 7. fatigue的值验证
 8. fatigue 的曲线调整，以及训练的reward设计，env length调整
 9. worker 要改成异质的，首先就是要把疲劳参数和step函数做一个修改，然后是网络参数的输入要改，异质worker的初始化方式也要修改，奖励函数和环境的长度也要做修改


# 问题，supervise traning 存在过拟合

fix bug
神经网络修正
对于None action的选择

把预测值改成预测delta

# 5.4
wandb上数据的关键变化点可以记录在table里面

action的返回值里面再包括额外的信息
obs的extra也应该要包含返回的额外信息

# debug一下为什么单个worker的decision这么差
self.task_manager.characters.acti_num_charc
1
self.task_mask[1:]
tensor([1., 1., 0., 0., 0., 0., 1., 0., 0.])
q
tensor([[ -6.2242,  -6.2513,  -6.4300, -20.0000, -20.0000, -20.0000, -20.0000,
          -6.5274, -20.0000, -20.0000]], device='cuda:0')
action_mask
tensor([[1., 1., 1., 0., 0., 0., 0., 1., 0., 0.]], device='cuda:0')

# 调小 box capacity
hoop 为4
但是bending tube为2试试
num product为2

问题：机器人数量增加，效率反而下降了


# 6.16
要实现这个PPO-lag discrete的话就要修改储存的memory，修改网络结构，计算adv surrogate要有区别
要么就用PPO-penatly

还要把low-level改成nearest path的方式


# 6.19 各算法对比
PPO
https://arxiv.org/abs/1707.06347

ppolag_dis有两个版本：  
   1. 没有cost
   2. 直接penalty in reward，
   3. 用cost critic, 用lagrangian结合
   4. cost_mask (by cost critic, by task predicticve neural, by filter)

ppolag_filter通过predictive的方式，加上cost mask作为硬约束

EBQ同样也是，考虑加mask和不加mask的区别
    1. 没有cost
    2. cost penalty in reward
    3. cost critic penalty in critic
    4. cost_mask (by cost critic, by task predicticve neural, by filter)


setting3:
    fatigue coe是已知的，输入给网络
    fatigue coe是未知的，用rl filter


## 6.21 决定还是用rl filter的方式
对比算法实现

能确保算法训练符合故事

EBQ 先去掉cost 约束试试
    if done_flag[0]:

### 7.2 记得保存训练模型
fatigue 的参数值 完成时间要好好调整一下

## 7.10
%图片2 的红字对齐要改，ppo图片要改，序号不对，
comparison algorithm怎么设计？分为加不加safe set 还是说直接对比性能就好了


## 7.16
RL sutton 的书133页 + DDQN的 原文
https://arxiv.org/pdf/1509.06461


        {
            "name": "test: rl_filter filter headless wandb",
            "type": "python",
            "request": "launch",
            "args" : ["--task", "Isaac-TaskAllocation-Direct-v1", "--algo", "rl_filter", "--headless", "--wandb_activate", "True", "--test", "True", "--load_dir", 
            "/rl_filter_2025-07-20_12-17-12/nn", "--load_name", "/HRTA_direct_ep_82400.pth", "--wandb_project", "test_HRTA_fatigue", "--test_times", "10"],
            "program": "${workspaceFolder}/train.py",
            "console": "integratedTerminal",
            "justMyCode": true,
        },




#### 算法记录
对比算法：

load_dir_list=(
nomask_4090_rl_filter_2025-07-25_15-02-16
penalty_4070_rl_filter_2025-07-29_22-22-18
4070_rl_filter_2025-07-20_12-17-12
mask_penalty_4090_rl_filter_2025-07-27_14-41-12
penalty_4070_dqn_2025-07-27_11-39-32
4090_dqn_2025-07-29_13-21-06
4070_penalty_ppo_dis_2025-07-31_13-37-58
4090_ppo_dis_2025-07-30_13-18-07
nomask_4070_ppolag_filter_dis_2025-07-23_22-24-04
4070_ppolag_filter_dis_2025-07-21_23-34-32
)


**DQN系列：**
1. D3QN 
    nomask_4090_rl_filter_2025-07-25_15-02-16

2. D3QN with penalty 
    penalty_4070_rl_filter_2025-07-29_22-22-18
3. PF-CD3Q
    4070_rl_filter_2025-07-20_12-17-12

    3.2
    mask_penalty_4090_rl_filter_2025-07-27_14-41-12

4. DQN (先不用做实验)

5. DQN with penalty
    penalty_4070_dqn_2025-07-27_11-39-32

6. PF-DQN
    4090_dqn_2025-07-29_13-21-06

**PPO系列：**
7. PPO-dis (先不用做实验)

8. PPO-dis with penalty
    4070_penalty_ppo_dis_2025-07-31_13-37-58
9. PF-PPO-dis
    4090_ppo_dis_2025-07-30_13-18-07
10. PPO-lag
    nomask_4070_ppolag_filter_dis_2025-07-23_22-24-04

11. PF-PPO-lag
    4070_ppolag_filter_dis_2025-07-21_23-34-32


对比网络：
1. PF-CD3Q 
2. 网络不加Fatigue的信息

对比filter的精确度
EKF
KF
PF

对比setting:
测试参数实时变化

程序的截止时间要改一下


PPO
https://arxiv.org/abs/1707.06347

ppolag_dis有两个版本：  
   1. 没有cost
   2. 直接penalty in reward，
   3. 用cost critic, 用lagrangian结合
   4. cost_mask (by cost critic, by task predicticve neural, by filter)

ppolag_filter通过predictive的方式，加上cost mask作为硬约束

EBQ同样也是，考虑加mask和不加mask的区别
    1. 没有cost
    2. cost penalty in reward
    3. cost critic penalty in critic
    4. cost_mask (by cost critic, by task predicticve neural, by filter)


setting3:
    fatigue coe是已知的，输入给网络
    fatigue coe是未知的，用rl filter


# TODO
训练轮次可以减少（已解决
DQN 的cost function penalty 还没写 (已加)
纯PPO 算法也要加进来， cost penalty也要加
现在要重新画实验数据的图，不能太雷同
想一想实验结果怎么表达，比如新加入箱型图
然后怎么记录这些实验的数据，模型权重，画图
KF的代码是否需要写 （已写）

第一张problem description的大小写也不对
方法那张图的大小写不对

filter图的mu 写成了lambda

### 可视化 
1.先把一个滤波器的画图所需所有参数做好接口 传给这个fatigue去画图，然后再变成多个滤波器的
2.展示结果的类型分为总的曲线变化对比，包括真值和预测值（分为one-step, multi-step）


filter：1.filter 曲线 2.箱线图
training： 四子图
testing: 1. makespan, progress, overwork, (包括箱线图，还有这个表格整理) 2.human变化，overwork和makespan的变化
task allocation result：low-level 暂定
case study：D3QN和PF-CD3Q，上半部分是fatigue曲线，下半部分是甘特图 

### mount bug fix

显示分区：  df -Th 
最后执行：mount -o remount -w /factory (factory为文件夹所在分区名，这里替换成 /dev/sdd4）即可


### 8.9重新跑实验
1.新训练了ppo-lag，所有test重跑
2.改良了pf filter的性能
3.可视化的所有图片要重新运行
4.然后这个task allocation这个概念要怎么改一下？，论文的spatial-aware的部分怎么展示？
5.实验1和4选一个的话，可能选1更好（暂不）
6.filter可以加一个可视化结果，human数量增加，精度下降（暂不，尤其是human=1的时候PF的结果最好
7. PF-CD3QP可以不展示（执行）

下一步，test 3找一个好的结果，重新画曲线

修改gitignore, 把figs文件的东西都记录一下


# 10.9 一审结果
改变dk做sensitivity study
加一个参数表格 以及添加算法RCPO
网络结构的 ablation
添加human的类型

# 10.11 
actor zero grad的顺序是否会影响性能？

action prob 好像有问题

修改方法：把函数接口全部改成parmeter list

https://zhuanlan.zhihu.com/p/700607830

#10.19
fatigue measurement noise的 这里可能还是要加一个zero shot performance的实验

#10.21
加一个sensitive的实验
关于这个cost limit

#11.25
正在跑sensitivity的实验
现在需要加上关于model的ablation study

#11.29
test times 的区别也要说明

#11.30
rl_filter_no_dueling_2025-11-29_18-11-35  4070

rl_filter_selfattn_2025-11-26_16-56-20 4070

rl_filter_no_noisy_2025-11-25_15-04-29 server

#11.2
mlp 3279383 -> 3.1M  3.1274633407592773
param.element_size()
sum([param.nelement() for param in self.online_net.parameters()])
sum([param.nelement()*param.element_size() for param in self.online_net.parameters()])


dqn 9357332 -> 8.9M 9357332


PF-CD3Q -> 9883670 9.4M 9.425802230834961

cost_param_dict = {'transformer.cost_decoder', 'transformer.cost_tgt_embed', 'transformer.cost_projection_layer'}

PPO 13369882 12.8M 12.750513076782227

PPO-Lag 16856094 16.1M  16.075223922729492

cost_training_dict = {
    'transformer.cost_decoder',
    'transformer.projection_layer_cost', 
    'fc_h_v_cost', 'fc_h_a_cost', 'fc_z_v_cost', 'fc_z_a_cost'
}
