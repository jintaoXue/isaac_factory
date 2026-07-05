Factory production environment by Isaac Sim
---

# Isaac Sim and Isaac Lab
Built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) and [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)

# Asset Description

# Product process

水喉

#3.31 
像素图片的坐标：width=2202 pixels, height=1645 pixels
左上角为原点[0,0]，右下角坐标为[2202, 1645]

isaac sim 坐标：
左上角为[55.02995, -6.10027], 右下角为 [-55.0706, 76.16652]
map x bound 55.02995, -55.0706
map y bound 76.16652, -6.10027

TODO：1. 图像像素和坐标需要进行映射，等代码跑起来了再check
      2. Vector Env 加速训练
      3. 定义逻辑接口

The corresponding large map data files are located in the following directory:
"~/work/Dataset/HC_data/map_data"
map_routes_human.json
map_routes_robot.json

map的machine related work id
gantry的停靠位置ids，以及Articulation和实际global位置的关系
还有storage area
带标记注释的地图


全库固定随机种子，训练和test要分开
hc_env_base里面要修改:
        if self._test:
            # np.random.seed(self.cfg_env_base.train_cfg['params']['seed'])
            np.random.seed(1)

如果是vector_env, 所有的物体，需要先设定好target_pose,然后最后用 apply step
还要设计好state的combine和切片
machines我觉得设计成单独类比较好

['/World/envs/env_0/obj/ConveyorBelt_A09_0_0/Belt']



yellowbase05到10存在偏移，坐标不在中心

robot 的路网点id最好和human是通用的，只是有些会有mask掉
然后route这块需要做precomputing

storage 的meta registration info 是否可以忽略

#4.19 
single env的step函数
vector env的step concate apply change

vec env: hc_vector_env_base.py
self.scene.write_data_to_sim()
self.scene.update(dt=self.physics_dt)

#4.24
发现cube等材料有90旋转，需要去掉

#4.29 
human robot reset 函数要修改


#5.4 重力取消

#海创factory

Aligned with a 4-layer real-time factory operations stack (formal academic/industrial terminology):
- Product Sequencing Agent: determine the optimal production order from the current manufacturing request
- Product Selector Agent: choose which product should be prioritized next for detailed task planning
- Process Task Planning Agent: real-time planning of the next key process task for the selected product 
- Human–Robot Allocation Agent: assign each planned task to the most suitable human, robot, or machine resource for execution

#记录gantry的关节位置和local位置的相对偏移关系

#when to set processed_task_record["task_done"] = True? update ongoing task records

#task excution and route generation

#缩短 多层决策带来的时间增加

#还剩带来的route planner 和material 状态， product_to_storage 的任务描述， material的update_task_availability_mask要确保，current task是完成的


#一定要检查是否是原址引用 

storage 的"robot_parking_areas_ids": [9], 不太对

del task record in task manager

storage 的管理


#route manager 需要节省运算空间，只放在vector env里面生成一个就可以

#6.4 logistic 的goal area id的initialize其实可以完善一下（好像也不用）

##6.5 material 的state update 有问题在processing task完成之后不一定就是next task可能是next processing task

6.6 mask要删掉 route函数要检查

# 7.1
可能每个machine都需要一个摄像头

# 7.2
 1. human的颜色改一下方便分辨
 在Raw USD Properties 改 material:binding
  红色 /obj/HC_factory/Looks/material____________1
  绿色 /obj/HC_factory/Looks/material____________2
  蓝色 /obj/HC_factory/Looks/material____________3

 2. 摄像头的种类和位置还要再设计一下
  
 3. 识别的标签，任务要设计一下
      请你帮我
            1. 设计实验
            2. 推荐问题解决方案
      设定   
            人的状态需要通过env中的相机的图片信息提取。
            相机是有多个的，多视角。每个machine的相机只观测machine周围的环境，还有storage. highrise camera可以观测更广的工厂环境
            人的状态(state information)包括: 
                  state (见cfg_human.py) 具体有 free 和 working_task
                  working_task又包含一系列的subtask (见cfg_process_subtask_gallery.py)
            agv和machine的状态是直接可以获取的
      目标  
            通过图片信息，文本信息（主要是task record, 见task_progress_manager.py等），识别人的working_task progress,也就是human进行到了哪一步
            subtask。比如logistic_for_pipe_cutting中                
            "subtasks": [
                    # human: 0, gantry: 1, machine: 2, robot: 3
                    ["go_to_material", "go_to_material", "wait", "go_to_material"],
                    ["material_on_gantry", "wait", "wait", "wait"],
                    ["control_gantry", "carry_to_robot", "wait", "wait"],
                    ["material_on_robot", "wait", "wait", "wait"],
                    ["go_to_goal_area", "move_to_goal_area", "wait", "carry_to_goal_area"],
                    ["material_on_gantry", "wait", "wait", "wait"],
                    ["control_gantry", "move_to_goal_area", "wait", "wait"],
                    ["material_on_goal_area", "wait", "wait", "done"],
                    ["done", "done", "done", "done"],
                ],
      
      输入数据
            应该是一个带前后帧的文本序列，图像序列
            task_record
            task_gallery
            subtask_gallery
      
      任务分解
            根据human环境着装和帽子的颜色不同，识别human id
            根据文本信息, 图像信息推理(比如task_record task_gallery subtask_gallery)识别subtask the human is doing
            识别subtask是doing or done的状态
      提示
            输入是多模态的，这带来了性能的提升。比如
            # human: 0, gantry: 1, machine: 2, robot: 3
            ["control_gantry", "carry_to_robot", "wait", "wait"],
            因为gantry可以直接获取信号，所以一旦gantry的subtask carry_to_robot一旦是done的状态，控制gantry的human也应该是done的状态

            又比如
            ["go_to_material", "go_to_material", "wait", "go_to_material"],
            这个时候human与其他任务是独立的，只能通过图片信息判断，human是否到达指定的位置


## 7.3 目前来说文字数据有点问题
还有摄像头需要加进去
