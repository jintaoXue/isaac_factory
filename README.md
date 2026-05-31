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