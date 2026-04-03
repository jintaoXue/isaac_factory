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
还有storage area
带标记注释的地图