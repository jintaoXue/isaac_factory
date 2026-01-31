


import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pickle
from ...utils.hybridAstar import hybridAStar
from ...utils import quaternion

def world_pose_to_navigation_pose(world_pose):
    position, orientation = world_pose[0][0].cpu().numpy(), world_pose[1][0].cpu().numpy()
    euler_angles = quaternion.quaternionToEulerAngles(orientation)
    nav_pose = [position[0], position[1], euler_angles[2]]
    return nav_pose

class MapRoute(object):
    
    def __init__(self, env_cfg):
        self.route_character_file_path = env_cfg.route_character_file_path
        self.route_agv_file_path = env_cfg.route_agv_file_path
        self.occupancy_map_path = env_cfg.occupancy_map_path
        self.cuda_device = torch.device(env_cfg.cuda_device_str)
        return
    
    def load_pre_def_routes(self):
        character_route_dic = None
        agv_route_dic = None
        sampling_flag = True 
        with open(os.path.expanduser(self.route_character_file_path), 'rb') as f:
            dic = pickle.load(f)
            if sampling_flag:
                character_route_dic = self.routes_down_sampling(dic, to_cuda=False, route_type='human')
            else:
                character_route_dic = dic
        with open(os.path.expanduser(self.route_agv_file_path), 'rb') as f:
            dic = pickle.load(f)
            if sampling_flag:
                agv_route_dic = self.routes_down_sampling(dic, to_cuda=False, route_type='robot')
            else:
                agv_route_dic = dic

        return character_route_dic, agv_route_dic

    def generate_and_save_route(self, human_pose_dic, agv_pose_dic):

        have_problem_routes_character = {
            'put_hoop_into_box':{'hoop_loading_inner':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]], 'hoop_loading_outer':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]],
                'bending_tube_loading_inner':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'bending_tube_loading_outer':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]],
                'cutting_cube': [[-22.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]],  'initial_pose_1': [[-22.0, 14.0, np.deg2rad(180)], [-32.0, 12.0, np.deg2rad(-90)]]}, 
            'put_bending_tube_into_box':{'hoop_loading_inner':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]], 'hoop_loading_outer':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]],
                'bending_tube_loading_inner':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'bending_tube_loading_outer':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]],                             
                'cutting_cube': [[-22.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]],  'initial_pose_1': [[-22.0, 14.0, np.deg2rad(180)], [-32.0, 12.0, np.deg2rad(-90)]]}, 
            'put_hoop_on_table': {'put_bending_tube_on_table':[[-12.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)]], 
                'hoop_loading_inner':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]], 'hoop_loading_outer':[[-12.0, 7.4, np.deg2rad(180)], [-16.26241, 7.4, np.deg2rad(180)]],
                'bending_tube_loading_inner':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'bending_tube_loading_outer':[[-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]],                             
                'cutting_cube': [[-22.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]],  'initial_pose_1': [[-22.0, 14.0, np.deg2rad(180)], [-32.0, 12.0, np.deg2rad(-90)]]},
            'put_bending_tube_on_table':{'put_hoop_on_table':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]],
                'hoop_loading_inner': [[-28.7, 12.0, np.deg2rad(45)],  [-22.0, 14.0, 0], [-16.26, 14.0, np.deg2rad(-90)]], 'hoop_loading_outer': [[-28.7, 12.0, np.deg2rad(45)],  [-22.0, 14.0, 0], [-16.26, 14.0, np.deg2rad(-90)]],
                'cutting_cube': [[-30.0, 7.0, np.deg2rad(-90)]], 'initial_pose_0': [[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-11.58, 10.0, np.deg2rad(-90)]]},
            'hoop_loading_inner': {'put_hoop_into_box': [[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'put_bending_tube_into_box':[[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 
                'put_hoop_on_table':[[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'put_bending_tube_on_table':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)]],
                'bending_tube_loading_inner':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'bending_tube_loading_outer': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]],
                'cutting_cube': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]], 'placing_product':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)]],
                'initial_pose_0': [[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'initial_pose_1': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-32.0, 12.0, np.deg2rad(-90)]]},
            'bending_tube_loading_inner': {'put_hoop_into_box':[[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 'put_bending_tube_into_box': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 
                'put_hoop_on_table': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]], 'hoop_loading_inner': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]],
                'hoop_loading_outer': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]], 'initial_pose_0': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]]},
            'hoop_loading_outer': {'put_hoop_into_box': [[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'put_bending_tube_into_box':[[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 
                'put_hoop_on_table':[[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'put_bending_tube_on_table':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)]],
                'bending_tube_loading_inner':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'bending_tube_loading_outer': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]],
                'cutting_cube': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]], 'placing_product':[[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)]],
                'initial_pose_0': [[-16.0, 7.2, np.deg2rad(0)], [-12.0, 7.4, np.deg2rad(0)]], 'initial_pose_1': [[-16.26, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-32.0, 12.0, np.deg2rad(-90)]]},
            'bending_tube_loading_outer': {'put_hoop_into_box':[[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 'put_bending_tube_into_box': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 
                'put_hoop_on_table': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]], 'hoop_loading_inner': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]],
                'hoop_loading_outer': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]], 'initial_pose_0': [[-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]]},
            'cutting_cube': {'put_hoop_into_box':[[-30, 10, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 'put_bending_tube_into_box': [[-30, 10, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]],
                'put_hoop_on_table': [[-30, 10, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]], 'put_bending_tube_on_table': [[-30, 8, np.deg2rad(90)]],
                'hoop_loading_inner':[[-30, 10, np.deg2rad(90)], [-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]], 'hoop_loading_outer':[[-30, 10, np.deg2rad(90)], [-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)],  [-16.26, 14.0, np.deg2rad(-90)]],
                'initial_pose_0':[[-30, 10, np.deg2rad(90)], [-26.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(-90)]], 'initial_pose_1':[[-30, 8, np.deg2rad(90)]]},
            'placing_product': {'put_hoop_into_box': [[-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 'put_bending_tube_into_box':[[-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 
                'put_hoop_on_table':[[-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(0)]], 'initial_pose_0': [[-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(0)]],                
                'hoop_loading_inner':[[-22.0, 14.0, np.deg2rad(0)], [-16.26, 14.0, np.deg2rad(-90)]], 'hoop_loading_outer':[[-22.0, 14.0, np.deg2rad(0)], [-16.26, 14.0, np.deg2rad(-90)]]},
            'initial_pose_0' : {'put_bending_tube_on_table': [[-11.58, 10.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-28.7, 12.0, np.deg2rad(-135)]], 'hoop_loading_inner':[[-12.0, 7.4, np.deg2rad(180)], [-16.0, 7.2, np.deg2rad(180)]],  
                'bending_tube_loading_inner':[[-12.0, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'hoop_loading_outer':[[-12.0, 7.4, np.deg2rad(180)], [-16.0, 7.2, np.deg2rad(180)]], 
                'bending_tube_loading_outer':[[-12.0, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-26.0, 14.0, np.deg2rad(180)]], 'cutting_cube':[[-12.0, 14.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)] , [-26.0, 14.0, np.deg2rad(180)], [-30, 10, np.deg2rad(-90)]], 
                'placing_product': [[-12.0, 14.0, np.deg2rad(0)], [-22.0, 14.0, np.deg2rad(0)]], 'initial_pose_1':[[-11.58, 10.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(180)], [-28.7, 12.0, np.deg2rad(-135)]]},
            'initial_pose_1': {'put_hoop_into_box': [[-30.0, 12.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]], 'put_bending_tube_into_box':[[-30.0, 12.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.0, 14.0, np.deg2rad(0)]],
                              'put_hoop_on_table':[[-30.0, 12.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-12.0, 14.0, np.deg2rad(0)]], 'hoop_loading_inner':[[-30.0, 12.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.26, 14.0, np.deg2rad(-90)]],
                              'hoop_loading_outer':[[-30.0, 12.0, np.deg2rad(90)], [-22.0, 14.0, np.deg2rad(0)], [-16.26, 14.0, np.deg2rad(-90)]], 'cutting_cube':[[-30, 8, np.deg2rad(-90)]],
                               'initial_pose_0': [[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, np.deg2rad(0)], [-11.58, 10.0, np.deg2rad(-90)]]}
            }
        have_problem_routes_agv = {
            'carry_box_to_hoop_table':
                {'carry_box_to_bending_tube_table': [[-12.0, 14.0, np.deg2rad(180)], [-22.0, 14.0, np.deg2rad(180)]], 'placing_product': [[-12.0, 14.0, np.deg2rad(180)], [-22.0, 14.0, np.deg2rad(180)]]},
            'carry_box_to_bending_tube_table': {'carry_box_to_hoop_table':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]], 
                'initial_pose_0':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]], 'initial_pose_1':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]],
                'initial_box_pose_0':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]], 'initial_box_pose_1':[[-28.7, 12.0, np.deg2rad(45)], [-22.0, 14.0, 0], [-12.4, 14.0, 0]]},
            'placing_product':{'carry_box_to_hoop_table':[[-22.0, 14.0, 0], [-12.4, 14.0, 0]], 
                'initial_pose_0':[[-22.0, 14.0, 0], [-12.4, 14.0, 0]], 'initial_pose_1':[[-22.0, 14.0, 0], [-12.4, 14.0, 0]],
                'initial_box_pose_0':[[-22.0, 14.0, 0], [-12.4, 14.0, 0]], 'initial_box_pose_1':[[-22.0, 14.0, 0], [-12.4, 14.0, 0]]},
        }

        self.xyResolution = 5
        self.obstacleX, self.obstacleY = hybridAStar.map_png(self.xyResolution, self.occupancy_map_path)
        self.planning_mid_point = [140, 220, 0]
        self.mapParameters = hybridAStar.calculateMapParameters(self.obstacleX, self.obstacleY, self.xyResolution, np.deg2rad(15.0))
        character_route_dic = self.generate_routes_helper(human_pose_dic, os.path.expanduser(self.route_character_file_path), have_problem_routes_character)
        agv_route_dic = self.generate_routes_helper(agv_pose_dic, os.path.expanduser(self.route_agv_file_path), have_problem_routes_agv)

        return character_route_dic, agv_route_dic

    def routes_down_sampling(self, routes_dic, to_cuda, route_type): 
        # min_len = 2e32
        # max_len = -1
        # self.max_x = -100
        # self.min_x = 100
        # self.max_y = -100
        # self.min_y = 100
        for (key, route_dic) in routes_dic.items():
            for (_key, route) in route_dic.items():
                if _key == 'placing_product' and route_type == 'robot':
                    continue
                else:
                    route_dic[_key] = self.down_sampling_helper(route, to_cuda, self.cuda_device, route_type)
                # x = route_dic[_key][0]
                # min_len = min(min_len, len(x))
                # max_len = max(max_len, len(x))
            routes_dic[key] = route_dic
        return routes_dic
    
    def down_sampling_helper(self, route, to_cuda, cuda_device, route_type):
        if route_type == 'human':
            interval = 5
        elif route_type == 'robot':
            interval = 3
        x,y,yaw = route
        x = [x[0]] + x[1:-1][::interval] + [x[-1]]
        y = [y[0]] + y[1:-1][::interval] + [y[-1]]
        yaw = [yaw[0]] + yaw[1:-1][::interval] + [yaw[-1]]
        if to_cuda:
            x = torch.tensor(x, device=cuda_device)
            y = torch.tensor(y, device=cuda_device)
            yaw = torch.tensor(yaw, device=cuda_device)
        # self.max_x = max(self.max_x, max(x))
        # self.max_y = max(self.max_x, max(x))
        # self.min_x = max(self.max_x, max(x))
        # self.min_y = max(self.max_x, max(x))
        return x,y,yaw
    
    def generate_routes_helper(self, pose_dic : dict, file_path, have_problem_routes: dict):
        path = os.path.expanduser(file_path)
        routes_dic = {}
        with open(path, 'rb') as f:
            routes_dic = pickle.load(f)
        for (key, s) in pose_dic.items():
            if key in routes_dic.keys():
                route_dic = routes_dic[key]
            else:
                route_dic = {}
            for (_key, g) in pose_dic.items():
                if _key == key or _key in route_dic.keys():
                    continue
                if key in have_problem_routes.keys() and _key in have_problem_routes[key].keys():
                    x, y, yaw = self.path_planner_multi_poses(s.copy(), g.copy(), have_problem_routes[key][_key].copy())
                else:
                    x, y, yaw = self.path_planner(s.copy(), g.copy())
                route_dic[_key] = (x,y,yaw)
            routes_dic[key] = route_dic
            with open(path, 'wb') as f:
                pickle.dump(routes_dic, f)
        return


    def path_planner_multi_poses(self, start, goal, interval_path_list):
        interval_path_list = [start] + interval_path_list + [goal]
        x= []
        y = []
        yaw = []
        trans_x = -50
        trans_y = -30
        for i in range(0, len(interval_path_list) - 1):
            s = interval_path_list[i].copy()
            g = interval_path_list[i+1].copy()
            _x, _y, _yaw = self.path_planner(s, g)
            x += _x
            y += _y
            yaw += _yaw
        _x = [(value - trans_x)*self.xyResolution for value in x]
        _y = [(value - trans_y)*self.xyResolution for value in y]
        visualize = False
        if visualize:

            import math
            for k in range(len(_x)):
                plt.cla()
                plt.xlim(min(self.obstacleX), max(self.obstacleX)) 
                plt.ylim(min(self.obstacleY), max(self.obstacleY))        
                # plt.xlim(x_limit[0], x_limit[1]) 
                # plt.ylim(y_limit[0], y_limit[1])                
                plt.xlim(0, 300) 
                plt.ylim(0, 250)
                plt.plot(self.obstacleX, self.obstacleY, "sk")
                # plt.plot(s, g, linewidth=1.5, color='r', zorder=0)
                plt.plot(_x, _y, linewidth=1.5, color='r', zorder=0)
                # hybridAStar.drawCar(s[0], s[1], s[2])
                # hybridAStar.drawCar(g[0], g[1], g[2])
                hybridAStar.drawCar(_x[k], _y[k], yaw[k])
                plt.arrow(_x[k], _y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.2, color='royalblue')
                # plt.title("Hybrid A*",fontsize=20)
                # plt.tick_params(axis='both', which='both', labelsize=15)
                plt.gca().invert_xaxis()
                plt.gca().invert_yaxis()
                plt.title("Path planning result",fontsize=30)
                plt.tick_params(axis='both', which='both', labelsize=20)
                plt.tight_layout()
                # plt.pause(0.01)
        return x, y, yaw

    def scale_pose(self, _pose: list):
        pose = _pose.copy()
        trans_x = -50
        trans_y = -30
        pose[0] = (pose[0]/self.xyResolution + trans_x)
        pose[1] = (pose[1]/self.xyResolution + trans_y)
        return pose

    def path_planner(self, s, g):
        dis = np.linalg.norm(np.array(s[:2]) - np.array(g[:2]))
        if dis < 0.1:
            x, y, yaw = [s[0], g[0]], [s[1], g[1]], [s[2], g[2]]
            return x,y,yaw

        # self.xyResolution = 5
        trans_x = -50
        trans_y = -30
        # Set Start, Goal x, y, theta
        # s = [0, 10, np.deg2rad(90)]
        # g = [-13.3, 6, np.deg2rad(90)]
        s[0] = (s[0] - trans_x)*self.xyResolution
        s[1] = (s[1] - trans_y)*self.xyResolution
        g[0] = (g[0] - trans_x)*self.xyResolution
        g[1] = (g[1] - trans_y)*self.xyResolution
        # self.obstacleX, self.obstacleY = hybridAStar.map_png(self.xyResolution)
        # # Calculate map Paramaters
        # self.mapParameters = hybridAStar.calculateself.MapParameters(self.obstacleX, self.obstacleY, self.xyResolution, np.deg2rad(15.0))
        # Run Hybrid A*
        dis_s_m = np.linalg.norm(np.array(s) - np.array(self.planning_mid_point))
        dis_g_m = np.linalg.norm(np.array(g) - np.array(self.planning_mid_point))
        import time  # 引入time模块
        if min(s[0], g[0]) < self.planning_mid_point[0] and self.planning_mid_point[0] < max(s[0], g[0]) and dis_s_m > 10 and dis_g_m > 10:
            self.planning_mid_point[2] = 0 if (g[0] - s[0]) >=0  else np.deg2rad(180)
            start_t = time.time()
            x1, y1, yaw1 = hybridAStar.run(s, self.planning_mid_point, self.mapParameters, plt)
            x2, y2, yaw2 = hybridAStar.run(self.planning_mid_point, g, self.mapParameters, plt)
            end_t = time.time()
            if end_t-start_t > 3.:
                a = 1
            x = x1 + x2[1:]
            y = y1 + y2[1:]
            yaw = yaw1 + yaw2[1:]
            a = 1
        else:
            start_t = time.time()
            x, y, yaw = hybridAStar.run(s, g, self.mapParameters, plt)
            end_t = time.time()
            if end_t-start_t > 3.:
                a = 1
        # x_limit = [min(self.obstacleX), max(self.obstacleX)]
        # y_limit = [min(self.obstacleY), max(self.obstacleY)]
        scale_flag = True
        if scale_flag:
            x = [value/self.xyResolution + trans_x for value in x]
            y = [value/self.xyResolution + trans_y for value in y]
            # self.obstacleX = [value/self.xyResolution + trans_x  for value in self.obstacleX]
            # self.obstacleY = [value/self.xyResolution + trans_y for value in self.obstacleY]
        # # Draw Animated Car
        # import math
        visualize = False
        def show_map_s_g():
            import math
            plt.cla()
            plt.xlim(min(self.obstacleX), max(self.obstacleX)) 
            plt.ylim(min(self.obstacleY), max(self.obstacleY))        
            # plt.xlim(x_limit[0], x_limit[1]) 
            # plt.ylim(y_limit[0], y_limit[1])                
            plt.xlim(0, 300) 
            plt.ylim(0, 250)
            plt.plot(self.obstacleX, self.obstacleY, "sk")
            # plt.plot(s, g, linewidth=1.5, color='r', zorder=0)
            # plt.plot(x, y, linewidth=1.5, color='r', zorder=0)
            hybridAStar.drawCar(s[0], s[1], s[2])
            hybridAStar.drawCar(g[0], g[1], g[2])
            plt.arrow(s[0], s[1], 1*math.cos(s[2]), 1*math.sin(s[2]), width=.1)
            plt.arrow(g[0], g[1], 1*math.cos(g[2]), 1*math.sin(g[2]), width=.1)
            plt.title("Hybrid A*")
            plt.pause(0.01)

        if visualize:
            # x_limit = [-50, 30]
            # y_limit= [-30, 40]
            s[0] = (s[0])/self.xyResolution + trans_x
            g[0] = (g[0])/self.xyResolution + trans_x
            s[1] = (s[1])/self.xyResolution + trans_y
            g[1] = (g[1])/self.xyResolution + trans_y
            import math
            for k in range(len(x)):
                plt.cla()
                plt.xlim(min(self.obstacleX), max(self.obstacleX)) 
                plt.ylim(min(self.obstacleY), max(self.obstacleY))        
                # plt.xlim(x_limit[0], x_limit[1]) 
                # plt.ylim(y_limit[0], y_limit[1])                
                plt.xlim(0, 300) 
                plt.ylim(0, 250)
                plt.plot(self.obstacleX, self.obstacleY, "sk")
                # plt.plot(s, g, linewidth=1.5, color='r', zorder=0)
                plt.plot(x, y, linewidth=1.5, color='r', zorder=0)
                # hybridAStar.drawCar(s[0], s[1], s[2])
                # hybridAStar.drawCar(g[0], g[1], g[2])
                hybridAStar.drawCar(x[k], y[k], yaw[k])
                plt.arrow(x[k], y[k], 1*math.cos(yaw[k]), 1*math.sin(yaw[k]), width=.1)
                plt.title("Hybrid A*")
                plt.pause(0.01)
        
        # plt.show()
        def sample(x, interval = 2):
            _x = x[1:-1]
            _x = _x[::interval]
            x = [x[0]] + _x + [x[-1]]
            return x
        x, y, yaw = sample(x), sample(y), sample(yaw)
        return x, y, yaw
