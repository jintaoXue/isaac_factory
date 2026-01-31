


import numpy as np
import torch

from ...utils import quaternion
from .human_fatigue_model import Characters
from .hc_env_cfg import HcEnvCfg, BoxCapacity
import copy

def random_zero_index(data):

    if data.count(0)>=1:
        indexs = np.argwhere(np.array(data) == 0)
        _idx = np.random.randint(low=0, high = len(indexs)) 
        return indexs[_idx][0]
    else:
        return -1
    
def world_pose_to_navigation_pose(world_pose):
    position, orientation = world_pose[0][0].cpu().numpy(), world_pose[1][0].cpu().numpy()
    euler_angles = quaternion.quaternionToEulerAngles(orientation)
    nav_pose = [position[0], position[1], euler_angles[2]]
    return nav_pose

def find_closest_pose(pose_dic, ego_pose, in_dis=5):
    dis = np.inf
    key = None
    for _key, val in pose_dic.items():
        _dis = np.linalg.norm(np.array(val[:2]) - np.array(ego_pose[:2]))
        if _dis < 0.1:
            return _key
        elif _dis < dis:
            key = _key
            dis = _dis
    assert dis < in_dis, 'error when get closest pose, distance is: {}'.format(dis)
    return key

class TaskManager(object):
    def __init__(self, character_list, agv_list, box_list, cuda_device, env_cfg, train_cfg) -> None:
        self.cuda_device = cuda_device
        self.characters = Characters(character_list=character_list, env_cfg=env_cfg, train_cfg=train_cfg)
        self.agvs = Agvs(agv_list = agv_list, env_cfg=env_cfg, train_cfg=train_cfg)
        self.boxs = TransBoxs(box_list=box_list, env_cfg=env_cfg)
        self.task_dic =  {-1:'none', 0: 'hoop_preparing', 1:'bending_tube_preparing', 2:'hoop_loading_inner', 3:'bending_tube_loading_inner', 4:'hoop_loading_outer', 
                          5:'bending_tube_loading_outer', 6:'cutting_cube', 7:'collect_product', 8:'placing_product'}
        self.task_in_set = set()
        self.task_in_dic = {}
        # self.task_mask = torch.zeros(len(self.task_dic), device=cuda_device)
        self.task_dic_inverse = {value: key for key, value in self.task_dic.items()}
        self.cfg = env_cfg
        self._test = train_cfg['test']
        if self._test:
           self._eval_times = train_cfg['test_times']
           self.acti_num_agv = train_cfg['acti_agv']
           self.acti_num_charc = train_cfg['acti_charc']
        self.obs = None
        return
    
    def reset(self, acti_num_charc, acti_num_agv):

        assert not ((acti_num_charc is None) ^ (acti_num_agv is None)), "warning"
        if self._test:
            if acti_num_charc is None:
                acti_num_agv = self.acti_num_agv
                acti_num_charc = self.acti_num_charc
        elif acti_num_charc is None:
            acti_num_agv =  np.random.randint(1, self.cfg.n_max_robot+1)
            acti_num_charc = np.random.randint(1, self.cfg.n_max_human+1)
            '''gantt chart'''
            # acti_num_agv =  2
            # acti_num_charc = 2
        self.ini_worker_pose = self.characters.reset(acti_num_charc)
        self.ini_agv_pose = self.agvs.reset(acti_num_agv)
        self.ini_box_pose = self.boxs.reset(acti_num_agv)
        self.task_in_set = set()
        self.task_in_dic = {}
        #fatigue datetype: 'task': {'phy_fatigue': torch.tensor([0.]), 'psy_fatigue': torch.tensor([0.]), 'state': {}}
        self.fatigue_data = {}
        self.fatigue_data_list = []
        # self.task_mask = torch.zeros(len(self.task_dic), device=self.cuda_device)
        # self.task_mask[0] = 1

    def find_closest_pose(self, pose_dic, ego_pose, in_dis=5):
        dis = np.inf
        key = None
        for _key, val in pose_dic.items():
            _dis = np.linalg.norm(np.array(val[:2]) - np.array(ego_pose[:2]))
            if _dis < 0.1:
                return _key
            elif _dis < dis:
                key = _key
                dis = _dis
        assert dis < in_dis, 'error when get closest pose, distance is: {}'.format(dis)
        return key

    def assign_task(self, task):
        
        charac_idx = self.characters.assign_task(task, random = False)
        if task in self.characters.task_range:
            if charac_idx != -1:
            # assert charac_idx >= 0, "charac idx should >= 0"
            # if charac_idx < 0:
            #     a = 1
                self.fatigue_data[task] = copy.deepcopy(self.obs) 
                self.fatigue_data[task]['phy_fatigue'] = torch.tensor([self.characters.fatigue_list[charac_idx].phy_fatigue], dtype=torch.float32)
                self.fatigue_data[task]['psy_fatigue'] = torch.tensor([self.characters.fatigue_list[charac_idx].psy_fatigue], dtype=torch.float32)
                self.fatigue_data[task]['charac_idx'] = torch.tensor(charac_idx, dtype=torch.int64) 
                self.fatigue_data[task]['task_str'] = task
                self.fatigue_data[task]['action'] = torch.tensor(self.task_dic_inverse[task]+1, dtype=torch.int32)
                self.fatigue_data[task]['action'] = torch.tensor(self.task_dic_inverse[task]+1, dtype=torch.int32)
                # self.characters.fatigue_list[charac_idx].update_predict_dic()
                # self.fatigue_data[task]['phy_delta_predict'] = torch.tensor([self.characters.fatigue_list[charac_idx].task_phy_prediction_dic[task]], dtype=torch.float32) 
                # self.fatigue_data[task]['psy_delta_predict'] = torch.tensor([self.characters.fatigue_list[charac_idx].task_psy_prediction_dic[task]], dtype=torch.float32)
                self.fatigue_data[task]['filter_phy_delta_predict'] = torch.tensor([self.characters.fatigue_list[charac_idx].task_filter_phy_prediction_dic[task]], dtype=torch.float32)
                # self.fatigue_data[task]['filter_phy_fat_accuracy'] = torch.tensor(self.characters.fatigue_list[charac_idx].get_filter_fat_predict_accuracy(filter_type = 'pf'), dtype=torch.float32)
                self.fatigue_data[task]['filter_phy_rec_coe_accuracy'] = torch.tensor(self.characters.fatigue_list[charac_idx].get_filter_recover_coe_accuracy(filter_type = 'pf'), dtype=torch.float32)
                self.fatigue_data[task]['filter_phy_fat_coe_accuracy'] = torch.tensor(self.characters.fatigue_list[charac_idx].get_filter_fatigue_coe_accuracy(filter_type = 'pf'), dtype=torch.float32)
                self.fatigue_data[task]['phy_delta_predict'] = torch.tensor([self.characters.fatigue_list[charac_idx].task_phy_prediction_dic[task]], dtype=torch.float32) 
                self.fatigue_data[task]['psy_delta_predict'] = torch.tensor([self.characters.fatigue_list[charac_idx].task_psy_prediction_dic[task]], dtype=torch.float32)

                self.fatigue_data['activate_other_filters'] = False
                if self.characters.fatigue_list[charac_idx].activate_other_filters:
                    self.fatigue_data['activate_other_filters'] = True
                    self.fatigue_data[task]['filter_phy_delta_predict_kf'] = torch.tensor([self.characters.fatigue_list[charac_idx].task_filter_phy_prediction_dic_kf[task]], dtype=torch.float32)
                    # self.fatigue_data[task]['filter_phy_fat_accuracy_kf'] = torch.tensor(self.characters.fatigue_list[charac_idx].get_filter_fat_predict_accuracy(filter_type = 'kf'), dtype=torch.float32)
                    self.fatigue_data[task]['filter_phy_rec_coe_accuracy_kf'] = torch.tensor(self.characters.fatigue_list[charac_idx].get_filter_recover_coe_accuracy(filter_type = 'kf'), dtype=torch.float32)
                    self.fatigue_data[task]['filter_phy_fat_coe_accuracy_kf'] = torch.tensor(self.characters.fatigue_list[charac_idx].get_filter_fatigue_coe_accuracy(filter_type = 'kf'), dtype=torch.float32)
                    self.fatigue_data[task]['filter_phy_delta_predict_ekf'] = torch.tensor([self.characters.fatigue_list[charac_idx].task_filter_phy_prediction_dic_ekf[task]], dtype=torch.float32)
                    # self.fatigue_data[task]['filter_phy_fat_accuracy_ekf'] = torch.tensor(self.characters.fatigue_list[charac_idx].get_filter_fat_predict_accuracy(filter_type = 'ekf'), dtype=torch.float32)
                    self.fatigue_data[task]['filter_phy_rec_coe_accuracy_ekf'] = torch.tensor(self.characters.fatigue_list[charac_idx].get_filter_recover_coe_accuracy(filter_type = 'ekf'), dtype=torch.float32)
                    self.fatigue_data[task]['filter_phy_fat_coe_accuracy_ekf'] = torch.tensor(self.characters.fatigue_list[charac_idx].get_filter_fatigue_coe_accuracy(filter_type = 'ekf'), dtype=torch.float32)

            else:
                return False
            # self.fatigue_data[task]['prediction_mask'] = torch.zeros((len(self.task_dic), 2), dtype=torch.float32)
            # self.fatigue_data[task]['prediction_mask'][self.task_dic_inverse[task]+1, :] = 1
        
        box_idx = self.boxs.assign_task(task, random = False)
        if box_idx >= 0:
            box_xyz, _ = self.boxs.list[box_idx].get_world_poses()
        else:
            box_xyz = None
        agv_idx = self.agvs.assign_task(task, box_idx, box_xyz, random = False)
        lacking_resource = False
        if charac_idx == -1 or agv_idx == -1 or box_idx == -1:
            lacking_resource = True            
        assert lacking_resource is False, "lacking resource problem"
        self.task_in_set.add(task)
        self.task_in_dic[task] = {'charac_idx': charac_idx, 'agv_idx': agv_idx, 'box_idx': box_idx, 'lacking_resource': lacking_resource}
        
        return True

    def task_clearing(self, task):

        charac_idx, agv_idx, box_idx = self.task_in_dic[task]['charac_idx'], self.task_in_dic[task]['agv_idx'], self.task_in_dic[task]['box_idx']
        task_range = self.characters.task_range
        task_range.add('none')
        if task in task_range:
            assert charac_idx >=0, "charac idx should >= 0"
            self.fatigue_data[task]['next_phy_fatigue'] = torch.tensor([self.characters.fatigue_list[charac_idx].phy_fatigue], dtype=torch.float32) 
            self.fatigue_data[task]['next_psy_fatigue'] = torch.tensor([self.characters.fatigue_list[charac_idx].psy_fatigue], dtype=torch.float32)
            # self.fatigue_data[task]['phy_delta_predict'] = torch.tensor([self.characters.fatigue_list[charac_idx].task_phy_prediction_dic[task]], dtype=torch.float32) 
            # self.fatigue_data[task]['psy_delta_predict'] = torch.tensor([self.characters.fatigue_list[charac_idx].task_psy_prediction_dic[task]], dtype=torch.float32) 
            del self.fatigue_data[task]['task_str']
            self.fatigue_data_list.append(self.fatigue_data[task])
            del self.fatigue_data[task]
        self.characters.reset_idx(charac_idx)
        self.agvs.reset_idx(agv_idx)
        self.boxs.reset_idx(box_idx)
        self.task_in_set.remove(task)
        del self.task_in_dic[task]
        return

    def step(self):
        for task in self.task_in_set:
            if self.task_in_dic[task]['lacking_resource']:
                if self.task_in_dic[task]['charac_idx'] == -1:
                    self.task_in_dic[task]['charac_idx'] = self.characters.assign_task(task)
                if self.task_in_dic[task]['box_idx'] == -1:
                    self.task_in_dic[task]['box_idx'] = self.boxs.assign_task(task)
                    #reset agv idx and find the suitable agv for box again
                    agv_idx = self.task_in_dic[task]['agv_idx']
                    if agv_idx >= 0:
                        self.agvs.reset_idx(agv_idx)
                        self.task_in_dic[task]['agv_idx'] = -1
                if self.task_in_dic[task]['agv_idx'] == -1:
                    box_idx = self.task_in_dic[task]['box_idx']
                    box_xyz, _ = self.boxs.list[box_idx].get_world_poses()
                    self.task_in_dic[task]['agv_idx'] = self.agvs.assign_task(task, box_idx, box_xyz)

                try:
                    list(self.task_in_dic[task].values()).index(-1)
                    self.task_in_dic[task]['lacking_resource'] = True
                except: 
                    self.task_in_dic[task]['lacking_resource'] = False
                
        return 

    def corresp_charac_agv_box_idx(self, task):
        if task not in self.task_in_dic.keys():
            return -1, -1, -1
        return self.task_in_dic[task]['charac_idx'], self.task_in_dic[task]['agv_idx'], self.task_in_dic[task]['box_idx']

    
class Materials(object):

    def __init__(self, cube_list : list, hoop_list : list, bending_tube_list : list, upper_tube_list: list, product_list : list) -> None:

        self.cube_list = cube_list
        self.upper_tube_list = upper_tube_list
        self.hoop_list = hoop_list
        self.bending_tube_list = bending_tube_list
        self.product_list = product_list

        self.cube_state_dic = {-1:"done", 0:"wait", 1:"in_list", 2:"conveying", 3:"conveyed", 4:"cutting", 5:"cut_done", 6:"pick_up_place_cut", 
                                   7:"placed_station_inner", 8:"placed_station_outer", 9:"welding_left", 10:"welding_right", 11:"welding_upper",
                                   12:"process_done", 13:"pick_up_place_product"}
        self.hoop_state_dic = {-1:"done", 0:"wait", 1:"in_box", 2:"on_table", 3:"in_list", 4:"loading", 5:"loaded"}
        self.bending_tube_state_dic = {-1:"done", 0:"wait", 1:"in_box",  2:"on_table", 3:"in_list", 4:"loading", 5:"loaded"}
        self.upper_tube_state_dic = {}
        self.product_state_dic = {0:"waitng", 1:'collected', 2:"placed"}
        self.hoop_state_dic = {0:"wait", 1:"in_box", 2:"on_table"}
        self.bending_tube_state_dic = {0:"wait", 1:"in_box", 2:"on_table"}

        self.initial_hoop_pose = []
        self.initial_bending_tube_pose = []
        self.initial_upper_tube_pose = []
        self.initial_cube_pose = []
        self.initial_product_pose = []
                
        for obj in self.hoop_list:
            self.initial_hoop_pose.append(obj.get_world_poses())
        for obj in self.bending_tube_list:
            self.initial_bending_tube_pose.append(obj.get_world_poses())
        for obj in self.upper_tube_list:
            self.initial_upper_tube_pose.append(obj.get_world_poses())
        for obj in self.cube_list:
            self.initial_cube_pose.append(obj.get_world_poses())
        for obj in self.product_list:
            self.initial_product_pose.append(obj.get_world_poses())
        
        # self.reset()

        position = [[[-14.44042, 4.77828, 0.6]], [[-13.78823, 4.77828, 0.6]], [[-14.44042, 5.59765, 0.6]], [[-13.78823, 5.59765, 0.6]]]
        orientation = [[1 ,0 ,0, 0]]
        self.position_depot_hoop, self.orientation_depot_hoop = torch.tensor(position, dtype=torch.float32), torch.tensor(orientation, dtype=torch.float32)
        
        position = [[[-31.64901, 4.40483, 1.1]], [[-30.80189, 4.40483, 1.1]], [[-31.64901, 5.31513, 1.1]], [[-30.80189, 5.31513, 1.1]]]
        orientation = [[-1.6081e-16, -6.1232e-17,  1.0000e+00, -6.1232e-17]]
        self.position_depot_bending_tube, self.orientation_depot_bending_tube = torch.tensor(position, dtype=torch.float32), torch.tensor(orientation, dtype=torch.float32)

        position = [[[-35., 15., 0]], [[-35, 16, 0]], [[-35, 17, 0]], [[-35, 18, 0]], [[-35, 19, 0]]]
        orientation = [[1 ,0 ,0, 0]]
        self.position_depot_product, self.orientation_depot_product = torch.tensor(position, dtype=torch.float32), torch.tensor(orientation, dtype=torch.float32)

        in_box_offsets = [[[0.5,0.5, 0.5]], [[-0.5,0.5, 0.5]], [[0.5,-0.5, 0.5]], [[-0.5, -0.5, 0.5]]]
        self.in_box_offsets = torch.tensor(in_box_offsets, dtype=torch.float32) 

    def reset(self):
        obj_list = self.hoop_list+self.bending_tube_list+self.upper_tube_list+self.cube_list+self.product_list
        pose_list = self.initial_hoop_pose+self.initial_bending_tube_pose+self.initial_upper_tube_pose+self.initial_cube_pose+self.initial_product_pose
        for obj, pose in zip(obj_list, pose_list):
            obj.set_world_poses(pose[0], pose[1])
            obj.set_velocities(torch.zeros((1,6)))

        self.cube_states = [0]*len(self.cube_list)
        self.hoop_states = [0]*len(self.hoop_list)
        self.bending_tube_states = [0]*len(self.bending_tube_list)
        self.upper_tube_states = [0]*len(self.upper_tube_list)
        self.product_states = [0]*len(self.product_list)
        '''#for workers and agv to conveying the materials'''
        # self.cube_convey_states = [0]*len(self.cube_list)

        self.hoop_convey_states = [0]*len(self.hoop_list)
        self.bending_tube_convey_states = [0]*len(self.bending_tube_list)
        # self.upper_tube_convey_states = [0]*len(self.upper_tube_list)
        #for belt conveyor
        self.cube_convey_index = -1
        #cutting machine
        self.cube_cut_index = -1
        #grippers
        self.pick_up_place_cube_index = -1
        self.pick_up_place_upper_tube_index = -1
        #for inner station
        self.inner_hoop_processing_index = -1
        self.inner_cube_processing_index = -1
        self.inner_bending_tube_processing_index = -1
        self.inner_upper_tube_processing_index = -1
        # self.inner_hoop_loading_index = -1
        self.inner_bending_tube_loading_index = -1
        #for outer station
        self.outer_hoop_processing_index = -1
        self.outer_cube_processing_index = -1   #equal to product processing index
        self.outer_bending_tube_processing_index = -1
        self.outer_upper_tube_processing_index = -1
        # self.outer_hoop_loading_index = -1
        self.outer_bending_tube_loading_index = -1
        #prduction progress
        self.pre_progress = 0

    def get_world_poses(self, list):
        poses = []
        for obj in list:
            poses.append(obj.get_world_poses())
        return poses
    
    def update_poses(self):
        pass

    def done(self):
        return min(self.product_states) == 2

    def progress(self):
        totall_progress = 2*len(self.product_states)
        progress = 2*self.product_states.count(2) + self.product_states.count(1)
        return progress/totall_progress
    
    def produce_product_req(self):
        try:
            self.product_states.index(0)
            return True
        except: 
            return False
    
    def have_collecting_product_req(self):
        #only when material is ready for propcessing (at depot, loaded, processing, processed), the collect product mission is activate
        will_have_product = any(10 <= x <= 13 for x in self.cube_states)
        prepared_hoops = any(2 <= x <= 5 for x in self.hoop_states)
        prepared_bending_tube = any(2 <= x <= 5 for x in self.bending_tube_states)
        ## upper tube is automatically prepared
        return will_have_product or (prepared_hoops and prepared_bending_tube)

    def find_next_raw_cube_index(self):
        # index 
        try:
            return self.cube_states.index(0)
        except:
            return -1
        # return self.cube_states.index(0)
    
    def find_next_raw_upper_tube_index(self):
        # index 
        try:
            return self.upper_tube_states.index(0)
        except:
            return -1
        # return self.upper_tube_states.index(0)
    
    def find_next_raw_hoop_index(self):
        # index 2 is on table
        try:
            return self.hoop_states.index(2)
        except:
            return -1
        # return self.hoop_states.index(0)
    
    def find_next_raw_bending_tube_index(self):
        # index 
        try:
            return self.bending_tube_states.index(2)
        except:
            return -1


class Agvs(object):

    def __init__(self, agv_list, env_cfg : HcEnvCfg, train_cfg) -> None:
        self.agv_list = agv_list
        self.state_dic = {0:"free", 1:"moving_to_box", 2:"carrying_box", 3:"waiting"}
        self.sub_task_dic = {0:"free", 1:"carry_box_to_hoop", 2:"carry_box_to_bending_tube", 3:"carry_box_to_hoop_table", 4:"carry_box_to_bending_tube_table", 5:'collect_product', 6:'placing_product'}
        self.task_range = {'hoop_preparing', 'bending_tube_preparing', 'collect_product','placing_product'}
        self.low2high_level_task_dic =  {"carry_box_to_hoop":'hoop_preparing', "carry_box_to_bending_tube":'bending_tube_preparing', "carry_box_to_hoop_table":'hoop_preparing', 
                                         "carry_box_to_bending_tube_table":'bending_tube_preparing', 'collect_product':'collect_product', 'placing_product':'placing_product'}
        
        self.poses_dic = {"carry_box_to_hoop": [-0.654, 8.0171, np.deg2rad(0)] , "carry_box_to_bending_tube": [-0.654, 11.62488, np.deg2rad(0)], 
                        "carry_box_to_hoop_table": [-11.69736, 5.71486, np.deg2rad(0)], "carry_box_to_bending_tube_table":[-33.55065, 5.71486, np.deg2rad(-90)] ,
                        'collect_product':[-21.76757, 10.78427, np.deg2rad(0)],'placing_product':[-38.54638, 12.40097, np.deg2rad(0)], 
                        'initial_pose_0':[-4.8783107, 8.017096, 0.0], 'initial_pose_1': [-4.8726454, 11.656976, 0.0],
                        'initial_box_pose_0': [-1.6895515, 8.0171, 0.0], 'initial_box_pose_1': [-1.7894887, 11.822739, 0.0]}
        self.poses_dic2num = {
            "carry_box_to_hoop": 0 , "carry_box_to_bending_tube": 1, 
            "carry_box_to_hoop_table": 2, "carry_box_to_bending_tube_table":3,
            'collect_product':4,'placing_product':5,
            'initial_pose_0':6, 'initial_pose_1':7,
            'initial_box_pose_0':8, 'initial_box_pose_1':9}
        # self.initial_pose_list = []
        # for obj in self.list:
        #     self.initial_pose_list.append(obj.get_world_poses())
        # self.initial_xy_yaw = []
        # for idx, agv in enumerate(self.list):
        #     xy_yaw = world_pose_to_navigation_pose(agv.get_world_poses())
        #     # self.initial_xy_yaw.append(xy_yaw)
        #     self.poses_dic[f'initial_pose_{idx}'] = xy_yaw
        self.routes_dic = None
        # self.corresp_charac_idxs = [-1]*self.num
        # self.corresp_box_idxs = [-1]*self.num

        self.picking_pose_hoop = [-0.654, 8.0171, np.deg2rad(0)]  #
        self.picking_pose_bending_tube = [-0.654, 11.62488, np.deg2rad(0)]
        self.picking_pose_table_hoop = [-11.69736, 5.71486, np.deg2rad(0)]
        self.picking_pose_table_bending_tube = [-33.55065, 5.71486, np.deg2rad(-90)] 
        self.collecting_product_pose = [-21.76757, 10.78427, np.deg2rad(0)]
        self.placing_product_pose = [-38.54638, 12.40097, np.deg2rad(0)]
        self.gantt_chart_data = train_cfg['gantt_chart_data']
        return
    
    def reset(self, acti_num_agv=None, random = None):
        # if acti_num_agv is None:
        #     acti_num_agv = np.random.randint(1, HRTaskAllocEnvCfg.n_max_robot)
        self.acti_num_agv = acti_num_agv
        self.states = [0]*acti_num_agv
        self.tasks = [0]*acti_num_agv
        self.movements = [0]*acti_num_agv
        self.list = self.agv_list[:acti_num_agv]
        self.x_paths = [[] for i in range(acti_num_agv)]
        self.y_paths = [[] for i in range(acti_num_agv)]
        self.yaws = [[] for i in range(acti_num_agv)]
        self.path_idxs = [0 for i in range(acti_num_agv)]
        if random is None:
            random = np.random.choice(len(self.poses_dic), acti_num_agv, replace=False)
        pose_list = list(self.poses_dic.values())
        pose_str_list = list(self.poses_dic.keys())
        initial_pose_str = []
        for i in range(0, len(self.agv_list)):
            if i < acti_num_agv:
                initial_pose_str.append(pose_str_list[random[i]])
                position = pose_list[random[i]][:2]+[0.1]
                self.agv_list[i].set_world_poses(torch.tensor([position]), torch.tensor([[1., 0., 0., 0.]]))
                self.agv_list[i].set_velocities(torch.zeros((1,6)))
                self.reset_idx(i)
                self.reset_path(i)
            else:
                position = [0, 0, -100]
                self.agv_list[i].set_world_poses(torch.tensor([position]), torch.tensor([[1., 0., 0., 0.]]))
                self.agv_list[i].set_velocities(torch.zeros((1,6)))

        self.poses_str = initial_pose_str
        if self.gantt_chart_data:
            self.time_step = []
            self.task_level_history = []
            for i in range(acti_num_agv):
                self.task_level_history.append([('free', "free", 'none', 0)]) #state, subtask, task, time_step
                self.time_step.append(0)
        return initial_pose_str

    def update_pose_str(self, idx):
        worker_position = self.list[idx].get_world_poses()
        wp = world_pose_to_navigation_pose(worker_position)
        wp_str = find_closest_pose(pose_dic=self.poses_dic, ego_pose=wp, in_dis=1000.)
        self.poses_str[idx] = wp_str
    
    def reset_idx(self, idx):
        if idx < 0 :
            return
        self.tasks[idx] = 0
        self.states[idx] = 0

    def assign_task(self, high_level_task, box_idx, box_xyz, random=False):
        #todo  
        if high_level_task not in self.task_range or box_idx == -2:
            return -2
        if box_xyz is None:
            return -1
        
        if random:
            idx = random_zero_index([a*b for a,b in zip(self.states,self.tasks)])
        else: 
            idx = self.find_available_agv(box_idx, box_xyz[0])
        # idx = self.find_available_agv(box_idx, box_xyz[0])

        if idx == -1:
            return idx
        if high_level_task == 'hoop_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 1 
        elif high_level_task == 'bending_tube_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 2
        elif high_level_task == 'collect_product':
            self.tasks[idx] = 5
        elif high_level_task == 'placing_product':
            self.tasks[idx] = 6 
        return idx
    
    def find_available_agv(self, box_idx, box_xyz):
        available = [a*b for a,b in zip(self.states,self.tasks)]
        if box_idx == -1: 
            try:
                return available.index(0)
            except: 
                return -1
        else:
            count = available.count(0)
            if count == 0:
                return -1
            elif count == 1:
                return available.index(0)
            else:
                if True:
                    min_dis_idx = -1
                    pre_dis = torch.inf
                    for agv_idx in range(0, self.acti_num_agv):
                        if available[agv_idx] == 0:
                            agv_xyz, _ = self.list[agv_idx].get_world_poses()
                            dis = torch.norm(agv_xyz[0] - box_xyz)
                            if dis.cpu() < pre_dis:
                                pre_dis = dis
                                min_dis_idx = agv_idx
                else:
                    min_dis_idx = -1
                    pre_dis = -torch.inf
                    for agv_idx in range(0, self.acti_num_agv):
                        if available[agv_idx] == 0:
                            agv_xyz, _ = self.list[agv_idx].get_world_poses()
                            dis = torch.norm(agv_xyz[0] - box_xyz)
                            if dis.cpu() > pre_dis:
                                pre_dis = dis
                                min_dis_idx = agv_idx
            return min_dis_idx
    
    def step_next_pose(self, agv_idx):
        reaching_flag = False
        #skip the initial pose
        # if len(self.x_paths[agv_idx]) == 0:
        #     position = [current_pose[0], current_pose[1], 0]
        #     euler_angles = [0,0, current_pose[2]]
        #     return position, quaternion.eulerAnglesToQuaternion(euler_angles), True

        self.path_idxs[agv_idx] += 1
        path_idx = self.path_idxs[agv_idx]
        # if agv_idx == 0:
        #     a = 1
        if path_idx >= (len(self.x_paths[agv_idx])):
            reaching_flag = True
            position = [self.x_paths[agv_idx][-1], self.y_paths[agv_idx][-1], 0]
            euler_angles = [0,0, self.yaws[agv_idx][-1]]
        else:
            position = [self.x_paths[agv_idx][path_idx], self.y_paths[agv_idx][path_idx], 0]
            euler_angles = [0,0, self.yaws[agv_idx][path_idx]]

        orientation = quaternion.eulerAnglesToQuaternion(euler_angles)
        self.movements[agv_idx] +=1
        return position, orientation, reaching_flag

    def step_gantt_chart(self, idx, state, subtask, task):
        self.time_step[idx] += 1
        if task == -1:
            task = 'none'
        self.task_level_history[idx].append((self.state_dic[state], self.sub_task_dic[subtask], task, self.time_step[idx]))
        
    def reset_path(self, agv_idx):
        self.x_paths[agv_idx] = []
        self.y_paths[agv_idx] = []
        self.yaws[agv_idx] = []
        self.path_idxs[agv_idx] = 0
    
    def low2high_level_task_mapping(self, task):
        task = self.sub_task_dic[task]
        if task in self.low2high_level_task_dic.keys():
            return self.low2high_level_task_dic[task]
        else: return -1
    
    def get_sum_movement(self):
        return sum(self.movements)


class TransBoxs(object):

    def __init__(self, box_list, env_cfg : HcEnvCfg) -> None:
        self.box_list = box_list
        self.state_dic = {0:"free", 1:"waiting", 2:"moving"}
        self.sub_task_dic = {0:"free", 1:"waiting_agv", 2:"moving_with_box", 3: "collect_product"}
        self.task_range = {'hoop_preparing', 'bending_tube_preparing', 'collect_product','placing_product'}

        # self.poses_dic = {'initial_box_pose_0': [-1.6895515, 8.0171, 0.0], 'initial_box_pose_1': [-1.7894887, 11.822739, 0.0]}
        self.high2low_level_task_dic = {'hoop_preparing':'carry_box_to_hoop', 'bending_tube_preparing':'carry_box_to_bending_tube', 'collect_product':'collect_product','placing_product':'placing_product'}
        self.poses_dic = {"carry_box_to_hoop": [-0.654, 8.0171, np.deg2rad(0)] , "carry_box_to_bending_tube": [-0.654, 11.62488, np.deg2rad(0)], 
                "carry_box_to_hoop_table": [-11.69736, 5.71486, np.deg2rad(0)], "carry_box_to_bending_tube_table":[-33.55065, 5.71486, np.deg2rad(-90)] ,
                'collect_product':[-21.76757, 10.78427, np.deg2rad(0)],'placing_product':[-38.54638, 12.40097, np.deg2rad(0)], 
                'initial_pose_0':[-4.8783107, 8.017096, 0.0], 'initial_pose_1': [-4.8726454, 11.656976, 0.0],
                'initial_box_pose_0': [-1.6895515, 8.0171, 0.0], 'initial_box_pose_1': [-1.7894887, 11.822739, 0.0]}
        
        self.poses_dic2num = {
            "carry_box_to_hoop": 0 , "carry_box_to_bending_tube": 1, 
            "carry_box_to_hoop_table": 2, "carry_box_to_bending_tube_table":3,
            'collect_product':4,'placing_product':5,
            'initial_pose_0':6, 'initial_pose_1':7,
            'initial_box_pose_0':8, 'initial_box_pose_1':9}
        # self.initial_pose_list = []
        # for obj in self.list:
        #     self.initial_pose_list.append(obj.get_world_poses())
        self.cfg = env_cfg
        self.capacity = BoxCapacity()
        self.routes_dic = None
        self.reset()

        return
    
    def reset(self, acti_num_box=None, random = None):
        if acti_num_box is None:
            acti_num_box = np.random.randint(2, 3)
        self.acti_num_box = acti_num_box
        self.list = self.box_list[:acti_num_box]
        self.states = [0]*acti_num_box
        self.tasks = [0]*acti_num_box
        self.high_level_tasks = ['']*acti_num_box
        self.hoop_idx_list =[[] for i in range(acti_num_box)]
        self.bending_tube_idx_sets = [set() for i in range(acti_num_box)]
        self.product_idx_list = [[] for i in range(acti_num_box)]
        self.counts = [0 for i in range(acti_num_box)]
        self.product_collecting_idx = -1
        if random is None:
            random = np.random.choice(len(self.poses_dic), acti_num_box, replace=False)
        pose_list = list(self.poses_dic.values())
        pose_str_list = list(self.poses_dic.keys())
        initial_pose_str = []
        for i in range(0, len(self.box_list)):
            if i < acti_num_box:
                initial_pose_str.append(pose_str_list[random[i]])
                position = pose_list[random[i]][:2]+[0.0]
                self.box_list[i].set_world_poses(torch.tensor([position]), torch.tensor([[1., 0., 0., 0.]]))
                self.box_list[i].set_velocities(torch.zeros((1,6)))
                self.reset_idx(i)
            else:
                position = [0, 0, -100]
                self.box_list[i].set_world_poses(torch.tensor([position]), torch.tensor([[1., 0., 0., 0.]]))
                self.box_list[i].set_velocities(torch.zeros((1,6)))

        self.poses_str = initial_pose_str
        return initial_pose_str

    def update_pose_str(self, idx):
        worker_position = self.list[idx].get_world_poses()
        wp = world_pose_to_navigation_pose(worker_position)
        wp_str = find_closest_pose(pose_dic=self.poses_dic, ego_pose=wp, in_dis=1000.)
        self.poses_str[idx] = wp_str

    def reset_idx(self, idx):
        if idx < 0 :
            return
        self.tasks[idx] = 0
        self.states[idx] = 0
        self.high_level_tasks[idx] = ''
        # self.corresp_charac_idxs[idx] = -1
        # self.corresp_agv_idxs[idx] = -1

    def assign_task(self, high_level_task, random=False):
        #todo
        if high_level_task not in self.task_range:
            return -2
        
        if high_level_task == 'placing_product':
            # idx = self.find_carrying_products_box_idx()
            # if idx >=0 :
            #     self.high_level_tasks[idx] = high_level_task
            #     self.tasks[idx] = 1 
            #     return idx
            # else:
            #     return -1 
            assert self.product_collecting_idx>=0, "placing product task wrong"
            self.high_level_tasks[self.product_collecting_idx] = high_level_task
            self.tasks[self.product_collecting_idx] = 1 
            return self.product_collecting_idx
        
        # idx = self.find_available_box()
        if random:
            idx = random_zero_index([a*b for a,b in zip(self.states,self.tasks)])
        else: 
            idx = self.find_available_box(high_level_task)
        if idx == -1:
            if high_level_task == 'collect_product':
                self.product_collecting_idx = -1
            return -1
        # if high_level_task == 'hoop_preparing' or high_level_task == 'bending_tube_preparing' or high_level_task == 'colle':
            # idx = self.find_available_charac()
        else:
            self.high_level_tasks[idx] = high_level_task
            self.tasks[idx] = 1 
            if high_level_task == 'collect_product':
                self.product_collecting_idx = idx
            return idx

    def find_available_box(self, task, idx=0):
        available = [a*b for a,b in zip(self.states,self.tasks)]
        # try:
        #     return available.index(0)
        # except: 
        #     return -1
        count = available.count(0)
        if count == 0:
            return -1
        elif count == 1:
            return available.index(idx)
        else:
            task_pose_str = self.high2low_level_task_dic[task]
            closet_idx = None
            shortest_path = None
            for i in range(0, len(available)):
                if available[i] != 0:
                    continue
                if self.poses_str[i] == task_pose_str: #the closet idx
                    return i
                x,y,yaw = self.routes_dic[self.poses_str[i]][task_pose_str]
                path_len = len(x)
                if closet_idx is None:
                    closet_idx = i
                    shortest_path = path_len
                else:
                    closet_idx = i if path_len < shortest_path else closet_idx
            return closet_idx
        
    def find_carrying_products_box_idx(self):
        for list, idx in zip(self.product_idx_list, range(len(self.product_idx_list))):
            if len(list) > 0:
                return idx
        else:
            return -1
        
    def find_full_products_box_idx(self):
        for list, idx in zip(self.product_idx_list, range(len(self.product_idx_list))):
            if len(list) >= self.capacity.product:
                return idx
        else:
            return -1
        
    def is_full_products(self):
        return self.find_full_products_box_idx() != -1
