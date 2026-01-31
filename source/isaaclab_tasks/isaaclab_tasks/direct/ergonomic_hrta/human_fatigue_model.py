import random
import numpy as np
import torch
import math
from ...utils import quaternion
from .ekf_filter import EkfFatigue, EKfRecover
from .kf_filter import KfFatigue, KfRecover
from .pf_filter import ParticleFilter, RecParticleFilter
# from .pf_filter_improved import ParticleFilter, RecParticleFilter
from .eg_hrta_env_cfg import HRTaskAllocEnvCfg, high_level_task_dic, high_level_task_rev_dic, BoxCapacity
import random
import os
from matplotlib import gridspec
import time

######### for human fatigue #####

def world_pose_to_navigation_pose(world_pose):
    position, orientation = world_pose[0][0].cpu().numpy(), world_pose[1][0].cpu().numpy()
    euler_angles = quaternion.quaternionToEulerAngles(orientation)
    nav_pose = [position[0], position[1], euler_angles[2]]
    return nav_pose

def random_zero_index(data):

    if data.count(0)>=1:
        indexs = np.argwhere(np.array(data) == 0)
        _idx = np.random.randint(low=0, high = len(indexs)) 
        return indexs[_idx][0]
    else:
        return -1

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

class Fatigue(object):

    def __init__(self, human_idx, human_types, env_cfg : HRTaskAllocEnvCfg, train_cfg) -> None:
        #task_human_subtasks_dic
        #"approaching" subtask in ommitted as it is high dynamic and hard to caculate
        self.cfg = env_cfg
        self.num_particles = self.cfg.num_particles
        self._test = train_cfg['test']
        self.hyper_param_time = self.cfg.hyper_param_time
        self.task_human_subtasks_dic =  {'none': ['free'], 'hoop_preparing': ['put_hoop_into_box', 'put_hoop_on_table']*BoxCapacity.hoop, 
            'bending_tube_preparing': ['put_bending_tube_into_box','put_bending_tube_on_table']*BoxCapacity.bending_tube, 
            'hoop_loading_inner': ['hoop_loading_inner'], 'bending_tube_loading_inner': ['bending_tube_loading_inner'], 
            'hoop_loading_outer': ['hoop_loading_outer'], 'bending_tube_loading_outer':['bending_tube_loading_outer'], 
            'cutting_cube':['cutting_cube'], 'collect_product':['free'], 'placing_product':['placing_product']*BoxCapacity.product}
        
        self.phy_free_state_dic = {"free", "waiting_box", "approaching"}
        self.psy_free_state_dic = {"free", "waiting_box", "approaching"}
        #coefficient dic: combine all the subtask and state
        self.raw_phy_fatigue_ce_dic = {"free": None, "waiting_box": None, "approaching": None, "put_hoop_into_box": 0.04, "put_bending_tube_into_box": 0.06, 
                        'put_hoop_on_table': 0.04, 'put_bending_tube_on_table': 0.06, 'hoop_loading_inner': 0.12, "hoop_loading_outer": 0.12, 'bending_tube_loading_inner': 0.15, 
                        'bending_tube_loading_outer': 0.15, "cutting_cube": 0.01, "placing_product": 0.15}
        self.raw_psy_fatigue_ce_dic = {"free": None, "waiting_box": None, "approaching": None, "put_hoop_into_box": 0.1, "put_bending_tube_into_box": 0.15, 
                        'put_hoop_on_table': 0.1, 'put_bending_tube_on_table': 0.15, 'hoop_loading_inner': 0.05, "hoop_loading_outer": 0.05, 'bending_tube_loading_inner': 0.1, 
                        'bending_tube_loading_outer': 0.1, "cutting_cube": 0.01, "placing_product": 0.3}
        self.raw_phy_recovery_ce_dic = {"free": 0.05, "waiting_box": 0.05, "approaching": 0.02}
        self.raw_psy_recovery_ce_dic = {"free": 0.05, "waiting_box": 0.05, "approaching": 0.02}
        self.human_types = human_types
        self.human_type_coe_dic = {"strong": 0.8, "normal": 1.0, "weak": 1.2}

        self.ONE_STEP_TIME = 0.1
        self.ftg_thresh_phy = self.cfg.ftg_thresh_phy
        self.ftg_thresh_psy = self.cfg.ftg_thresh_psy
        self.ftg_task_mask = None
        
        # self.device = cuda_device
        self.idx = human_idx
        # self.phy_recovery_coefficient = self.phy_recovery_ce_dic[human_type]
        # self.psy_recovery_coefficient = self.psy_recovery_ce_dic[human_type]
        
        self.phy_fatigue = None
        self.psy_fatigue = None
        self.time_step = None
        # self.time_step_level_f_history = None
        self.subtask_level_f_history = None
        self.task_levle_f_history = None
        self.phy_history = None # value, state, time
        self.psy_history = None
        # self.time_history = None
        self.visualize = False
        self.visualize_inference_time = False
        # self.time_latency_study = False
        self.pf_inference_time_step_list = []
        self.kf_inference_time_step_list = []
        self.ekf_inference_time_step_list = []
        self.pf_subtask_inference_time_step_dict = {}
        self.activate_other_filters = train_cfg['other_filters']
        self.gantt_chart_data = train_cfg['gantt_chart_data']
        return

    def reset(self):
        if self.time_step is not None and self.time_step > 100 and self.visualize and self.activate_other_filters:
            self.plot_comprehensive_fatigue_analysis()
        
        if self.time_step is not None and self.time_step > 100 and len(self.pf_inference_time_step_list) > 0 and self.visualize_inference_time:
            mean_pf, mean_kf, mean_ekf = np.mean(self.pf_inference_time_step_list), np.mean(self.kf_inference_time_step_list), np.mean(self.ekf_inference_time_step_list)
            print(f"PF inference time step: {mean_pf, len(self.pf_inference_time_step_list)}, \
              KF inference time step: {mean_kf, len(self.kf_inference_time_step_list)}, \
              EKF inference time step: {mean_ekf, len(self.ekf_inference_time_step_list)}")
            
            # print(f"PF subtask {subtask} inference time step: {mean_subtask}")
            # self.pf_inference_time_step_list = []
            # self.kf_inference_time_step_list = []
            # self.ekf_inference_time_step_list = []
            # self.pf_subtask_inference_time_step_dict = {}
            # if self.cfg.use_partial_filter:
            #     for k, v in self.phy_fatigue_ce_dic.items():
            #         if v is not None:
            #             filter : ParticleFilter = self.pfs_phy_fat[k]
            #             filter.plot_results(filter.times, filter.F_estimates, filter.lambda_estimates, 'fatigue_' + k)
            #     R_filter : RecParticleFilter = self.pfs_phy_rec['free']
            #     R_filter.plot_results(R_filter.times, R_filter.F_estimates, R_filter.lambda_estimates, name='recover')
        self.time_step = 0
        self.phy_fatigue = 0
        self.psy_fatigue = 0
        self.pre_state_type = 'free'
        
        scale_phy = 3
        scale_psy = 0.5
        scale_phy_recover= 0.3
        self.human_type = random.choice(self.human_types)
        self.phy_fatigue_ce_dic = self.scale_coefficient(scale_phy*self.human_type_coe_dic[self.human_type], self.raw_phy_fatigue_ce_dic)
        self.psy_fatigue_ce_dic = self.scale_coefficient(scale_psy, self.raw_psy_fatigue_ce_dic)
        self.phy_recovery_ce_dic = self.scale_coefficient(scale_phy_recover, self.raw_phy_recovery_ce_dic)
        self.psy_recovery_ce_dic = self.scale_coefficient(scale_psy, self.raw_psy_recovery_ce_dic)

        random_percent = 0.2
        random_bound = 0.1
        random_per_for_pf = 0.3
        # self.random_percent = random_percent
        self.pfs_phy_fat_ce_dic = self.add_coefficient_randomness(random_percent, random_bound, self.phy_fatigue_ce_dic)
        self.pfs_phy_rec_ce_dic = self.add_coefficient_randomness(random_percent, 0.02, self.phy_recovery_ce_dic)
        self.pfs_phy_fat = {}
        self.pfs_phy_rec = {}

        for (key, v) in self.phy_fatigue_ce_dic.items():
            v_pf = self.pfs_phy_fat_ce_dic.get(key, v)
            if v is not None:
                # self.pfs_phy_fat[key] = EkfFatigue(dt=1, num_steps=100, true_lambda=v, F0=0, Q=np.diag([0.01, 0.0001]), R=np.array([[0.1]]), x0=np.array([0., 0.1]), P0=np.diag([1.0, 1.0]))
                self.pfs_phy_fat[key] = ParticleFilter(dt=0.1, num_steps=100, true_lambda=v, F0=self.phy_fatigue, num_particles=self.cfg.num_particles, sigma_w=0.00000003, 
                    sigma_v=0.005, lamda_init = v_pf, upper_bound=v_pf*(1+random_per_for_pf), lower_bound=v_pf*(1-random_per_for_pf))
                # self.pfs_phy_fat[key] = ParticleFilter(dt=0.1, num_steps=100, true_lambda=v, F0=0, num_particles=500, sigma_w=0.01, sigma_v=0.001, lamda_init = v, upper_bound=v*(1+random_percent), lower_bound=v*(1+random_percent))
                self.pfs_phy_fat_ce_dic[key] = np.sum(self.pfs_phy_fat[key].particles * self.pfs_phy_fat[key].weights)
        for (key, v) in self.phy_recovery_ce_dic.items():
            v_pf = self.pfs_phy_rec_ce_dic.get(key, v)
            if v is not None:
                # self.pfs_phy_rec[key] = EKfRecover(dt=0.1, num_steps=100, true_mu=v, R0=0, Q=np.diag([0.01, 0.0001]), R=np.array([[0.1]]), x0=np.array([0., 0.1]), P0=np.diag([1.0, 1.0])) 
                self.pfs_phy_rec[key] = RecParticleFilter(dt=0.1, num_steps=100, true_lambda=v, F0=self.phy_fatigue, num_particles=500, sigma_w=0.00000003, 
                    sigma_v=0.005, lamda_init = v_pf, upper_bound=v_pf*(1+random_per_for_pf), lower_bound=v_pf*(1-random_per_for_pf)) 
                self.pfs_phy_rec_ce_dic[key] = np.sum(self.pfs_phy_rec[key].particles * self.pfs_phy_rec[key].weights)

        self.task_phy_prediction_dic = {task: 0.  for (key, task) in high_level_task_dic.items()} 
        self.task_psy_prediction_dic = {task: 0.  for (key, task) in high_level_task_dic.items()} 
        self.task_phy_prediction_dic = self.update_predict_dic()
        self.task_filter_phy_prediction_dic = self.update_filter_predict_dic()
        self.update_ftg_mask()
        self.ftg_task_mask = torch.ones(len(high_level_task_dic))

        if self.visualize or self.gantt_chart_data:
            self.phy_history = [(0, self.time_step)] # value, time_step
            self.psy_history = [(0, self.time_step)]
            # self.time_step_level_f_history = [('free', 'free', 'none', self.pfs_phy_rec_ce_dic['free'], self.phy_fatigue, self.phy_fatigue, self.time_step)] #state, subtask, task, time_step 
            self.subtask_level_f_history = [('free', 'free', 'none', self.phy_fatigue, self.psy_fatigue, self.time_step)] #state, subtask, task, time_step
            _predict_list = self.one_task_fatigue_prediction('none', self.pfs_phy_fat_ce_dic, self.pfs_phy_rec_ce_dic)
            self.task_levle_f_history = [('free', 'none', self.phy_fatigue, self.psy_fatigue, self.time_step, _predict_list)] #state, task, time_step 
            _predict_list_true = self.one_task_fatigue_prediction('none', self.phy_fatigue_ce_dic, self.phy_recovery_ce_dic)
            self.task_levle_f_history_true = [('free', 'none', self.phy_fatigue, self.psy_fatigue, self.time_step, _predict_list_true)] #state, task, time_step 

        if self.activate_other_filters:
            self.reset_for_other_filters()
        return

    def reset_for_other_filters(self):
        """
        用于可视化或者对比时，初始化各类滤波器（PF/KF/EKF）
        """
        # 初始化KF和EKF的字典
        self.kfs_phy_fat_ce_dic = {k:v for k,v in self.pfs_phy_fat_ce_dic.items()}
        self.kfs_phy_rec_ce_dic = {k:v for k,v in self.pfs_phy_rec_ce_dic.items()}
        self.ekfs_phy_fat_ce_dic = {k:v for k,v in self.pfs_phy_fat_ce_dic.items()}
        self.ekfs_phy_rec_ce_dic = {k:v for k,v in self.pfs_phy_rec_ce_dic.items()}
        self.kfs_phy_fat = {}
        self.kfs_phy_rec = {}
        self.ekfs_phy_fat = {}
        self.ekfs_phy_rec = {}
        
        for (key, v_real) in self.phy_fatigue_ce_dic.items():
            if v_real is not None:
                # 从PF字典获取对应的值用于x0初始化
                v_pf = self.pfs_phy_fat_ce_dic.get(key, v_real)
                # KF - 需要正确的参数维度
                self.kfs_phy_fat[key] = KfFatigue(dt=0.1, num_steps=100, true_lambda=v_real, init_lambda=v_pf, F0=self.phy_fatigue, 
                                                   Q=np.diag([0.0005, 0.0001]), R=np.array([[0.0005]]), 
                                                   x0=np.array([self.phy_fatigue, v_pf]), P0=np.diag([1.0, 1.0]))
                # EKF - 需要正确的参数维度
                self.ekfs_phy_fat[key] = EkfFatigue(dt=0.1, num_steps=100, true_lambda=v_real, init_lambda=v_pf, F0=self.phy_fatigue, 
                                                     Q=np.diag([0.0005, 0.0001]), R=np.array([[0.0005]]), 
                                                     x0=np.array([self.phy_fatigue, v_pf]), P0=np.diag([1.0, 1.0]))
        
        for (key, v_real) in self.phy_recovery_ce_dic.items():
            if v_real is not None:
                # 从PF字典获取对应的值用于x0初始化
                v_pf = self.pfs_phy_rec_ce_dic.get(key, v_real)
                # KF recovery - 需要正确的参数维度
                self.kfs_phy_rec[key] = KfRecover(dt=0.1, num_steps=100, true_mu=v_real, init_mu=v_pf, R0=self.phy_fatigue, 
                                                   Q=np.diag([0.0005, 0.001]), R=np.array([[0.001]]), 
                                                   x0=np.array([self.phy_fatigue, v_pf]), P0=np.diag([1.0, 1.0]))
                # EKF recovery - 需要正确的参数维度
                self.ekfs_phy_rec[key] = EKfRecover(dt=0.1, num_steps=100, true_mu=v_real, init_mu=v_pf, R0=self.phy_fatigue, 
                                                     Q=np.diag([0.0005, 0.001]), R=np.array([[0.001]]), 
                                                     x0=np.array([self.phy_fatigue, v_pf]), P0=np.diag([1.0, 1.0]))
        self.task_filter_phy_prediction_dic_kf = self.update_filter_predict_dic(filter_type = 'kf')
        self.task_filter_phy_prediction_dic_ekf = self.update_filter_predict_dic(filter_type = 'ekf')
        
        if self.visualize:
            _predict_list = self.one_task_fatigue_prediction('none', self.pfs_phy_fat_ce_dic, self.pfs_phy_rec_ce_dic)
            self.task_levle_f_history_kf = [('free', 'none', self.phy_fatigue, self.psy_fatigue, self.time_step, _predict_list)]
            self.task_levle_f_history_ekf = [('free', 'none', self.phy_fatigue, self.psy_fatigue, self.time_step, _predict_list)]

        return

    def have_overwork(self):
        return self.phy_fatigue>self.ftg_thresh_phy or self.psy_fatigue>self.ftg_thresh_psy
    
    def compute_fatigue_cost(self):
        return self.phy_fatigue

    def scale_coefficient(self, scale, dic : dict):
        return {key: (v * scale if v is not None else None)  for (key, v) in dic.items()}
    
    def add_coefficient_randomness(self, scale, random_bound, dic : dict):
        _dict = {}
        for (key, v) in dic.items():
            if v is not None :
                random = v*np.random.uniform(-scale, scale)
                random = np.clip(random, -random_bound, random_bound)
                _dict[key] = (v + random) 
            else:
                _dict[key] = None

        return _dict

    def get_phy_fatigue_coe(self):
        if self.cfg.use_partial_filter:
            return list(self.pfs_phy_fat_ce_dic.values())[3:]
        return list(self.phy_fatigue_ce_dic.values())[3:]

    def step(self, state_type, subtask, task, ftg_prediction = None):
        if self.cfg.use_partial_filter == True:
            # 合并疲劳和恢复过滤器列表
            pf_filters = {**self.pfs_phy_fat, **self.pfs_phy_rec}
            start_time = time.time()
            esitmate_phy_fatigue_coe, _phy_fatigue_prediction = self.step_filter(self.phy_fatigue, state_type, subtask, self.ONE_STEP_TIME, self.phy_fatigue_ce_dic, self.phy_recovery_ce_dic, pf_filters)
            end_time = time.time()  
            self.pf_inference_time_step_list.append(end_time - start_time)
            if subtask not in self.pf_subtask_inference_time_step_dict:
                self.pf_subtask_inference_time_step_dict[subtask] = []
            if self.visualize_inference_time:
                self.pf_subtask_inference_time_step_dict[subtask].append(end_time - start_time)
            # if np.isnan(esitmate_phy_fatigue_coe):
            #     a = 1
            if self.activate_other_filters:
                # KF filter
                kf_filters = {**self.kfs_phy_fat, **self.kfs_phy_rec}
                start_time = time.time()
                esitmate_phy_fatigue_coe_kf, _phy_fatigue_prediction_kf = self.step_filter(self.phy_fatigue, state_type, subtask, self.ONE_STEP_TIME, self.kfs_phy_fat_ce_dic, self.kfs_phy_rec_ce_dic, kf_filters)
                end_time = time.time()
                if self.visualize_inference_time:
                    self.kf_inference_time_step_list.append(end_time - start_time)
                # EKF filter
                ekf_filters = {**self.ekfs_phy_fat, **self.ekfs_phy_rec}
                start_time = time.time()
                esitmate_phy_fatigue_coe_ekf, _phy_fatigue_prediction_ekf = self.step_filter(self.phy_fatigue, state_type, subtask, self.ONE_STEP_TIME, self.ekfs_phy_fat_ce_dic, self.ekfs_phy_rec_ce_dic, ekf_filters)
                end_time = time.time()
                if self.visualize_inference_time:
                    self.ekf_inference_time_step_list.append(end_time - start_time)
        
        self.phy_fatigue = self.step_helper_delta_phy_fatigue(self.phy_fatigue, state_type, subtask, self.ONE_STEP_TIME,  self.phy_fatigue_ce_dic, self.phy_recovery_ce_dic, self.phy_free_state_dic)
        self.task_phy_prediction_dic = self.update_predict_dic()
        self.task_filter_phy_prediction_dic = self.update_filter_predict_dic()
        if self.activate_other_filters:
            self.task_filter_phy_prediction_dic_kf = self.update_filter_predict_dic(filter_type = 'kf')
            self.task_filter_phy_prediction_dic_ekf = self.update_filter_predict_dic(filter_type = 'ekf')
        # _phy_fatigue_prediction = self.step_helper_delta_phy_fatigue(self.phy_fatigue, state_type, subtask, self.ONE_STEP_TIME, 
        #     self.pfs_phy_fat_ce_dic, self.pfs_phy_rec_ce_dic, self.phy_free_state_dic)
        # self.psy_fatigue = self.step_helper_psy(self.psy_fatigue, state_type, subtask, self.ONE_STEP_TIME)
        self.time_step += 1
        if self.visualize or self.gantt_chart_data:
            self.phy_history.append((self.phy_fatigue, self.time_step))
            self.psy_history.append((self.psy_fatigue, self.time_step))
            # self.time_step_level_f_history.append((state_type, task, subtask, esitmate_phy_fatigue_coe, _phy_fatigue_prediction, self.phy_fatigue, self.time_step))
            pre_state_type, pre_subtask,_, _, _, _= self.subtask_level_f_history[-1]
            if pre_state_type != state_type or pre_subtask != subtask:
                self.subtask_level_f_history.append((state_type, task, subtask, self.phy_fatigue, self.psy_fatigue, self.time_step)) #state, subtask, time_step
            _, pre_task, _ , _, _, _= self.task_levle_f_history[-1]
            if pre_task != task:
                predict_list_true = self.one_task_fatigue_prediction(task, self.phy_fatigue_ce_dic, self.phy_recovery_ce_dic)
                self.task_levle_f_history_true.append((state_type, task, self.phy_fatigue, self.psy_fatigue, self.time_step, predict_list_true))
                predict_list = self.one_task_fatigue_prediction(task, self.pfs_phy_fat_ce_dic, self.pfs_phy_rec_ce_dic)
                self.task_levle_f_history.append((state_type, task, self.phy_fatigue, self.psy_fatigue, self.time_step, predict_list)) 
                if self.activate_other_filters:
                    predict_list_kf = self.one_task_fatigue_prediction(task, self.kfs_phy_fat_ce_dic, self.kfs_phy_rec_ce_dic)
                    self.task_levle_f_history_kf.append((state_type, task, self.phy_fatigue, self.psy_fatigue, self.time_step, predict_list_kf)) #state, subtask, time_step
                    predict_list_ekf = self.one_task_fatigue_prediction(task, self.ekfs_phy_fat_ce_dic, self.ekfs_phy_rec_ce_dic)
                    self.task_levle_f_history_ekf.append((state_type, task, self.phy_fatigue, self.psy_fatigue, self.time_step, predict_list_ekf)) #state, subtask, time_step
        if self.cfg.use_partial_filter == True:
            self.update_ftg_mask(self.task_filter_phy_prediction_dic)
        else:
            self.update_ftg_mask(ftg_prediction)

        return 

    def step_filter(self, F, state_type, subtask, step_time, fatigue_coe_dic, recover_coe_dic, filter_list):
        
        if state_type in self.phy_free_state_dic:
            _filter = filter_list[state_type]
            F = F*math.exp(-recover_coe_dic[state_type]*step_time)
            measure = self.add_measure_noise(F)
            _filter.step(measure, F, self.time_step)
            # 根据过滤器类型更新相应的系数字典
            if isinstance(_filter, ParticleFilter):
                estimated_value = _filter.lambda_estimates[-1]
                if not np.isnan(estimated_value):
                    self.pfs_phy_rec_ce_dic[state_type] = estimated_value
            elif isinstance(_filter, KfRecover):
                estimated_value = _filter.mu_estimates[-1]
                if not np.isnan(estimated_value):
                    self.kfs_phy_rec_ce_dic[state_type] = estimated_value
            elif isinstance(_filter, EKfRecover):
                estimated_value = _filter.mu_estimates[-1]
                if not np.isnan(estimated_value):
                    self.ekfs_phy_rec_ce_dic[state_type] = estimated_value
        else:
            assert subtask in fatigue_coe_dic.keys()
            _filter = filter_list[subtask]
            _lambda = -fatigue_coe_dic[subtask]
            F = F + (1-F)*(1-math.exp(_lambda*step_time))
            measure = self.add_measure_noise(F)
            _filter.step(measure, F, self.time_step)
            # 根据过滤器类型更新相应的系数字典
            if isinstance(_filter, ParticleFilter):
                estimated_value = _filter.lambda_estimates[-1]
                if not np.isnan(estimated_value):
                    self.pfs_phy_fat_ce_dic[subtask] = estimated_value
            elif isinstance(_filter, KfFatigue):
                estimated_value = _filter.lambda_estimates[-1]
                if not np.isnan(estimated_value):
                    self.kfs_phy_fat_ce_dic[subtask] = estimated_value
            elif isinstance(_filter, EkfFatigue):
                estimated_value = _filter.lambda_estimates[-1]
                if not np.isnan(estimated_value):
                    self.ekfs_phy_fat_ce_dic[subtask] = estimated_value

        # 根据过滤器类型返回相应的估计值
        if isinstance(_filter, ParticleFilter):
            return _filter.lambda_estimates[-1], _filter.F_estimates[-1]
        elif isinstance(_filter, KfRecover):
            return _filter.mu_estimates[-1], _filter.R_estimates[-1]
        elif isinstance(_filter, EKfRecover):
            return _filter.mu_estimates[-1], _filter.R_estimates[-1]
        elif isinstance(_filter, KfFatigue):
            return _filter.lambda_estimates[-1], _filter.F_estimates[-1]
        elif isinstance(_filter, EkfFatigue):
            return _filter.lambda_estimates[-1], _filter.F_estimates[-1]
        else:
            # 默认情况
            return _filter.lambda_estimates[-1], _filter.F_estimates[-1]
    
    def add_measure_noise(self, F):
        random = np.random.normal(self.cfg.measure_noise_mu, self.cfg.measure_noise_sigma, 1)
        # random = 0.
        # random = np.clip(random, -0.1, 0.1)
        F = np.clip(F + random, 0.0, 1.0) 
        return F
    
    def update_ftg_mask(self, prediction : dict = None):

        if prediction is not None:
            #adopt rule based mask
            _fatigue = np.array(list(prediction.values())) + self.phy_fatigue
            _mask = np.where(_fatigue < self.ftg_thresh_phy, 1, 0)
            self.ftg_task_mask = torch.from_numpy(_mask) 
            self.ftg_task_mask[0] = 1
        else:
            pass
        
        return

    def update_predict_dic(self):
        # step_time_scale = (1+self.hyper_param_time*math.log(1+self.phy_fatigue))
        # self.task_phy_prediction_dic = {task: self.phy_fatigue  for (key, task) in high_level_task_dic.items()} 
        # self.task_psy_prediction_dic = {task: self.psy_fatigue  for (key, task) in high_level_task_dic.items()} 
        # for key, v in self.task_phy_prediction_dic.items():
        #     subtask_seq = self.task_human_subtasks_dic[key]
        #     for subtask in subtask_seq:
        #         time = self.ONE_STEP_TIME
        #         if 'put' in subtask or subtask == 'placing_product':
        #             time = self.cfg.human_putting_time * self.ONE_STEP_TIME * step_time_scale
        #         elif 'loading' in subtask:
        #             time = self.cfg.human_loading_time * self.ONE_STEP_TIME * step_time_scale
        #         elif subtask == 'cutting_cube':
        #             time = self.cfg.cutting_machine_oper_len * self.ONE_STEP_TIME * step_time_scale
        #         self.task_phy_prediction_dic[key] = self.step_helper_phy(self.task_phy_prediction_dic[key], subtask, subtask, time)
        #         self.task_psy_prediction_dic[key] = self.step_helper_psy(self.task_psy_prediction_dic[key], subtask, subtask, time)
        #     self.task_phy_prediction_dic[key] -= self.phy_fatigue
        #     self.task_psy_prediction_dic[key] -= self.psy_fatigue
        phy_predict = self.update_predict_helper(self.phy_fatigue_ce_dic, self.phy_recovery_ce_dic, self.phy_free_state_dic)
        return phy_predict
    
    def update_filter_predict_dic(self, filter_type = 'pf'):

        if filter_type == 'pf':
            # start_time = time.time()
            filter_phy_predict = self.update_predict_helper(self.pfs_phy_fat_ce_dic, self.pfs_phy_rec_ce_dic, self.phy_free_state_dic)
            # end_time = time.time()
            # if self.visualize_inference_time:
            #     print(f"Update filter predict helper time: {end_time - start_time}")
        elif filter_type == 'kf':
            filter_phy_predict = self.update_predict_helper(self.kfs_phy_fat_ce_dic, self.kfs_phy_rec_ce_dic, self.phy_free_state_dic)
        elif filter_type == 'ekf':
            filter_phy_predict = self.update_predict_helper(self.ekfs_phy_fat_ce_dic, self.ekfs_phy_rec_ce_dic, self.phy_free_state_dic)
        else:
            raise ValueError(f"Invalid filter type: {filter_type}")
        return filter_phy_predict

    def update_predict_helper(self, phy_ce_dic, phy_recover_ce_dic, phy_free_state_dic):
        step_time_scale = (1+self.hyper_param_time*math.log(1+self.phy_fatigue))
        phy_predict_dic = {task: self.phy_fatigue  for (key, task) in high_level_task_dic.items()}
        # psy_predict_dic = {task: self.psy_fatigue  for (key, task) in high_level_task_dic.items()} 
        for key, v in phy_predict_dic.items():
            subtask_seq = self.task_human_subtasks_dic[key]
            for subtask in subtask_seq:
                time = self.ONE_STEP_TIME
                if 'put' in subtask or subtask == 'placing_product':
                    time = self.cfg.human_putting_time * self.ONE_STEP_TIME * step_time_scale
                elif 'loading' in subtask:
                    time = self.cfg.human_loading_time * self.ONE_STEP_TIME * step_time_scale
                elif subtask == 'cutting_cube':
                    time = self.cfg.cutting_machine_oper_len * self.ONE_STEP_TIME * step_time_scale
                phy_predict_dic[key] = self.step_helper_delta_phy_fatigue(phy_predict_dic[key], subtask, subtask, time, phy_ce_dic, phy_recover_ce_dic, phy_free_state_dic)
                # psy_predict_dic[key] = self.step_helper_delta_psy_fatigue(psy_predict_dic[key], subtask, subtask, time, psy_ce_dic, psy_recover_ce_dic, phy_free_state_dic, psy_free_state_dic)
            phy_predict_dic[key] -= self.phy_fatigue
            # psy_predict_dic[key] -= self.psy_fatigue
        return phy_predict_dic

    def one_task_fatigue_prediction(self, task, phy_fatigue_ce_dic, phy_recovery_ce_dic):
        """
        返回指定高层任务下所有子任务的疲劳预测值列表
        """
        # 当前疲劳值作为初始值

        F_0 = self.phy_fatigue
        predict_list = [F_0]
        step_time_scale = (1 + self.hyper_param_time * math.log(1 + self.phy_fatigue))
        subtask_seq = self.task_human_subtasks_dic[task]
        for subtask in subtask_seq:
            time = self.ONE_STEP_TIME
            if 'put' in subtask or subtask == 'placing_product':
                time = self.cfg.human_putting_time * self.ONE_STEP_TIME * step_time_scale
            elif 'loading' in subtask:
                time = self.cfg.human_loading_time * self.ONE_STEP_TIME * step_time_scale
            elif subtask == 'cutting_cube':
                time = self.cfg.cutting_machine_oper_len * self.ONE_STEP_TIME * step_time_scale
            # 计算疲劳预测
            for i in range(0, int(time/self.ONE_STEP_TIME)):
                F_0 = self.step_helper_delta_phy_fatigue(F_0, subtask, subtask, 0.1, phy_fatigue_ce_dic, phy_recovery_ce_dic, self.phy_free_state_dic)
                predict_list.append(F_0)

        return predict_list

    def step_helper_delta_phy_fatigue(self, F_0, state_type, subtask, step_time, fatigue_coe_dic, recover_coe_dic, free_state_dic):
        # forgetting-fatigue-recovery exponential model
        # paper name: Incorporating Human Fatigue and Recovery Into the Learning–Forgetting Process
        if state_type in free_state_dic:
            F_0 = F_0*math.exp(-recover_coe_dic[state_type]*step_time)
        else:
            assert subtask in fatigue_coe_dic.keys()
            _lambda = -fatigue_coe_dic[subtask]
            F_0 = F_0 + (1-F_0)*(1-math.exp(_lambda*step_time))
        return F_0
    
    # def step_helper_phy(self, F_0, state_type, subtask, step_time, fatigue_coe, revcover_coe):
    #     # forgetting-fatigue-recovery exponential model
    #     # paper name: Incorporating Human Fatigue and Recovery Into the Learning–Forgetting Process
    #     if state_type in self.phy_free_state_dic:
    #         F_0 = F_0*math.exp(-self.phy_recovery_ce_dic[state_type]*step_time)
    #     else:
    #         assert subtask in self.phy_fatigue_ce_dic.keys()
    #         _lambda = -self.phy_fatigue_ce_dic[subtask]
    #         F_0 = F_0 + (1-F_0)*(1-math.exp(_lambda*step_time))
    #     return F_0
    
    # def step_helper_psy(self, F_0, state_type, subtask, step_time):
    #     # forgetting-fatigue-recovery exponential model
    #     # paper name: Incorporating Human Fatigue and Recovery Into the Learning–Forgetting Process
    #     if state_type in self.psy_free_state_dic:
    #         F_0 = F_0*math.exp(-self.psy_recovery_ce_dic[state_type]*step_time)
    #     else:
    #         assert subtask in self.psy_fatigue_ce_dic.keys()
    #         _lambda = -self.psy_fatigue_ce_dic[subtask]
    #         F_0 = F_0 + (1-F_0)*(1-math.exp(_lambda*step_time))
    #     return F_0
    
    def plot_curve(self):
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-white')
        # plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['pdf.fonttype'] = 42
        # plt.tick_params(axis='both', labelsize=50)
        params = {'legend.fontsize': 15,
            'legend.handlelength': 2}
        plt.rcParams.update(params)
        # plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签字体大小
        # plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签字体大小
        fig = plt.figure(figsize=(20,10), dpi=100)
        # gs = gridspec(1,4, )
        # gs = fig.add_gridspec(1,4) 
        ax = plt.subplot(111)
        # ax_2 = plt.subplot(212)
        ax.set_title('Fatigue curve, Human type:' + self.human_type, fontsize=20)
        ax.set_xlabel('time step', fontsize=15)
        ax.tick_params(axis='both', which='both', labelsize=15)
        line_labels = ['Physiological fatigue', 'Psychological fatigue']
        data = [self.phy_history, self.psy_history]
        color_dict = {'Physiological fatigue': 'crimson', 'Psychological fatigue': 'orange', 'EDQN2': 'forestgreen', 'EBQ-G': 'dodgerblue', 'EBQ-N': 'palevioletred', 'EBQ-GN':'blueviolet', "NoSp": 'silver'}
        for _data, line_label in zip(data, line_labels):
            _data = np.array(_data)
            x,y = _data[:, 1], _data[:, 0]
            ax.plot(x, y, '-', color=color_dict[line_label], label=line_label, ms=5, linewidth=2, marker='.', linestyle='dashed')
        
        vlines = np.array(self.task_levle_f_history)[:, -1]
        ax.vlines(vlines.astype(np.int32), 0, 1, linestyles='dashed', colors='silver')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        # path = os.path.dirname(__file__)
        # fig.savefig('{}.pdf'.format(path + '/' + 'polyline'), bbox_inches='tight')
    
    def plot_comprehensive_fatigue_analysis(self):
        fatigue_name_dic = {"free": None, "waiting_box": None, "approaching": None, "put_hoop_into_box": "put flange into cage", "put_bending_tube_into_box": "put bend duct into cage", 
                        'put_hoop_on_table': "put flange on side storage", 'put_bending_tube_on_table': "put bend duct on side storage", 'hoop_loading_inner': "load flange on welding station 1",
                         "hoop_loading_outer": "load flange on welding station 2", 'bending_tube_loading_inner': "load bend duct on welding station 1", 
                        'bending_tube_loading_outer': "load bend duct on welding station 2", "cutting_cube": "activate station controlling code", "placing_product": "place made product on storage"}
        recover_name_dic = {"free": "free", "waiting_box": "waiting", "approaching": "walking"}
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['pdf.fonttype'] = 42
        params = {'legend.fontsize': 12,
            'legend.handlelength': 2}
        plt.rcParams.update(params)
        
        # 检查是否有其他过滤器数据
        has_kf = hasattr(self, 'kfs_phy_fat') and len(self.kfs_phy_fat) > 0
        has_ekf = hasattr(self, 'ekfs_phy_fat') and len(self.ekfs_phy_fat) > 0
        
        # 获取所有filter的subtask
        fatigue_filters = list(self.pfs_phy_fat.keys())
        recovery_filters = list(self.pfs_phy_rec.keys())
        total_filters = len(fatigue_filters) + len(recovery_filters)
        
        # 计算子图布局：上半部分为task-level预测对比，下半部分为filter lambda估计
        if total_filters == 0:
            # 只有疲劳预测图
            fig, axes = plt.subplots(1, 1, figsize=(22, 8), dpi=100)
            ax1 = axes
        else:
            # 计算filter子图的布局
            cols = 3  # 每行3个子图
            rows = (total_filters + cols - 1) // cols  # 计算需要的行数
            
            # 创建子图：上半部分1个图，下半部分filter子图
            fig = plt.figure(figsize=(20, 12 + 2.5*rows), dpi=100)
            
            # 使用GridSpec来单独控制行高
            gs = gridspec.GridSpec(rows + 1, cols, height_ratios=[1.2] + [1]*rows)
            
            # 上半部分：task-level疲劳预测对比图
            ax1 = plt.subplot(gs[0, :])
        
        # ====== task-level预测曲线与真值对比 ======
        #构造无系数误差的预测曲线
        if hasattr(self, 'task_levle_f_history_true'):
            true_pred_segments = []
            for record in self.task_levle_f_history_true:
                _, _, _, _, cur_time, predict_list = record
                if len(predict_list) < 2:
                    continue
                seg_x = [cur_time + i for i in range(len(predict_list))]
                seg_y = predict_list
                true_pred_segments.append((seg_x, seg_y))
            # 画无系数误差的预测曲线
            for idx, (seg_x, seg_y) in enumerate(true_pred_segments):
                ax1.plot(seg_x, seg_y, color='red', linewidth=1.2, 
                         marker='x', markevery=[0, -1], markersize=6, 
                         label='True task-level predicted fatigue' if idx==0 else "")
        # 1. 构造PF预测曲线（分段画线）
        pred_segments = []
        for record in self.task_levle_f_history:
            _, _, _, _, cur_time, predict_list = record
            if len(predict_list) < 2:
                continue
            seg_x = [cur_time + i for i in range(len(predict_list))]
            seg_y = predict_list
            pred_segments.append((seg_x, seg_y))
        # 画每一段，只在首尾保留marker
        for idx, (seg_x, seg_y) in enumerate(pred_segments):
            # 只画首尾marker
            ax1.plot(seg_x, seg_y, color='blue', linewidth=1.2, 
                     marker='o', markevery=[0, -1], markersize=8, 
                     label='PF task-level predicted fatigue' if idx==0 else "")

        # 2. 构造KF预测曲线
        if has_kf and hasattr(self, 'task_levle_f_history_kf'):
            kf_pred_segments = []
            for record in self.task_levle_f_history_kf:
                _, _, _, _, cur_time, predict_list = record
                if len(predict_list) < 2:
                    continue
                seg_x = [cur_time + i for i in range(len(predict_list))]
                seg_y = predict_list
                kf_pred_segments.append((seg_x, seg_y))
            # 画KF预测
            for idx, (seg_x, seg_y) in enumerate(kf_pred_segments):
                ax1.plot(seg_x, seg_y, color='green', linewidth=1.2, 
                         marker='s', markevery=[0, -1], markersize=6, 
                         label='KF task-level predicted fatigue' if idx==0 else "")

        # 3. 构造EKF预测曲线
        if has_ekf and hasattr(self, 'task_levle_f_history_ekf'):
            ekf_pred_segments = []
            for record in self.task_levle_f_history_ekf:
                _, _, _, _, cur_time, predict_list = record
                if len(predict_list) < 2:
                    continue
                seg_x = [cur_time + i for i in range(len(predict_list))]
                seg_y = predict_list
                ekf_pred_segments.append((seg_x, seg_y))
            # 画EKF预测
            for idx, (seg_x, seg_y) in enumerate(ekf_pred_segments):
                ax1.plot(seg_x, seg_y, color='orange', linewidth=1.2, 
                         marker='^', markevery=[0, -1], markersize=4, 
                         label='EKF task-level predicted fatigue' if idx==0 else "")

        # 4. 构造真值曲线
        true_time_steps = [t for _, t in self.phy_history]
        true_fatigue = [v for v, _ in self.phy_history]
        # 5. 绘制真值
        ax1.plot(true_time_steps, true_fatigue, label='True fatigue', color='red', alpha=0.7, linewidth=1.2)
        # 6. task切换的垂直线
        task_switch_times = [record[4] for record in self.task_levle_f_history]
        ax1.vlines(task_switch_times, ymin=min(true_fatigue+[y for _, y in pred_segments for y in y]), ymax=max(true_fatigue+[y for _, y in pred_segments for y in y]), linestyles='dashed', colors='silver', alpha=0.5, label='Task switch')
        ax1.set_xlabel('Time step', fontsize=17)
        ax1.set_ylabel('Fatigue value', fontsize=17)
        ax1.tick_params(axis='both', which='both', labelsize=15)
        ax1.set_title('Task-level fatigue prediction (PF/KF/EKF) vs true value, human type: ' + self.human_type, fontsize=17, fontweight='bold') 
        ax1.legend(fontsize=15)
        ax1.grid(True, alpha=0.3)
        
        # 添加缩写说明
        ax1.text(0.64, 0.3, 'PF: Particle Filter\nKF: Kalman Filter\nEKF: Extended Kalman Filter', 
                transform=ax1.transAxes, fontsize=15, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # ====== 下半部分：filter lambda估计图 ======
        if total_filters > 0:
            # 绘制疲劳filter的lambda估计
            for i, subtask in enumerate(fatigue_filters):
                row = i // cols + 1  # +1 因为第一行是疲劳预测图
                col = i % cols
                ax = plt.subplot(gs[row, col])
                
                # PF过滤器
                pf_filter = self.pfs_phy_fat[subtask]
                true_lambda = pf_filter.true_lambda
                if len(pf_filter.lambda_estimates) > 1:
                    times = range(len(pf_filter.lambda_estimates))
                    ax.plot(times, pf_filter.lambda_estimates, '-o', color='blue', 
                           label='PF Estimated λ', linewidth=2, markersize=4)
                
                # KF过滤器
                if has_kf and subtask in self.kfs_phy_fat:
                    kf_filter = self.kfs_phy_fat[subtask]
                    if len(kf_filter.lambda_estimates) > 1:
                        times = range(len(kf_filter.lambda_estimates))
                        ax.plot(times, kf_filter.lambda_estimates, '-s', color='green', 
                               label='KF Estimated λ', linewidth=2, markersize=4)
                
                # EKF过滤器
                if has_ekf and subtask in self.ekfs_phy_fat:
                    ekf_filter = self.ekfs_phy_fat[subtask]
                    if len(ekf_filter.lambda_estimates) > 1:
                        times = range(len(ekf_filter.lambda_estimates))
                        ax.plot(times, ekf_filter.lambda_estimates, '-^', color='orange', 
                               label='EKF Estimated λ', linewidth=2, markersize=4)
                
                # 真值线
                ax.axhline(y=true_lambda, color='red', linestyle='--', 
                        label='True λ', linewidth=2)
                ax.set_title(f'Subtask: {fatigue_name_dic[subtask]}', fontsize=17, fontweight='bold')
                ax.set_xlabel('Estimation step', fontsize=15)
                ax.set_ylabel('λ value', fontsize=15)
                ax.tick_params(axis='both', which='both', labelsize=14)
                ax.legend(fontsize=13)
                ax.grid(True, alpha=0.3)
            
            # 绘制恢复filter的mu估计
            for i, state_type in enumerate(recovery_filters):
                idx = len(fatigue_filters) + i
                row = idx // cols + 1  # +1 因为第一行是疲劳预测图
                col = idx % cols
                ax = plt.subplot(gs[row, col])
                
                # PF恢复过滤器
                pf_filter = self.pfs_phy_rec[state_type]
                true_lambda = pf_filter.true_lambda
                if len(pf_filter.lambda_estimates) > 1:
                    times = range(len(pf_filter.lambda_estimates))
                    ax.plot(times, pf_filter.lambda_estimates, '-o', color='blue', 
                           label='PF Estimated μ', linewidth=2, markersize=4)
                
                # KF恢复过滤器
                if has_kf and state_type in self.kfs_phy_rec:
                    kf_filter = self.kfs_phy_rec[state_type]
                    if len(kf_filter.mu_estimates) > 1:
                        times = range(len(kf_filter.mu_estimates))
                        ax.plot(times, kf_filter.mu_estimates, '-s', color='green', 
                               label='KF Estimated μ', linewidth=2, markersize=4)
                
                # EKF恢复过滤器
                if has_ekf and state_type in self.ekfs_phy_rec:
                    ekf_filter = self.ekfs_phy_rec[state_type]
                    if len(ekf_filter.mu_estimates) > 1:
                        times = range(len(ekf_filter.mu_estimates))
                        ax.plot(times, ekf_filter.mu_estimates, '-^', color='orange', 
                               label='EKF Estimated μ', linewidth=2, markersize=4)
                
                # 真值线
                ax.axhline(y=true_lambda, color='red', linestyle='--', 
                          label='True μ', linewidth=2)
                ax.set_title(f'Subtask: {recover_name_dic[state_type]}', fontsize=17, fontweight='bold')
                ax.set_xlabel('Estimation step', fontsize=15)
                ax.set_ylabel('μ value', fontsize=15)
                ax.tick_params(axis='both', which='both', labelsize=14)
                ax.legend(fontsize=13)
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(total_filters, rows * cols):
                row = i // cols + 1
                col = i % cols
                ax = plt.subplot(gs[row, col])
                ax.set_visible(False)
        
        # plt.suptitle(f'Comprehensive Fatigue Analysis - Human type: {self.human_type} (PF/KF/EKF)', fontsize=16)
        plt.tight_layout()
        # plt.show()
        # path = os.path.dirname(__file__)
        path = '/home/xue/work/Isaac-Production/figs/filter'
        fig.savefig('{}.pdf'.format(path), bbox_inches='tight')
        a=1
    
    def get_filter_recover_coe_accuracy(self, filter_type = 'pf'):
        true_coe = np.array(list(self.phy_recovery_ce_dic.values()))
        if filter_type == 'pf':
            filter_prediction = np.array(list(self.pfs_phy_rec_ce_dic.values()))
        elif filter_type == 'kf':
            filter_prediction = np.array(list(self.kfs_phy_rec_ce_dic.values()))
        elif filter_type == 'ekf':
            filter_prediction = np.array(list(self.ekfs_phy_rec_ce_dic.values()))
        else:
            raise ValueError(f"Invalid filter type: {filter_type}")
        return np.sqrt(np.square((true_coe - filter_prediction)/true_coe).mean())

    def get_filter_fat_predict_accuracy(self, filter_type = 'pf'):
        
        true = np.array(list(self.task_phy_prediction_dic.values()))
        true = true.ravel()[np.flatnonzero(true)]
        if filter_type == 'pf':
            filter_prediction = np.array(list(self.task_filter_phy_prediction_dic.values()))
        elif filter_type == 'kf':
            filter_prediction = np.array(list(self.task_filter_phy_prediction_dic_kf.values()))
        elif filter_type == 'ekf':
            filter_prediction = np.array(list(self.task_filter_phy_prediction_dic_ekf.values()))
        else:
            raise ValueError(f"Invalid filter type: {filter_type}")
        filter_prediction = filter_prediction.ravel()[np.flatnonzero(filter_prediction)]
        accu = np.sqrt(np.square((true - filter_prediction)/true).mean())
        return accu
    
    def get_filter_fatigue_coe_accuracy(self, filter_type = 'pf'):
        none_type_num = 3
        true_coe = np.array(list(self.phy_fatigue_ce_dic.values())[none_type_num:])
        if filter_type == 'pf':
            filter_prediction = np.array(list(self.pfs_phy_fat_ce_dic.values())[none_type_num:])
        elif filter_type == 'kf':
            filter_prediction = np.array(list(self.kfs_phy_fat_ce_dic.values())[none_type_num:])
        elif filter_type == 'ekf':
            filter_prediction = np.array(list(self.ekfs_phy_fat_ce_dic.values())[none_type_num:])
        else:
            raise ValueError(f"Invalid filter type: {filter_type}")
        return np.sqrt(np.square((true_coe - filter_prediction)/true_coe).mean())


class Characters(object):

    def __init__(self, character_list, env_cfg : HRTaskAllocEnvCfg, train_cfg) -> None:
        self.character_list = character_list
        self.state_character_dic = {0:"free", 1:"approaching", 2:"waiting_box", 3:"putting_in_box", 4:"putting_on_table", 5:"loading", 6:"cutting_machine"}
        self.task_range = {'hoop_preparing', 'bending_tube_preparing', 'hoop_loading_inner', 'bending_tube_loading_inner', 'hoop_loading_outer', 'bending_tube_loading_outer', "cutting_cube", 
                           'placing_product'}
        self.sub_task_character_dic = {0:"free", 1:"put_hoop_into_box", 2:"put_bending_tube_into_box", 3:"put_hoop_on_table", 4:"put_bending_tube_on_table", 
                                    5:'hoop_loading_inner', 6:'bending_tube_loading_inner', 7:'hoop_loading_outer', 8: 'bending_tube_loading_outer', 9: 'cutting_cube', 10:'placing_product'}
        
        self.low2high_level_task_dic = {"put_hoop_into_box":"hoop_preparing", "put_bending_tube_into_box":'bending_tube_preparing', "put_hoop_on_table":'hoop_preparing', 
                        "put_bending_tube_on_table":'bending_tube_preparing', 'hoop_loading_inner':'hoop_loading_inner', 'bending_tube_loading_inner':'bending_tube_loading_inner', 
                        'hoop_loading_outer':'hoop_loading_outer', 'bending_tube_loading_outer': 'bending_tube_loading_outer', 'cutting_cube': 'cutting_cube', 'placing_product':'placing_product'}
        
        self.high2low_level_task_dic = {'hoop_preparing':'put_hoop_into_box', 'bending_tube_preparing':'put_bending_tube_into_box', 
                                'hoop_loading_inner':'hoop_loading_inner', 'bending_tube_loading_inner':'bending_tube_loading_inner', 
                                'hoop_loading_outer':'hoop_loading_outer', 'bending_tube_loading_outer':'bending_tube_loading_outer', "cutting_cube":'cutting_cube', 
                    'placing_product':'placing_product'}


        self.poses_dic = {"put_hoop_into_box": [1.28376, 6.48821, np.deg2rad(0)] , "put_bending_tube_into_box": [1.28376, 13.12021, np.deg2rad(0)], 
                        "put_hoop_on_table": [-12.26318, 4.72131, np.deg2rad(0)], "put_bending_tube_on_table":[-32, 8.0, np.deg2rad(-90)],
                        'hoop_loading_inner':[-16.26241, 6.0, np.deg2rad(180)],'bending_tube_loading_inner':[-29.06123, 6.3725, np.deg2rad(0)],
                        'hoop_loading_outer':[-16.26241, 6.0, np.deg2rad(180)], 'bending_tube_loading_outer': [-29.06123, 6.3725, np.deg2rad(0)],
                        'cutting_cube':[-29.83212, -1.54882, np.deg2rad(0)], 'placing_product':[-40.47391, 12.91755, np.deg2rad(0)],
                        'initial_pose_0':[-11.5768, 6.48821, 0.0], 'initial_pose_1':[-30.516169, 7.5748153, 0.0]}
        
        self.poses_dic2num = {"put_hoop_into_box": 0 , "put_bending_tube_into_box": 1, 
                "put_hoop_on_table": 2, "put_bending_tube_on_table":3,
                'hoop_loading_inner':4,'bending_tube_loading_inner':5,
                'hoop_loading_outer':6, 'bending_tube_loading_outer': 7,
                'cutting_cube':8, 'placing_product':9,
                'initial_pose_0':10, 'initial_pose_1':11}
        
        self.routes_dic = None

        self.picking_pose_hoop = [1.28376, 6.48821, np.deg2rad(0)] 
        self.picking_pose_bending_tube = [1.28376, 13.12021, np.deg2rad(0)] 
        self.picking_pose_table_hoop = [-12.26318, 4.72131, np.deg2rad(0)]
        self.picking_pose_table_bending_tube = [-32, 8.0, np.deg2rad(-90)]

        self.loading_pose_hoop = [-16.26241, 6.0, np.deg2rad(180)]
        self.loading_pose_bending_tube = [-29.06123, 6.3725, np.deg2rad(0)]

        self.cutting_cube_pose = [-29.83212, -1.54882, np.deg2rad(0)]

        self.placing_product_pose = [-40.47391, 12.91755, np.deg2rad(0)]
        # _cfg = HRTaskAllocEnvCfg
        self.cfg = env_cfg
        self.train_cfg = train_cfg
        self.PUTTING_TIME = env_cfg.human_putting_time
        self.LOADING_TIME = env_cfg.human_loading_time
        self.CUTTING_MACHINE_TIME = env_cfg.cutting_machine_oper_len
        self.RANDOM_TIME = env_cfg.human_time_random
        self.hyper_param_time = env_cfg.hyper_param_time

        self.n_max_human = env_cfg.n_max_human
        self.fatigue_list : list[Fatigue] = []
        self.human_types = ["strong", "normal", "weak"]
        for i in range(0,len(self.character_list)):
            self.fatigue_list.append(Fatigue(i, self.human_types, env_cfg, self.train_cfg))
        self.fatigue_task_masks = None
        return
    
    def reset(self, acti_num_charc = None, random = None):
        if acti_num_charc is None:
            acti_num_charc = np.random.randint(1, 4)
        self.acti_num_charc = acti_num_charc
        self.states = [0]*acti_num_charc
        self.tasks = [0]*acti_num_charc
        self.movements = [0]*acti_num_charc
        self.list = self.character_list[:acti_num_charc]
        self.x_paths = [[] for i in range(acti_num_charc)]
        self.y_paths = [[] for i in range(acti_num_charc)]
        self.yaws = [[] for i in range(acti_num_charc)]
        self.path_idxs = [0 for i in range(acti_num_charc)]
        if random is None:
            random = np.random.choice(len(self.poses_dic), acti_num_charc, replace=False)
        pose_list = list(self.poses_dic.values())
        pose_str_list = list(self.poses_dic.keys())
        initial_pose_str = []
        for i in range(0, len(self.character_list)):
            if i < acti_num_charc:
                position = pose_list[random[i]][:2]+[0.0415]
                initial_pose_str.append(pose_str_list[random[i]])
                self.character_list[i].set_world_poses(torch.tensor([position]), torch.tensor([[1., 0., 0., 0.]]))
                self.character_list[i].set_velocities(torch.zeros((1,6)))
                self.reset_idx(i)
                self.reset_path(i)
            else:
                position = [0, 0, -100]
                self.character_list[i].set_world_poses(torch.tensor([position]), torch.tensor([[1., 0., 0., 0.]]))
                self.character_list[i].set_velocities(torch.zeros((1,6)))
        
        self.loading_operation_time_steps = [0. for i in range(acti_num_charc)]
        
        #1 is avaiable, 0 means worker is over fatigue threshold
        self.fatigue_task_masks = torch.zeros((self.n_max_human, len(high_level_task_dic)), dtype=torch.int32)
        for i in range(0, acti_num_charc):
            fatigue : Fatigue = self.fatigue_list[i]
            fatigue.reset()
            self.fatigue_task_masks[i] = fatigue.ftg_task_mask
        self.cost_mask_from_net = None
        self.poses_str = initial_pose_str

        return initial_pose_str

    def update_pose_str(self, idx):
        worker_position = self.list[idx].get_world_poses()
        wp = world_pose_to_navigation_pose(worker_position)
        wp_str = find_closest_pose(pose_dic=self.poses_dic, ego_pose=wp, in_dis=1000.)
        self.poses_str[idx] = wp_str

    # def get_fatigue_task_masks(self):
    #     fatigue_task_masks = torch.zeros((self.n_max_human, len(high_level_task_dic)), device=self.cfg.cuda_device_str)
    #     for i in range(0, self.acti_num_charc):
    #         fatigue_task_masks[i] = self.fatigue_list[i].ftg_task_mask
    #     return fatigue_task_masks

    def reset_idx(self, idx):
        if idx < 0 :
            return
        self.states[idx] = 0
        self.tasks[idx] = 0

    def assign_task(self, high_level_task, random = False):
        #todo 
        if high_level_task not in self.task_range:
            return -2

        # _fatigue_mask = self.fatigue_task_masks[:self.acti_num_charc, _fatigue_mask_idx]
        #task == 0 means the human doing no task, is free
        # np_task = np.array(self.tasks)
        # np_task = np.where(_fatigue_mask, np_task, -1)
        worker_tasks = self.tasks
        _fatigue_mask_idx = high_level_task_rev_dic[high_level_task] + 1
        if self.train_cfg['use_fatigue_mask']:
            _fatigue_mask = self.fatigue_task_masks[:self.acti_num_charc, _fatigue_mask_idx].tolist()
            worker_tasks = [self.tasks[i] if _fatigue_mask[i] else -1 for i in range(len(_fatigue_mask))]
        elif self.cost_mask_from_net is not None:
            assert len(self.cost_mask_from_net) == 1, "error cost mask shape from cost function"
            _fatigue_mask = self.cost_mask_from_net[0, _fatigue_mask_idx, :]
            _fatigue_mask[self.acti_num_charc:] = 0
            worker_tasks = [ self.tasks[i] if _fatigue_mask[i].item() else -1 for i in range(len(_fatigue_mask))]
        
        # elif self.cfg.use_partial_filter:
            
        if random:
            idx = random_zero_index(worker_tasks)
        else:
            idx = self.find_available_charac(worker_tasks, high_level_task)
            
        if idx == -1:
            return idx
        if high_level_task == 'hoop_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 1 
        elif high_level_task == 'bending_tube_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 2
        elif high_level_task == 'hoop_loading_inner':
            self.tasks[idx] = 5
        elif high_level_task == 'bending_tube_loading_inner':
            self.tasks[idx] = 6
        elif high_level_task == 'hoop_loading_outer':
            self.tasks[idx] = 7
        elif high_level_task == 'bending_tube_loading_outer':
            self.tasks[idx] = 8
        elif high_level_task == 'cutting_cube':
            if random:
                self.tasks[idx] = 9
            #TODO warning
            else:
                for _idx in range(0, len(self.list)):
                    xyz, _ = self.list[_idx].get_world_poses()
                    if worker_tasks[_idx] == 0 and xyz[0][0] < -22:
                        self.tasks[_idx] = 9
                        return _idx
                self.tasks[idx] = 9
        elif high_level_task == 'placing_product':
            self.tasks[idx] = 10
        
        return idx
    
    # def find_available_charac(self, mask : list, idx=0):
    #     try:
    #         return mask.index(idx)
    #     except: 
    #         return -1

    def find_available_charac(self, mask, task, idx=0):
        # try:
        #     return self.tasks.index(idx)
        # except: 
        #     return -1
        count = mask.count(0)
        if count == 0:
            return -1
        elif count == 1:
            return mask.index(idx)
        else:
            task_pose_str = self.high2low_level_task_dic[task]
            closet_idx = None
            shortest_path = None
            for i in range(0, len(mask)):
                if mask[i] != 0:
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

    def step_next_pose(self, charac_idx = 0):
        reaching_flag = False
        #skip the initial pose
        # if len(self.x_paths[agv_idx]) == 0:
        #     position = [current_pose[0], current_pose[1], 0]
        #     euler_angles = [0,0, current_pose[2]]
        #     return position, quaternion.eulerAnglesToQuaternion(euler_angles), True

        self.path_idxs[charac_idx] += 1
        path_idx = self.path_idxs[charac_idx]
        # if agv_idx == 0:
        #     a = 1
        if path_idx == (len(self.x_paths[charac_idx]) - 1):
            reaching_flag = True
            position = [self.x_paths[charac_idx][-1], self.y_paths[charac_idx][-1], 0]
            euler_angles = [0,0, self.yaws[charac_idx][-1]]
        else:
            position = [self.x_paths[charac_idx][path_idx], self.y_paths[charac_idx][path_idx], 0]
            euler_angles = [0,0, self.yaws[charac_idx][path_idx]]

        orientation = quaternion.eulerAnglesToQuaternion(euler_angles)
        self.movements[charac_idx] +=1
        return position, orientation, reaching_flag
    
    def step_processing(self, idx):
        fatigue : Fatigue = self.fatigue_list[idx]
        step_time = 1/(1+self.hyper_param_time*math.log(1+fatigue.phy_fatigue)) 
        return step_time

    def step_fatigue(self, idx, state, subtask, task, ftg_prediction = None):
        
        state_type = self.state_character_dic[state]
        subtask = self.sub_task_character_dic[subtask]
        fatigue : Fatigue = self.fatigue_list[idx]
        if task == -1:
            task = 'none'
        fatigue.step(state_type, subtask, task, ftg_prediction)
        self.fatigue_task_masks[idx] = fatigue.ftg_task_mask

    def get_fatigue(self, idx):
        fatigue : Fatigue = self.fatigue_list[idx]
        return fatigue.phy_fatigue, fatigue.psy_fatigue
    
    def have_overwork(self):
        for i in range(self.acti_num_charc):
            if self.fatigue_list[i].have_overwork():
                return True
        return False
    
    def get_overwork_phy_values(self):
        overwork_phy_values = []
        for i in range(self.acti_num_charc):
            if self.fatigue_list[i].have_overwork():
                overwork_phy_values.append(self.fatigue_list[i].phy_fatigue)
        return overwork_phy_values
    
    def compute_fatigue_cost(self):
        cost = []
        for i in range(self.acti_num_charc):
            cost.append(self.fatigue_list[i].compute_fatigue_cost()) 
        return np.mean(cost)

    def reset_path(self, charac_idx):
        self.x_paths[charac_idx] = []
        self.y_paths[charac_idx] = []
        self.yaws[charac_idx] = []
        self.path_idxs[charac_idx] = 0

    def low2high_level_task_mapping(self, task):
        task = self.sub_task_character_dic[task]
        if task in self.low2high_level_task_dic.keys():
            return self.low2high_level_task_dic[task]
        else: return -1
    
    def get_sum_movement(self):
        return sum(self.movements)