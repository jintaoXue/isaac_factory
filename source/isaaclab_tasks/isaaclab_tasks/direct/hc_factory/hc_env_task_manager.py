"""任务与资源运行时类。

hc_task_cfg 中的 HumanConfig / AgvConfig / MaterialConfig 等是「枚举与常量的配置命名空间」；
本文件里的 Agvs、Materials 承担 Isaac 仿真状态（路径、索引、列表等）。TaskManager 继承 TaskConfig。
"""
import copy

import numpy as np
import torch

from ...utils import quaternion
from .human_fatigue_model import Characters
from .hc_env_cfg import HcEnvCfg
from .hc_task_cfg import TaskConfig

# 未使用的工人/AGV 槽位
_IDX_UNUSED = -2


def random_zero_index(data):
    if data.count(0) >= 1:
        indexs = np.argwhere(np.array(data) == 0)
        _idx = np.random.randint(low=0, high=len(indexs))
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
    assert dis < in_dis, "error when get closest pose, distance is: {}".format(dis)
    return key


class TaskManager(TaskConfig):
    """高层任务表、资源需求来自 TaskConfig；本类负责任务分配与疲劳快照等运行时逻辑。"""

    @classmethod
    def _default_resource_req(cls) -> dict:
        return {"human": True, "agv": True}

    @classmethod
    def _resource_req(cls, task: str) -> dict:
        r = dict(cls._default_resource_req())
        r.update(cls.TASK_RESOURCE_REQUIREMENTS.get(task, {}))
        r.pop("box", None)
        r.pop("machine", None)
        return r

    def __init__(self, character_list, agv_list, cuda_device, env_cfg, train_cfg) -> None:
        super().__init__()
        self.cuda_device = cuda_device
        self.characters = Characters(character_list=character_list, env_cfg=env_cfg, train_cfg=train_cfg)
        self.agvs = Agvs(agv_list=agv_list, env_cfg=env_cfg, train_cfg=train_cfg)
        self.task_dic = dict(self.TASK_DICT)
        self.task_in_set = set()
        self.task_in_dic = {}
        self.task_dic_inverse = dict(self.TASK_DICT_INVERSE)
        self.cfg = env_cfg
        self._test = train_cfg["test"]
        if self._test:
            self._eval_times = train_cfg["test_times"]
            self.acti_num_agv = train_cfg["acti_agv"]
            self.acti_num_charc = train_cfg["acti_charc"]
        self.obs = None

    def reset(self, acti_num_charc, acti_num_agv):
        assert not ((acti_num_charc is None) ^ (acti_num_agv is None)), "warning"
        if self._test:
            if acti_num_charc is None:
                acti_num_agv = self.acti_num_agv
                acti_num_charc = self.acti_num_charc
        elif acti_num_charc is None:
            hm, rm = self.cfg.n_max_human, self.cfg.n_max_robot
            if hm > 0:
                acti_num_charc = np.random.randint(0, hm + 1)
            else:
                acti_num_charc = 0
            if rm > 0:
                acti_num_agv = np.random.randint(0, rm + 1)
            else:
                acti_num_agv = 0
        self.ini_worker_pose = self.characters.reset(acti_num_charc)
        self.ini_agv_pose = self.agvs.reset(acti_num_agv)
        self.task_in_set = set()
        self.task_in_dic = {}
        self.fatigue_data = {}
        self.fatigue_data_list = []

    def assign_task(self, task):
        if task not in self.TASK_DICT.values():
            return False
        req = self._resource_req(task)

        charac_idx = _IDX_UNUSED
        if req.get("human", True):
            charac_idx = self.characters.assign_task(task, random=False)
            if charac_idx == -1:
                return False
            if task in self.characters.task_range and charac_idx != -1:
                self.fatigue_data[task] = copy.deepcopy(self.obs)
                self.fatigue_data[task]["phy_fatigue"] = torch.tensor(
                    [self.characters.fatigue_list[charac_idx].phy_fatigue], dtype=torch.float32
                )
                self.fatigue_data[task]["psy_fatigue"] = torch.tensor(
                    [self.characters.fatigue_list[charac_idx].psy_fatigue], dtype=torch.float32
                )
                self.fatigue_data[task]["charac_idx"] = torch.tensor(charac_idx, dtype=torch.int64)
                self.fatigue_data[task]["task_str"] = task
                tid = self.task_dic_inverse.get(task, -1)
                self.fatigue_data[task]["action"] = torch.tensor(tid + 1, dtype=torch.int32)
                fl = self.characters.fatigue_list[charac_idx]
                self.fatigue_data[task]["filter_phy_delta_predict"] = torch.tensor(
                    [fl.task_filter_phy_prediction_dic[task]], dtype=torch.float32
                )
                self.fatigue_data[task]["filter_phy_rec_coe_accuracy"] = torch.tensor(
                    fl.get_filter_recover_coe_accuracy(filter_type="pf"), dtype=torch.float32
                )
                self.fatigue_data[task]["filter_phy_fat_coe_accuracy"] = torch.tensor(
                    fl.get_filter_fatigue_coe_accuracy(filter_type="pf"), dtype=torch.float32
                )
                self.fatigue_data[task]["phy_delta_predict"] = torch.tensor(
                    [fl.task_phy_prediction_dic[task]], dtype=torch.float32
                )
                self.fatigue_data[task]["psy_delta_predict"] = torch.tensor(
                    [fl.task_psy_prediction_dic[task]], dtype=torch.float32
                )
                self.fatigue_data["activate_other_filters"] = False
                if fl.activate_other_filters:
                    self.fatigue_data["activate_other_filters"] = True
                    self.fatigue_data[task]["filter_phy_delta_predict_kf"] = torch.tensor(
                        [fl.task_filter_phy_prediction_dic_kf[task]], dtype=torch.float32
                    )
                    self.fatigue_data[task]["filter_phy_rec_coe_accuracy_kf"] = torch.tensor(
                        fl.get_filter_recover_coe_accuracy(filter_type="kf"), dtype=torch.float32
                    )
                    self.fatigue_data[task]["filter_phy_fat_coe_accuracy_kf"] = torch.tensor(
                        fl.get_filter_fatigue_coe_accuracy(filter_type="kf"), dtype=torch.float32
                    )
                    self.fatigue_data[task]["filter_phy_delta_predict_ekf"] = torch.tensor(
                        [fl.task_filter_phy_prediction_dic_ekf[task]], dtype=torch.float32
                    )
                    self.fatigue_data[task]["filter_phy_rec_coe_accuracy_ekf"] = torch.tensor(
                        fl.get_filter_recover_coe_accuracy(filter_type="ekf"), dtype=torch.float32
                    )
                    self.fatigue_data[task]["filter_phy_fat_coe_accuracy_ekf"] = torch.tensor(
                        fl.get_filter_fatigue_coe_accuracy(filter_type="ekf"), dtype=torch.float32
                    )
        else:
            charac_idx = _IDX_UNUSED

        agv_idx = _IDX_UNUSED
        if req.get("agv", True):
            agv_idx = self.agvs.assign_task(task, random=False)
        else:
            agv_idx = _IDX_UNUSED

        lacking_resource = False
        if req.get("human", True) and charac_idx == -1:
            lacking_resource = True
        if req.get("agv", True) and agv_idx == -1:
            lacking_resource = True
        assert not lacking_resource, "lacking resource problem"

        self.task_in_set.add(task)
        self.task_in_dic[task] = {
            "charac_idx": charac_idx,
            "agv_idx": agv_idx,
            "lacking_resource": lacking_resource,
        }
        return True

    def task_clearing(self, task):
        charac_idx = self.task_in_dic[task]["charac_idx"]
        agv_idx = self.task_in_dic[task]["agv_idx"]
        task_range = self.characters.task_range
        task_range.add("none")
        if task in task_range and charac_idx >= 0:
            self.fatigue_data[task]["next_phy_fatigue"] = torch.tensor(
                [self.characters.fatigue_list[charac_idx].phy_fatigue], dtype=torch.float32
            )
            self.fatigue_data[task]["next_psy_fatigue"] = torch.tensor(
                [self.characters.fatigue_list[charac_idx].psy_fatigue], dtype=torch.float32
            )
            del self.fatigue_data[task]["task_str"]
            self.fatigue_data_list.append(self.fatigue_data[task])
            del self.fatigue_data[task]
        self.characters.reset_idx(charac_idx)
        self.agvs.reset_idx(agv_idx)
        self.task_in_set.remove(task)
        del self.task_in_dic[task]

    def step(self):
        for task in self.task_in_set:
            if not self.task_in_dic[task]["lacking_resource"]:
                continue
            req = self._resource_req(task)
            if req.get("human", True) and self.task_in_dic[task]["charac_idx"] == -1:
                self.task_in_dic[task]["charac_idx"] = self.characters.assign_task(task)
            if req.get("agv", True) and self.task_in_dic[task]["agv_idx"] == -1:
                self.task_in_dic[task]["agv_idx"] = self.agvs.assign_task(task)
            idx_vals = [self.task_in_dic[task]['charac_idx'], self.task_in_dic[task]['agv_idx']]
            need = [req.get("human", True), req.get("agv", True)]
            any_missing = any(need[i] and idx_vals[i] == -1 for i in range(2))
            self.task_in_dic[task]["lacking_resource"] = any_missing

    def corresp_charac_agv_box_idx(self, task):
        """兼容旧接口：第三项为箱子索引，已无箱子恒为 -1。"""
        if task not in self.task_in_dic.keys():
            return -1, -1, -1
        d = self.task_in_dic[task]
        return d["charac_idx"], d["agv_idx"], -1


class Materials(object):

    def __init__(
        self,
        cube_list: list,
        hoop_list: list,
        bending_tube_list: list,
        upper_tube_list: list,
        product_list: list,
    ) -> None:
        self.cube_list = cube_list
        self.upper_tube_list = upper_tube_list
        self.hoop_list = hoop_list
        self.bending_tube_list = bending_tube_list
        self.product_list = product_list

        self.cube_state_dic = {
            -1: "done",
            0: "wait",
            1: "in_list",
            2: "conveying",
            3: "conveyed",
            4: "cutting",
            5: "cut_done",
            6: "pick_up_place_cut",
            7: "placed_station_inner",
            8: "placed_station_outer",
            9: "welding_left",
            10: "welding_right",
            11: "welding_upper",
            12: "process_done",
            13: "pick_up_place_product",
        }
        self.hoop_state_dic = {0: "wait", 1: "in_container", 2: "on_table"}
        self.bending_tube_state_dic = {0: "wait", 1: "in_container", 2: "on_table"}
        self.product_state_dic = {0: "waitng", 1: "collected", 2: "placed"}

        self.initial_hoop_pose = [obj.get_world_poses() for obj in self.hoop_list]
        self.initial_bending_tube_pose = [obj.get_world_poses() for obj in self.bending_tube_list]
        self.initial_upper_tube_pose = [obj.get_world_poses() for obj in self.upper_tube_list]
        self.initial_cube_pose = [obj.get_world_poses() for obj in self.cube_list]
        self.initial_product_pose = [obj.get_world_poses() for obj in self.product_list]

        position = [[[-14.44042, 4.77828, 0.6]], [[-13.78823, 4.77828, 0.6]], [[-14.44042, 5.59765, 0.6]], [[-13.78823, 5.59765, 0.6]]]
        orientation = [[1, 0, 0, 0]]
        self.position_depot_hoop = torch.tensor(position, dtype=torch.float32)
        self.orientation_depot_hoop = torch.tensor(orientation, dtype=torch.float32)

        position = [[[-31.64901, 4.40483, 1.1]], [[-30.80189, 4.40483, 1.1]], [[-31.64901, 5.31513, 1.1]], [[-30.80189, 5.31513, 1.1]]]
        orientation = [[-1.6081e-16, -6.1232e-17, 1.0000e00, -6.1232e-17]]
        self.position_depot_bending_tube = torch.tensor(position, dtype=torch.float32)
        self.orientation_depot_bending_tube = torch.tensor(orientation, dtype=torch.float32)

        position = [[[-35.0, 15.0, 0]], [[-35, 16, 0]], [[-35, 17, 0]], [[-35, 18, 0]], [[-35, 19, 0]]]
        orientation = [[1, 0, 0, 0]]
        self.position_depot_product = torch.tensor(position, dtype=torch.float32)
        self.orientation_depot_product = torch.tensor(orientation, dtype=torch.float32)

    def reset(self):
        obj_list = (
            self.hoop_list
            + self.bending_tube_list
            + self.upper_tube_list
            + self.cube_list
            + self.product_list
        )
        pose_list = (
            self.initial_hoop_pose
            + self.initial_bending_tube_pose
            + self.initial_upper_tube_pose
            + self.initial_cube_pose
            + self.initial_product_pose
        )
        for obj, pose in zip(obj_list, pose_list):
            obj.set_world_poses(pose[0], pose[1])
            obj.set_velocities(torch.zeros((1, 6)))

        self.cube_states = [0] * len(self.cube_list)
        self.hoop_states = [0] * len(self.hoop_list)
        self.bending_tube_states = [0] * len(self.bending_tube_list)
        self.upper_tube_states = [0] * len(self.upper_tube_list)
        self.product_states = [0] * len(self.product_list)

        self.hoop_convey_states = [0] * len(self.hoop_list)
        self.bending_tube_convey_states = [0] * len(self.bending_tube_list)
        self.cube_convey_index = -1
        self.cube_cut_index = -1
        self.pick_up_place_cube_index = -1
        self.pick_up_place_upper_tube_index = -1
        self.inner_hoop_processing_index = -1
        self.inner_cube_processing_index = -1
        self.inner_bending_tube_processing_index = -1
        self.inner_upper_tube_processing_index = -1
        self.inner_bending_tube_loading_index = -1
        self.outer_hoop_processing_index = -1
        self.outer_cube_processing_index = -1
        self.outer_bending_tube_processing_index = -1
        self.outer_upper_tube_processing_index = -1
        self.outer_bending_tube_loading_index = -1
        self.pre_progress = 0

    def done(self):
        return min(self.product_states) == 2

    def progress(self):
        totall_progress = 2 * len(self.product_states)
        progress = 2 * self.product_states.count(2) + self.product_states.count(1)
        return progress / totall_progress if totall_progress else 0.0

    def produce_product_req(self):
        try:
            self.product_states.index(0)
            return True
        except ValueError:
            return False

    def have_collecting_product_req(self):
        will_have_product = any(10 <= x <= 13 for x in self.cube_states)
        prepared_hoops = any(2 <= x <= 5 for x in self.hoop_states)
        prepared_bending_tube = any(2 <= x <= 5 for x in self.bending_tube_states)
        return will_have_product or (prepared_hoops and prepared_bending_tube)


class Agvs(object):

    def __init__(self, agv_list, env_cfg: HcEnvCfg, train_cfg) -> None:
        self.agv_list = agv_list
        self.state_dic = {0: "free", 1: "moving_to_pickup", 2: "carrying_load", 3: "waiting"}
        self.sub_task_dic = {
            0: "free",
            1: "carry_box_to_hoop",
            2: "carry_box_to_bending_tube",
            3: "carry_box_to_hoop_table",
            4: "carry_box_to_bending_tube_table",
            5: "collect_product",
            6: "placing_product",
        }
        self.task_range = {"hoop_preparing", "bending_tube_preparing", "collect_product", "placing_product"}
        self.low2high_level_task_dic = {
            "carry_box_to_hoop": "hoop_preparing",
            "carry_box_to_bending_tube": "bending_tube_preparing",
            "carry_box_to_hoop_table": "hoop_preparing",
            "carry_box_to_bending_tube_table": "bending_tube_preparing",
            "collect_product": "collect_product",
            "placing_product": "placing_product",
        }

        # 键名与 routes_agv.pkl 中路网名保持一致
        self.poses_dic = {
            "carry_box_to_hoop": [-0.654, 8.0171, np.deg2rad(0)],
            "carry_box_to_bending_tube": [-0.654, 11.62488, np.deg2rad(0)],
            "carry_box_to_hoop_table": [-11.69736, 5.71486, np.deg2rad(0)],
            "carry_box_to_bending_tube_table": [-33.55065, 5.71486, np.deg2rad(-90)],
            "collect_product": [-21.76757, 10.78427, np.deg2rad(0)],
            "placing_product": [-38.54638, 12.40097, np.deg2rad(0)],
            "initial_pose_0": [-4.8783107, 8.017096, 0.0],
            "initial_pose_1": [-4.8726454, 11.656976, 0.0],
        }
        self.routes_dic = None
        self.collecting_product_pose = [-21.76757, 10.78427, np.deg2rad(0)]
        self.placing_product_pose = [-38.54638, 12.40097, np.deg2rad(0)]
        self.gantt_chart_data = train_cfg["gantt_chart_data"]

    def reset(self, acti_num_agv=None, random=None):
        self.acti_num_agv = acti_num_agv
        self.states = [0] * acti_num_agv
        self.tasks = [0] * acti_num_agv
        self.movements = [0] * acti_num_agv
        self.list = self.agv_list[:acti_num_agv]
        self.x_paths = [[] for _ in range(acti_num_agv)]
        self.y_paths = [[] for _ in range(acti_num_agv)]
        self.yaws = [[] for _ in range(acti_num_agv)]
        self.path_idxs = [0 for _ in range(acti_num_agv)]
        if random is None:
            nposes = len(self.poses_dic)
            if acti_num_agv <= 0:
                random = np.array([], dtype=int)
            else:
                random = np.random.choice(nposes, acti_num_agv, replace=False)
        pose_list = list(self.poses_dic.values())
        pose_str_list = list(self.poses_dic.keys())
        initial_pose_str = []
        for i in range(0, len(self.agv_list)):
            if i < acti_num_agv:
                initial_pose_str.append(pose_str_list[random[i]])
                position = pose_list[random[i]][:2] + [0.1]
                self.agv_list[i].set_world_poses(torch.tensor([position]), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
                self.agv_list[i].set_velocities(torch.zeros((1, 6)))
                self.reset_idx(i)
                self.reset_path(i)
            else:
                position = [0, 0, -100]
                self.agv_list[i].set_world_poses(torch.tensor([position]), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
                self.agv_list[i].set_velocities(torch.zeros((1, 6)))

        self.poses_str = initial_pose_str
        if self.gantt_chart_data:
            self.time_step = []
            self.task_level_history = []
            for i in range(acti_num_agv):
                self.task_level_history.append([("free", "free", "none", 0)])
                self.time_step.append(0)
        return initial_pose_str

    def reset_idx(self, idx):
        if idx < 0:
            return
        self.tasks[idx] = 0
        self.states[idx] = 0

    def assign_task(self, high_level_task, random=False):
        if high_level_task not in self.task_range:
            return -2

        if random:
            idx = random_zero_index([a * b for a, b in zip(self.states, self.tasks)])
        else:
            available = [a * b for a, b in zip(self.states, self.tasks)]
            try:
                idx = available.index(0)
            except ValueError:
                idx = -1

        if idx == -1:
            return idx
        if high_level_task == "hoop_preparing":
            self.tasks[idx] = 1
        elif high_level_task == "bending_tube_preparing":
            self.tasks[idx] = 2
        elif high_level_task == "collect_product":
            self.tasks[idx] = 5
        elif high_level_task == "placing_product":
            self.tasks[idx] = 6
        return idx

    def step_next_pose(self, agv_idx):
        reaching_flag = False
        self.path_idxs[agv_idx] += 1
        path_idx = self.path_idxs[agv_idx]
        if path_idx >= (len(self.x_paths[agv_idx])):
            reaching_flag = True
            position = [self.x_paths[agv_idx][-1], self.y_paths[agv_idx][-1], 0]
            euler_angles = [0, 0, self.yaws[agv_idx][-1]]
        else:
            position = [self.x_paths[agv_idx][path_idx], self.y_paths[agv_idx][path_idx], 0]
            euler_angles = [0, 0, self.yaws[agv_idx][path_idx]]

        orientation = quaternion.eulerAnglesToQuaternion(euler_angles)
        self.movements[agv_idx] += 1
        return position, orientation, reaching_flag

    def step_gantt_chart(self, idx, state, subtask, task):
        self.time_step[idx] += 1
        if task == -1:
            task = "none"
        self.task_level_history[idx].append(
            (self.state_dic[state], self.sub_task_dic[subtask], task, self.time_step[idx])
        )

    def reset_path(self, agv_idx):
        self.x_paths[agv_idx] = []
        self.y_paths[agv_idx] = []
        self.yaws[agv_idx] = []
        self.path_idxs[agv_idx] = 0

    def low2high_level_task_mapping(self, task):
        task = self.sub_task_dic[task]
        if task in self.low2high_level_task_dic.keys():
            return self.low2high_level_task_dic[task]
        else:
            return -1

    def get_sum_movement(self):
        return sum(self.movements)
