"""Auto-generated env_state_action_dict sample (offline dissection / preprocess smoke).

Regenerate by dumping a live env dict at a breakpoint (write EnvStateActionSample yourself),
or keep this file and run: ``python .../src/buffer_preprocess.py``.
"""
from __future__ import annotations

import torch

# source dump path: /home/xue/work/isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/env_asset_cfg/cfg_env_state_action_sample.py

EnvStateActionSample: dict = {
    'time_step': 116,
    'episode_num': 0,
    'machine': {
        'num00_rotaryPipeAutomaticWeldingMachine': {
            'state': ['free', 'free'],
            'processing_time_step': [0, 0],
            'target_joints_position': [None, None],
            'ongoing_task_record_index': [None, None],
            'key_variables': {
                'type_name': 'num00_rotaryPipeAutomaticWeldingMachine',
                'working_area_ids': {
                    'num00_rotaryPipeAutomaticWeldingMachine_part_01_station': {
                        'human_working_areas_ids': [56],
                        'robot_parking_areas_ids': [57],
                        'gantry_parking_areas_ids': [56],
                    },
                    'num00_rotaryPipeAutomaticWeldingMachine_part_02_station': {
                        'human_working_areas_ids': [60],
                        'robot_parking_areas_ids': [61],
                        'gantry_parking_areas_ids': [60],
                    },
                },
                'num_workstations': 2,
                'material_placement_cfg': {
                    'num00_rotaryPipeAutomaticWeldingMachine_part_01_station': {
                        'position': torch.tensor([43.12303161621094, 15.737210273742676, 1.1460000276565552], dtype=torch.float32),
                        'orientation': torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
                    },
                    'num00_rotaryPipeAutomaticWeldingMachine_part_02_station': {
                        'position': torch.tensor([35.143211364746094, 15.737210273742676, 1.1460000276565552], dtype=torch.float32),
                        'orientation': torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
                    },
                },
            },
        },
        'num01_weldingRobot': {
            'state': ['free'],
            'processing_time_step': [0],
            'target_joints_position': [None],
            'ongoing_task_record_index': [None],
            'key_variables': {
                'type_name': 'num01_weldingRobot',
                'working_area_ids': {
                    'num01_weldingRobot_part02_robot_arm_and_base': {
                        'human_working_areas_ids': [66],
                        'robot_parking_areas_ids': [65],
                        'gantry_parking_areas_ids': [66],
                    },
                },
                'num_workstations': 1,
                'material_placement_cfg': {
                    'num01_weldingRobot_part02_robot_arm_and_base': {
                        'position': torch.tensor([24.687070846557617, 14.369290351867676, 1.1460000276565552], dtype=torch.float32),
                        'orientation': torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
                    },
                },
            },
        },
        'num02_rollerbedCNCPipeIntersectionCuttingMachine': {
            'state': ['working_logistic_for_pipe_cutting'],
            'processing_time_step': [0],
            'target_joints_position': [None],
            'ongoing_task_record_index': [0],
            'key_variables': {
                'type_name': 'num02_rollerbedCNCPipeIntersectionCuttingMachine',
                'working_area_ids': {
                    'num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station': {
                        'human_working_areas_ids': [90],
                        'robot_parking_areas_ids': [78],
                        'gantry_parking_areas_ids': [78],
                    },
                },
                'num_workstations': 1,
                'material_placement_cfg': {
                    'num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station': {
                        'position': torch.tensor([10.067359924316406, 16.869869232177734, 1.1460000276565552], dtype=torch.float32),
                        'orientation': torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
                    },
                },
            },
        },
        'num03_laserCuttingMachine': {
            'state': ['free'],
            'processing_time_step': [0],
            'target_joints_position': [None],
            'ongoing_task_record_index': [None],
            'key_variables': {
                'type_name': 'num03_laserCuttingMachine',
                'working_area_ids': {
                    'num03_laserCuttingMachine': {
                        'human_working_areas_ids': [113],
                        'robot_parking_areas_ids': [111],
                        'gantry_parking_areas_ids': [111],
                    },
                },
                'num_workstations': 1,
                'material_placement_cfg': {
                    'num03_laserCuttingMachine': {
                        'position': torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                        'orientation': torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
                    },
                },
            },
        },
        'num04_groovingMachineLarge': {
            'state': ['free'],
            'processing_time_step': [0],
            'target_joints_position': [None],
            'ongoing_task_record_index': [None],
            'key_variables': {
                'type_name': 'num04_groovingMachineLarge',
                'working_area_ids': {
                    'num04_groovingMachineLarge_part01_large_fixed_base': {
                        'human_working_areas_ids': [138],
                        'robot_parking_areas_ids': [136],
                        'gantry_parking_areas_ids': [136],
                    },
                },
                'num_workstations': 1,
                'material_placement_cfg': {
                    'num04_groovingMachineLarge_part01_large_fixed_base': {
                        'position': torch.tensor([-7.647950172424316, 15.335689544677734, 1.458109974861145], dtype=torch.float32),
                        'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, -0.707099974155426], dtype=torch.float32),
                    },
                },
            },
        },
        'num05_groovingMachineSmall': {
            'state': ['free'],
            'processing_time_step': [0],
            'ongoing_task_record_index': [None],
            'key_variables': {
                'type_name': 'num05_groovingMachineSmall',
                'working_area_ids': {
                    'num05_groovingMachineSmall_part01_small_fixed_base': {
                        'human_working_areas_ids': [160],
                        'robot_parking_areas_ids': [139],
                        'gantry_parking_areas_ids': [139],
                    },
                },
                'num_workstations': 1,
                'material_placement_cfg': {
                    'num05_groovingMachineSmall_part01_small_fixed_base': {
                        'position': torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                        'orientation': torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
                    },
                },
            },
            'target_joints_position': [None],
        },
        'num06_highPressureFoamingMachine': {
            'state': ['free'],
            'processing_time_step': [0],
            'ongoing_task_record_index': [None],
            'key_variables': {
                'type_name': 'num06_highPressureFoamingMachine',
                'working_area_ids': {
                    'num06_highPressureFoamingMachine': {
                        'human_working_areas_ids': [130],
                        'robot_parking_areas_ids': [131],
                        'gantry_parking_areas_ids': [131],
                    },
                },
                'num_workstations': 1,
                'material_placement_cfg': {
                    'num06_highPressureFoamingMachine': {
                        'position': torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                        'orientation': torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
                    },
                },
            },
            'target_joints_position': [None],
        },
        'num08_workbench': {
            'state': ['free', 'free'],
            'processing_time_step': [0, 0],
            'ongoing_task_record_index': [None, None],
            'key_variables': {
                'type_name': 'num08_workbench',
                'working_area_ids': {
                    'num08_workbench_station_00': {
                        'human_working_areas_ids': [45],
                        'robot_parking_areas_ids': [40],
                        'gantry_parking_areas_ids': [47],
                    },
                    'num08_workbench_station_01': {
                        'human_working_areas_ids': [49],
                        'robot_parking_areas_ids': [41],
                        'gantry_parking_areas_ids': [47],
                    },
                },
                'num_workstations': 2,
                'material_placement_cfg': {
                    'num08_workbench_station_00': {
                        'position': torch.tensor([28.442920684814453, -2.0866899490356445, 0.5341200232505798], dtype=torch.float32),
                        'orientation': torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
                    },
                    'num08_workbench_station_01': {
                        'position': torch.tensor([25.166990280151367, -3.0504701137542725, 0.5341200232505798], dtype=torch.float32),
                        'orientation': torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
                    },
                },
            },
        },
        'num07_gantry_group': {
            'state': ['working_logistic_for_pipe_cutting', 'free', 'invalid', 'invalid'],
            'ongoing_task_record_index': [0, None, None, None],
            'key_variables': {
                'type_name': 'num07_gantry_group',
                'working_area_ids': {
                    'num07_gantry_group_station_00': {
                        'human_working_areas_ids': [],
                        'robot_parking_areas_ids': [],
                        'gantry_parking_areas_ids': [],
                    },
                    'num07_gantry_group_station_01': {
                        'human_working_areas_ids': [],
                        'robot_parking_areas_ids': [],
                        'gantry_parking_areas_ids': [],
                    },
                    'num07_gantry_group_station_02': {
                        'human_working_areas_ids': [],
                        'robot_parking_areas_ids': [],
                        'gantry_parking_areas_ids': [],
                    },
                    'num07_gantry_group_station_03': {
                        'human_working_areas_ids': [],
                        'robot_parking_areas_ids': [],
                        'gantry_parking_areas_ids': [],
                    },
                },
                'num_workstations': 4,
                'material_placement_cfg': {},
            },
            'target_area_id': [None, None, None, None],
            'target_area_xy': [None, None, None, None],
            'target_joints_position': [None, None, None, None],
        },
    },
    'material': {
        'num_00_ProductWaterPipe': {
            'key_variables': {
                'type_name': 'ProductWaterPipe',
                'type_id': '00',
                'idx': 0,
            },
            'finished_task': 'none',
            'ongoing_task_record_index': 0,
            'submaterials': {
                'product_00_pipe': {
                    'storage_name': 'BlackStorage_00',
                },
                'product_00_flange': {
                    'storage_name': 'GroundStorage_01',
                },
                'product_00_elbow': {
                    'storage_name': 'GroundStorage_00',
                },
                'product_00_semi': {
                    'storage_name': 'disappear',
                },
                'product_00_maded': {
                    'storage_name': 'disappear',
                },
            },
        },
        'num_01_ProductWaterPipe': {
            'key_variables': {
                'type_name': 'ProductWaterPipe',
                'type_id': '00',
                'idx': 1,
            },
            'finished_task': 'none',
            'ongoing_task_record_index': None,
            'submaterials': {
                'product_00_pipe': {
                    'storage_name': 'BlackStorage_00',
                },
                'product_00_flange': {
                    'storage_name': 'GroundStorage_01',
                },
                'product_00_elbow': {
                    'storage_name': 'GroundStorage_00',
                },
                'product_00_semi': {
                    'storage_name': 'disappear',
                },
                'product_00_maded': {
                    'storage_name': 'disappear',
                },
            },
        },
        'num_02_ProductWaterPipe': {
            'key_variables': {
                'type_name': 'ProductWaterPipe',
                'type_id': '00',
                'idx': 2,
            },
            'finished_task': 'none',
            'ongoing_task_record_index': None,
            'submaterials': {
                'product_00_pipe': {
                    'storage_name': 'BlackStorage_00',
                },
                'product_00_flange': {
                    'storage_name': 'GroundStorage_01',
                },
                'product_00_elbow': {
                    'storage_name': 'GroundStorage_00',
                },
                'product_00_semi': {
                    'storage_name': 'disappear',
                },
                'product_00_maded': {
                    'storage_name': 'disappear',
                },
            },
        },
        'num_03_ProductWaterPipe': {
            'key_variables': {
                'type_name': 'ProductWaterPipe',
                'type_id': '00',
                'idx': 3,
            },
            'finished_task': 'none',
            'ongoing_task_record_index': None,
            'submaterials': {
                'product_00_pipe': {
                    'storage_name': 'BlackStorage_00',
                },
                'product_00_flange': {
                    'storage_name': 'GroundStorage_01',
                },
                'product_00_elbow': {
                    'storage_name': 'GroundStorage_00',
                },
                'product_00_semi': {
                    'storage_name': 'disappear',
                },
                'product_00_maded': {
                    'storage_name': 'disappear',
                },
            },
        },
        'num_04_ProductWaterPipe': {
            'key_variables': {
                'type_name': 'ProductWaterPipe',
                'type_id': '00',
                'idx': 4,
            },
            'finished_task': 'none',
            'ongoing_task_record_index': None,
            'submaterials': {
                'product_00_pipe': {
                    'storage_name': 'BlackStorage_00',
                },
                'product_00_flange': {
                    'storage_name': 'GroundStorage_01',
                },
                'product_00_elbow': {
                    'storage_name': 'GroundStorage_00',
                },
                'product_00_semi': {
                    'storage_name': 'disappear',
                },
                'product_00_maded': {
                    'storage_name': 'disappear',
                },
            },
        },
    },
    'human': {
        'num_00_NormalHuman': {
            'key_variables': {
                'type_name': 'NormalHuman',
                'idx': 0,
            },
            'state': 'working_logistic_for_pipe_cutting',
            'ongoing_task_record_index': 0,
            'current_area_id': 202,
            'target_area_id': 39,
            'subtask_time_counter': 0,
            'generated_route': [
                {
                    'x': 0.48584050967927084,
                    'y': 2.3771750090176997,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 0.750773677014557,
                    'y': 2.3523325785107954,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 1.0157068443498574,
                    'y': 2.327490148003891,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 1.2806400116851435,
                    'y': 2.3026477174969866,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 1.5455731790204297,
                    'y': 2.2778052869900822,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 1.8105063463557087,
                    'y': 2.252962856483178,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 2.075439513691009,
                    'y': 2.2281204259762735,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 2.3403726810262953,
                    'y': 2.203277995469369,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 2.6053058483615814,
                    'y': 2.1784355649624647,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 2.8702390156968676,
                    'y': 2.1535931344555603,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 3.135172183032161,
                    'y': 2.1287507039486577,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 3.400105350367447,
                    'y': 2.1039082734417534,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 3.665038517702733,
                    'y': 2.079065842934849,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 3.9299716850380264,
                    'y': 2.0542234124279446,
                    'orientation': torch.tensor([0.7393707036972046, 0.0, 0.0, 0.6732842922210693], dtype=torch.float32),
                },
                {
                    'x': 4.174007717516652,
                    'y': 1.982744226611139,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 4.418043749995277,
                    'y': 1.9112650407943335,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 4.66207978247391,
                    'y': 1.839785854977528,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 4.906115814952543,
                    'y': 1.7683066691607232,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 5.150151847431168,
                    'y': 1.6968274833439176,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 5.3941878799097935,
                    'y': 1.625348297527113,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 5.638223912388419,
                    'y': 1.5538691117103074,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 5.882259944867052,
                    'y': 1.4823899258935027,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 6.126295977345684,
                    'y': 1.4109107400766971,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 6.370332009824317,
                    'y': 1.3394315542598925,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 6.614368042302942,
                    'y': 1.267952368443087,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 6.858404074781568,
                    'y': 1.1964731826262822,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 7.1024401072602075,
                    'y': 1.1249939968094766,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 7.346476139738833,
                    'y': 1.053514810992672,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 7.5905121722174655,
                    'y': 0.9820356251758664,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 7.834548204696091,
                    'y': 0.9105564393590617,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 8.07858423717471,
                    'y': 0.8390772535422562,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 8.32262026965335,
                    'y': 0.7675980677254515,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 8.566656302131975,
                    'y': 0.6961188819086459,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 8.810692334610607,
                    'y': 0.6246396960918412,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 9.054728367089233,
                    'y': 0.5531605102750357,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 9.298764399567865,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.8003342151641846, 0.0, 0.0, 0.5995380878448486], dtype=torch.float32),
                },
                {
                    'x': 9.555174900674537,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 9.811585401781208,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 10.067995902887887,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 10.324406403994551,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 10.58081690510123,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 10.837227406207901,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 11.093637907314573,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 11.350048408421245,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 11.606458909527916,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 11.862869410634595,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 12.119279911741266,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 12.375690412847938,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 12.63210091395461,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 12.88851141506128,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 13.14492191616796,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 13.401332417274624,
                    'y': 0.481681324458231,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 13.63030804995013,
                    'y': 0.5924980487782445,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 13.859283682625644,
                    'y': 0.703314773098259,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 14.088259315301144,
                    'y': 0.8141314974182725,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 14.317234947976651,
                    'y': 0.924948221738287,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 14.546210580652158,
                    'y': 1.0357649460583005,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 14.775186213327657,
                    'y': 1.1465816703783132,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 15.004161846003171,
                    'y': 1.2573983946983276,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 15.233137478678678,
                    'y': 1.368215119018342,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 15.462113111354178,
                    'y': 1.4790318433383565,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 15.691088744029692,
                    'y': 1.58984856765837,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 15.920064376705199,
                    'y': 1.7006652919783836,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 16.1490400093807,
                    'y': 1.8114820162983971,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 16.378015642056205,
                    'y': 1.9222987406184107,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 16.606991274731712,
                    'y': 2.033115464938424,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 16.83596690740722,
                    'y': 2.1439321892584378,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 17.064942540082725,
                    'y': 2.254748913578453,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 17.293918172758232,
                    'y': 2.3655656378984666,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 17.52289380543374,
                    'y': 2.476382362218482,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 17.751869438109246,
                    'y': 2.5871990865384955,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 17.980845070784753,
                    'y': 2.698015810858509,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 18.20982070346026,
                    'y': 2.8088325351785226,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 18.43879633613576,
                    'y': 2.919649259498536,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 18.667771968811273,
                    'y': 3.0304659838185497,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 18.89674760148678,
                    'y': 3.1412827081385633,
                    'orientation': torch.tensor([0.5312051773071289, 0.0, 0.0, 0.8472318649291992], dtype=torch.float32),
                },
                {
                    'x': 19.154461569193117,
                    'y': 3.1576724085421954,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 19.41217553689946,
                    'y': 3.1740621089458276,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 19.669889504605806,
                    'y': 3.19045180934946,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 19.92760347231215,
                    'y': 3.206841509753092,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 20.1853174400185,
                    'y': 3.223231210156724,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 20.443031407724845,
                    'y': 3.2396209105603564,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 20.700745375431183,
                    'y': 3.2560106109639886,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 20.958459343137534,
                    'y': 3.2724003113676208,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 21.216173310843878,
                    'y': 3.288790011771253,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 21.473887278550215,
                    'y': 3.305179712174887,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 21.73160124625656,
                    'y': 3.321569412578519,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 21.989315213962904,
                    'y': 3.3379591129821495,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 22.247029181669248,
                    'y': 3.3543488133857817,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 22.5047431493756,
                    'y': 3.370738513789414,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 22.762457117081937,
                    'y': 3.387128214193046,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 23.02017108478828,
                    'y': 3.4035179145966783,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 23.27788505249463,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.6842929720878601, 0.0, 0.0, 0.7291939854621887], dtype=torch.float32),
                },
                {
                    'x': 23.531129991859242,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 23.784374931223855,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 24.037619870588472,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 24.290864809953085,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 24.5441097493177,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 24.79735468868231,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 25.05059962804693,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 25.30384456741154,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 25.55708950677616,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 25.81033444614077,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 26.06357938550539,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 26.316824324870005,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 26.57006926423462,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 26.82331420359923,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 27.07655914296385,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 27.32980408232846,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 27.58304902169308,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 27.836293961057695,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 28.08953890042231,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 28.342783839786925,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 28.59602877915154,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 28.84927371851615,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 29.10251865788077,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 29.355763597245385,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 29.609008536610002,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 29.86225347597462,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
                {
                    'x': 30.115498415339232,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.707099974155426, 0.0, 0.0, 0.707099974155426], dtype=torch.float32),
                },
            ],
            'route_index': 115,
            'route_length': 120,
            'detour_active': False,
            'detour_blocker_key': None,
            'detour_until_route_index': None,
            'yield_active': False,
            'yield_blocker_key': None,
        },
        'num_01_NormalHuman': {
            'key_variables': {
                'type_name': 'NormalHuman',
                'idx': 1,
            },
            'state': 'free',
            'ongoing_task_record_index': None,
            'current_area_id': 217,
            'target_area_id': None,
            'subtask_time_counter': 0,
            'generated_route': [],
            'route_index': 0,
            'route_length': 0,
            'detour_active': False,
            'detour_blocker_key': None,
            'detour_until_route_index': None,
            'yield_active': False,
            'yield_blocker_key': None,
        },
        'num_02_NormalHuman': {
            'key_variables': {
                'type_name': 'NormalHuman',
                'idx': 2,
            },
            'state': 'free',
            'ongoing_task_record_index': None,
            'current_area_id': 215,
            'target_area_id': None,
            'subtask_time_counter': 0,
            'generated_route': [],
            'route_index': 0,
            'route_length': 0,
            'detour_active': False,
            'detour_blocker_key': None,
            'detour_until_route_index': None,
            'yield_active': False,
            'yield_blocker_key': None,
        },
        'num_03_NormalHuman': {
            'key_variables': {
                'type_name': 'NormalHuman',
                'idx': 3,
            },
            'state': 'free',
            'ongoing_task_record_index': None,
            'current_area_id': 203,
            'target_area_id': None,
            'subtask_time_counter': 0,
            'generated_route': [],
            'route_index': 0,
            'route_length': 0,
            'detour_active': False,
            'detour_blocker_key': None,
            'detour_until_route_index': None,
            'yield_active': False,
            'yield_blocker_key': None,
        },
        'num_04_NormalHuman': {
            'key_variables': {
                'type_name': 'NormalHuman',
                'idx': 4,
            },
            'state': 'free',
            'ongoing_task_record_index': None,
            'current_area_id': 203,
            'target_area_id': None,
            'subtask_time_counter': 0,
            'generated_route': [],
            'route_index': 0,
            'route_length': 0,
            'detour_active': False,
            'detour_blocker_key': None,
            'detour_until_route_index': None,
            'yield_active': False,
            'yield_blocker_key': None,
        },
    },
    'robot': {
        'num_00_AGV': {
            'key_variables': {
                'type_name': 'AGV',
                'idx': 0,
            },
            'state': 'working_logistic_for_pipe_cutting',
            'ongoing_task_record_index': 0,
            'current_area_id': 272,
            'target_area_id': 38,
            'generated_route': [
                {
                    'x': -7.353241477016326,
                    'y': 5.5222591849571465,
                    'orientation': torch.tensor([0.9977226257324219, 0.0, 0.0, -0.06745040416717529], dtype=torch.float32),
                },
                {
                    'x': -7.092168966798631,
                    'y': 5.486797833174739,
                    'orientation': torch.tensor([0.9977226257324219, 0.0, 0.0, -0.06745040416717529], dtype=torch.float32),
                },
                {
                    'x': -6.831096456580916,
                    'y': 5.451336481392337,
                    'orientation': torch.tensor([0.9977226257324219, 0.0, 0.0, -0.06745040416717529], dtype=torch.float32),
                },
                {
                    'x': -6.5700239463632215,
                    'y': 5.415875129609928,
                    'orientation': torch.tensor([0.9977226257324219, 0.0, 0.0, -0.06745040416717529], dtype=torch.float32),
                },
                {
                    'x': -6.308951436145506,
                    'y': 5.380413777827524,
                    'orientation': torch.tensor([0.9977226257324219, 0.0, 0.0, -0.06745040416717529], dtype=torch.float32),
                },
                {
                    'x': -6.047878925927812,
                    'y': 5.344952426045118,
                    'orientation': torch.tensor([0.9977226257324219, 0.0, 0.0, -0.06745040416717529], dtype=torch.float32),
                },
                {
                    'x': -5.798251771411259,
                    'y': 5.4256149396658095,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -5.548624616894706,
                    'y': 5.506277453286499,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -5.298997462378168,
                    'y': 5.58693996690719,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -5.049370307861615,
                    'y': 5.6676024805278775,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -4.799743153345062,
                    'y': 5.748264994148569,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -4.55011599882851,
                    'y': 5.82892750776926,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -4.300488844311964,
                    'y': 5.909590021389949,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -4.050861689795418,
                    'y': 5.99025253501064,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -3.8012345352788657,
                    'y': 6.0709150486313295,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -3.551607380762313,
                    'y': 6.151577562252021,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -3.3019802262457603,
                    'y': 6.232240075872712,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -3.0523530717292218,
                    'y': 6.312902589493401,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -2.802725917212669,
                    'y': 6.393565103114092,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -2.5530987626961164,
                    'y': 6.474227616734781,
                    'orientation': torch.tensor([0.9878145456314087, 0.0, 0.0, 0.15563544631004333], dtype=torch.float32),
                },
                {
                    'x': -2.299853823331503,
                    'y': 6.460683350429003,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': -2.046608883966897,
                    'y': 6.447139084123222,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': -1.7933639446022696,
                    'y': 6.433594817817442,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': -1.5401190052376634,
                    'y': 6.420050551511663,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': -1.286874065873036,
                    'y': 6.406506285205884,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': -1.0336291265084299,
                    'y': 6.392962018900105,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': -0.7803841871438166,
                    'y': 6.379417752594325,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': -0.5271392477791892,
                    'y': 6.365873486288546,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': -0.27389430841458307,
                    'y': 6.3523292199827655,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': -0.020649369049955624,
                    'y': 6.338784953676985,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': 0.2325955703146505,
                    'y': 6.325240687371208,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': 0.48584050967927084,
                    'y': 6.311696421065427,
                    'orientation': torch.tensor([0.9996431469917297, 0.0, 0.0, -0.026712803170084953], dtype=torch.float32),
                },
                {
                    'x': 0.750773677014557,
                    'y': 6.292699268324855,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 1.0157068443498574,
                    'y': 6.27370211558428,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 1.2806400116851435,
                    'y': 6.254704962843707,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 1.5455731790204297,
                    'y': 6.235707810103131,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 1.8105063463557087,
                    'y': 6.216710657362558,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 2.075439513691009,
                    'y': 6.197713504621985,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 2.3403726810262953,
                    'y': 6.178716351881411,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 2.6053058483615814,
                    'y': 6.159719199140838,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 2.8702390156968676,
                    'y': 6.140722046400263,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 3.135172183032161,
                    'y': 6.121724893659689,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 3.400105350367447,
                    'y': 6.102727740919114,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 3.665038517702733,
                    'y': 6.0837305881785415,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 3.9299716850380264,
                    'y': 6.064733435437967,
                    'orientation': torch.tensor([0.9993595480918884, 0.0, 0.0, -0.03578382730484009], dtype=torch.float32),
                },
                {
                    'x': 4.1858823606064774,
                    'y': 6.0735098978151925,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 4.4417930361749285,
                    'y': 6.082286360192418,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 4.6977037117433795,
                    'y': 6.09106282256964,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 4.953614387311838,
                    'y': 6.0998392849468654,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 5.209525062880289,
                    'y': 6.108615747324089,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 5.46543573844874,
                    'y': 6.117392209701315,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 5.721346414017191,
                    'y': 6.126168672078538,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 5.977257089585649,
                    'y': 6.134945134455762,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 6.2331677651541,
                    'y': 6.143721596832986,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 6.489078440722551,
                    'y': 6.152498059210211,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 6.744989116291009,
                    'y': 6.161274521587437,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 7.00089979185946,
                    'y': 6.170050983964659,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 7.256810467427911,
                    'y': 6.178827446341884,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 7.512721142996369,
                    'y': 6.187603908719108,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 7.76863181856482,
                    'y': 6.196380371096334,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 8.024542494133279,
                    'y': 6.205156833473557,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 8.28045316970173,
                    'y': 6.213933295850781,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 8.53636384527018,
                    'y': 6.222709758228005,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 8.792274520838639,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([0.9998530745506287, 0.0, 0.0, 0.01713995449244976], dtype=torch.float32),
                },
                {
                    'x': 9.051850583687369,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 9.311426646536106,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 9.571002709384835,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 9.830578772233565,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 10.090154835082295,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 10.349730897931025,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 10.609306960779762,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 10.868883023628491,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 11.128459086477221,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 11.388035149325951,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 11.64761121217468,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 11.90718727502341,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 12.16676333787214,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 12.42633940072087,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 12.6859154635696,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 12.94549152641833,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 13.20506758926706,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 13.46464365211579,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 13.724219714964526,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 13.983795777813256,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 14.243371840661986,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 14.502947903510716,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 14.762523966359446,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 15.022100029208175,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 15.281676092056905,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 15.541252154905635,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 15.800828217754365,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 16.060404280603095,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 16.319980343451824,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 16.579556406300554,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 16.839132469149284,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 17.09870853199802,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 17.35828459484675,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 17.61786065769548,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 17.87743672054421,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 18.13701278339294,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 18.396588846241677,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 18.656164909090407,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 18.915740971939137,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 19.175317034787867,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 19.434893097636596,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 19.694469160485326,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 19.954045223334056,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 20.213621286182786,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 20.473197349031516,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 20.732773411880245,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 20.992349474728975,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 21.251925537577705,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 21.511501600426435,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 21.771077663275165,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 22.030653726123894,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 22.290229788972624,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 22.54980585182136,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 22.80938191467009,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 23.06895797751882,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 23.32853404036755,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 23.58811010321628,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 23.84768616606501,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 24.10726222891374,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 24.36683829176247,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 24.6264143546112,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 24.88599041745993,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 25.14556648030866,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 25.405142543157393,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 25.664718606006122,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 25.924294668854852,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 26.18387073170359,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 26.44344679455232,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 26.70302285740105,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 26.962598920249786,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 27.222174983098515,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 27.481751045947245,
                    'y': 6.23148622060523,
                    'orientation': torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                },
                {
                    'x': 27.705067037932405,
                    'y': 6.103687193077736,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 27.92838302991757,
                    'y': 5.975888165550238,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 28.151699021902726,
                    'y': 5.8480891380227416,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 28.375015013887886,
                    'y': 5.720290110495245,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 28.598331005873046,
                    'y': 5.592491082967749,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 28.821646997858206,
                    'y': 5.464692055440253,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 29.044962989843366,
                    'y': 5.336893027912756,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 29.268278981828526,
                    'y': 5.209094000385258,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 29.491594973813687,
                    'y': 5.081294972857764,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 29.71491096579885,
                    'y': 4.953495945330266,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 29.938226957784007,
                    'y': 4.8256969178027695,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 30.16154294976917,
                    'y': 4.697897890275275,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 30.384858941754327,
                    'y': 4.570098862747777,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 30.60817493373949,
                    'y': 4.4422998352202825,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 30.83149092572465,
                    'y': 4.314500807692784,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 31.054806917709808,
                    'y': 4.18670178016529,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 31.278122909694968,
                    'y': 4.058902752637792,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 31.50143890168013,
                    'y': 3.9311037251102956,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 31.72475489366529,
                    'y': 3.803304697582801,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 31.94807088565045,
                    'y': 3.675505670055303,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 32.17138687763561,
                    'y': 3.5477066425278085,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
                {
                    'x': 32.39470286962077,
                    'y': 3.4199076150003105,
                    'orientation': torch.tensor([0.9664173722267151, 0.0, 0.0, -0.2569775879383087], dtype=torch.float32),
                },
            ],
            'route_index': 115,
            'route_length': 158,
            'detour_active': False,
            'detour_blocker_key': None,
            'detour_until_route_index': None,
            'yield_active': False,
            'yield_blocker_key': None,
        },
        'num_01_AGV': {
            'key_variables': {
                'type_name': 'AGV',
                'idx': 1,
            },
            'state': 'free',
            'ongoing_task_record_index': None,
            'current_area_id': 255,
            'target_area_id': None,
            'generated_route': [],
            'route_index': 0,
            'route_length': 0,
            'detour_active': False,
            'detour_blocker_key': None,
            'detour_until_route_index': None,
            'yield_active': False,
            'yield_blocker_key': None,
        },
    },
    'storage': {
        'BlackStorage_00': {
            'key_variables': {
                'type_name': 'Black Storage 00 nearby num08_workbench robot_parking_areas_ids left bottom corner',
                'idx': 0,
                'class_name': 'BlackStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [39],
                    'robot_parking_areas_ids': [38],
                    'gantry_parking_areas_ids': [39],
                },
                'placement_type': 'parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel',
                'placement_cfg': {
                    'capacity': 6,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [30.76917266845703, 1.8350062370300293, 0.5622950792312622],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [30.76917266845703, 1.4010863304138184, 0.5622950792312622],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [30.76917266845703, 0.9660962820053101, 0.5622950792312622],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [30.76917266845703, 1.609276294708252, 0.9362651109695435],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_04',
                            'position': torch.tensor([
                                [30.76917266845703, 1.195926308631897, 0.9368050694465637],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_05',
                            'position': torch.tensor([
                                [30.76917266845703, 1.3956762552261353, 1.2912850379943848],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'partial',
            'num_material': 5,
            'material_type': 'product_00_pipe',
            'material_idx_list': [0, 1, 2, 3, 4],
        },
        'BlackStorage_01': {
            'key_variables': {
                'type_name': 'Black Storage 01 nearby num07_gantry_group machine right bottom corner',
                'idx': 1,
                'class_name': 'BlackStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [42],
                    'robot_parking_areas_ids': [41],
                    'gantry_parking_areas_ids': [42],
                },
                'placement_type': 'parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel',
                'placement_cfg': {
                    'capacity': 6,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [22.66964340209961, 1.7952226400375366, 0.5622950792312622],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [22.66964340209961, 1.3613027334213257, 0.5622950792312622],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [22.66964340209961, 0.9263126850128174, 0.5622950792312622],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [22.66964340209961, 1.5694926977157593, 0.9362651109695435],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_04',
                            'position': torch.tensor([
                                [22.66964340209961, 1.1561427116394043, 0.9368050694465637],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_05',
                            'position': torch.tensor([
                                [22.66964340209961, 1.3558926582336426, 1.2912850379943848],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'BlackStorage_02': {
            'key_variables': {
                'type_name': 'Black Storage 02 nearby num07_gantry_group machine right side corner',
                'idx': 2,
                'class_name': 'BlackStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [53],
                    'robot_parking_areas_ids': [54],
                    'gantry_parking_areas_ids': [53],
                },
                'placement_type': 'vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis',
                'placement_cfg': {
                    'capacity': 6,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [17.190616607666016, -1.9436981678009033, 0.38559406995773315],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [17.62453842163086, -1.9436981678009033, 0.38559406995773315],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [18.059528350830078, -1.9436981678009033, 0.38559406995773315],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [17.41634750366211, -1.9436981678009033, 0.7595640420913696],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_04',
                            'position': torch.tensor([
                                [17.829696655273438, -1.9436981678009033, 0.7601040601730347],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_05',
                            'position': torch.tensor([
                                [17.629947662353516, -1.9436981678009033, 1.1145840883255005],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'BlackStorage_03': {
            'key_variables': {
                'type_name': 'Black Storage 03 nearby num04_groovingMachineLarge machine left side corner',
                'idx': 3,
                'class_name': 'BlackStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [221],
                    'robot_parking_areas_ids': [219],
                    'gantry_parking_areas_ids': [221],
                },
                'placement_type': 'vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis',
                'placement_cfg': {
                    'capacity': 6,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-5.131072044372559, 16.237483978271484, 0.3855948746204376],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-4.697152137756348, 16.237483978271484, 0.3855948746204376],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-4.262162208557129, 16.237483978271484, 0.3855948746204376],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-4.905342102050781, 16.237483978271484, 0.7595648765563965],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_04',
                            'position': torch.tensor([
                                [-4.491991996765137, 16.237483978271484, 0.7601048946380615],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_05',
                            'position': torch.tensor([
                                [-4.691741943359375, 16.237483978271484, 1.1145849227905273],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'BlackStorage_04': {
            'key_variables': {
                'type_name': 'Black Storage 04 nearby num04_groovingMachineLarge machine right side corner',
                'idx': 4,
                'class_name': 'BlackStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [140],
                    'robot_parking_areas_ids': [139],
                    'gantry_parking_areas_ids': [140],
                },
                'placement_type': 'vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis',
                'placement_cfg': {
                    'capacity': 6,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-13.793753623962402, 16.237483978271484, 0.3855948746204376],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-13.359833717346191, 16.237483978271484, 0.3855948746204376],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-12.924843788146973, 16.237483978271484, 0.3855948746204376],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-13.568023681640625, 16.237483978271484, 0.7595648765563965],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_04',
                            'position': torch.tensor([
                                [-13.15467357635498, 16.237483978271484, 0.7601048946380615],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_05',
                            'position': torch.tensor([
                                [-13.354423522949219, 16.237483978271484, 1.1145849227905273],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'YellowStorage_00': {
            'key_variables': {
                'type_name': 'Yellow Storage 00, located opposite the num04_groovingMachineLarge machine',
                'idx': 0,
                'class_name': 'YellowStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [116],
                    'robot_parking_areas_ids': [272],
                    'gantry_parking_areas_ids': [116],
                },
                'placement_type': 'vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis',
                'placement_cfg': {
                    'capacity': 4,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-5.778212547302246, -1.9107236862182617, 0.562299907207489],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-5.344292640686035, -1.9107236862182617, 0.562299907207489],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-5.746922492980957, -1.9107236862182617, 0.9362699389457703],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-5.3335723876953125, -1.9107236862182617, 0.9368098974227905],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'YellowStorage_01': {
            'key_variables': {
                'type_name': 'Yellow Storage 01 adjacent to Yellow Storage_00 on the right side, located opposite the num04_groovingMachineLarge machine',
                'idx': 1,
                'class_name': 'YellowStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [116],
                    'robot_parking_areas_ids': [272],
                    'gantry_parking_areas_ids': [116],
                },
                'placement_type': 'vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis',
                'placement_cfg': {
                    'capacity': 4,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-7.439878463745117, -1.9107236862182617, 0.562299907207489],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-7.005958557128906, -1.9107236862182617, 0.562299907207489],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-7.408588409423828, -1.9107236862182617, 0.9362699389457703],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-6.995238304138184, -1.9107236862182617, 0.9368098974227905],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'YellowStorage_02': {
            'key_variables': {
                'type_name': 'Yellow Storage 02 adjacent to Yellow Storage_01 on the right side, located opposite the num04_groovingMachineLarge machine',
                'idx': 2,
                'class_name': 'YellowStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [117],
                    'robot_parking_areas_ids': [272],
                    'gantry_parking_areas_ids': [117],
                },
                'placement_type': 'vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis',
                'placement_cfg': {
                    'capacity': 4,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-9.206778526306152, -1.9107236862182617, 0.562299907207489],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-8.772858619689941, -1.9107236862182617, 0.562299907207489],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-9.17548942565918, -1.9107236862182617, 0.9362699389457703],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-8.762139320373535, -1.9107236862182617, 0.9368098974227905],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'YellowStorage_03': {
            'key_variables': {
                'type_name': 'Yellow Storage 03 adjacent to Yellow Storage_02 on the right side, located opposite the num04_groovingMachineLarge machine',
                'idx': 3,
                'class_name': 'YellowStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [118],
                    'robot_parking_areas_ids': [273],
                    'gantry_parking_areas_ids': [118],
                },
                'placement_type': 'vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis',
                'placement_cfg': {
                    'capacity': 4,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-10.841633796691895, -1.9107236862182617, 0.562299907207489],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-10.407713890075684, -1.9107236862182617, 0.562299907207489],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-10.810344696044922, -1.9107236862182617, 0.9362699389457703],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-10.396994590759277, -1.9107236862182617, 0.9368098974227905],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'YellowStorage_04': {
            'key_variables': {
                'type_name': 'Yellow Storage 04 adjacent to Yellow Storage_03 on the right side, located opposite the num04_groovingMachineLarge machine',
                'idx': 4,
                'class_name': 'YellowStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [118],
                    'robot_parking_areas_ids': [273],
                    'gantry_parking_areas_ids': [118],
                },
                'placement_type': 'vertical, the storage is vertical to the material, the material should be rotated 90 degrees around the z axis',
                'placement_cfg': {
                    'capacity': 4,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-12.568239212036133, -1.9107236862182617, 0.562299907207489],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-12.134319305419922, -1.9107236862182617, 0.562299907207489],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-12.53695011138916, -1.9107236862182617, 0.9362699389457703],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-12.123600006103516, -1.9107236862182617, 0.9368098974227905],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [0.7071067690849304, 0.0, 0.0, 0.7071067690849304],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'YellowStorage_05': {
            'key_variables': {
                'type_name': 'Yellow Storage 05 adjacent to Black Storage_05 on the right side, located opposite the semiFinishedArea_02',
                'idx': 5,
                'class_name': 'YellowStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [144],
                    'robot_parking_areas_ids': [299],
                    'gantry_parking_areas_ids': [144],
                },
                'placement_type': 'parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel',
                'placement_cfg': {
                    'capacity': 4,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-21.318622589111328, 15.086153030395508, 0.5623005628585815],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-21.318622589111328, 14.652233123779297, 0.5623005628585815],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-21.318622589111328, 15.054863929748535, 0.9362705945968628],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-21.318622589111328, 14.64151382446289, 0.9368106126785278],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'YellowStorage_06': {
            'key_variables': {
                'type_name': 'Yellow Storage 06 adjacent to Yellow Storage_05, located opposite the semiFinishedArea_02',
                'idx': 6,
                'class_name': 'YellowStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [144],
                    'robot_parking_areas_ids': [299],
                    'gantry_parking_areas_ids': [144],
                },
                'placement_type': 'parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel',
                'placement_cfg': {
                    'capacity': 4,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-21.318622589111328, 16.863752365112305, 0.5623006224632263],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-21.318622589111328, 16.429832458496094, 0.5623006224632263],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-21.318622589111328, 16.832462310791016, 0.9362706542015076],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-21.318622589111328, 16.419113159179688, 0.9368106126785278],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'YellowStorage_07': {
            'key_variables': {
                'type_name': 'Yellow Storage 07 adjacent to Yellow Storage_06, located opposite the semiFinishedArea_02',
                'idx': 7,
                'class_name': 'YellowStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [144],
                    'robot_parking_areas_ids': [299],
                    'gantry_parking_areas_ids': [144],
                },
                'placement_type': 'parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel',
                'placement_cfg': {
                    'capacity': 4,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-21.318622589111328, 18.696853637695312, 0.5623006820678711],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-21.318622589111328, 18.2629337310791, 0.5623006820678711],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-21.318622589111328, 18.665563583374023, 0.9362707138061523],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-21.318622589111328, 18.252214431762695, 0.9368107318878174],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'YellowStorage_08': {
            'key_variables': {
                'type_name': 'Yellow Storage 08 adjacent to Yellow Storage_07, located opposite the semiFinishedArea_02',
                'idx': 8,
                'class_name': 'YellowStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [148],
                    'robot_parking_areas_ids': [299],
                    'gantry_parking_areas_ids': [148],
                },
                'placement_type': 'parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel',
                'placement_cfg': {
                    'capacity': 4,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-27.998458862304688, 15.086153030395508, 0.5623005628585815],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-27.998458862304688, 14.652233123779297, 0.5623005628585815],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-27.998458862304688, 15.054863929748535, 0.9362705945968628],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-27.998458862304688, 14.64151382446289, 0.9368106126785278],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'YellowStorage_09': {
            'key_variables': {
                'type_name': 'Yellow Storage 09 adjacent to Yellow Storage_08 located opposite the semiFinishedArea_02',
                'idx': 9,
                'class_name': 'YellowStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [148],
                    'robot_parking_areas_ids': [299],
                    'gantry_parking_areas_ids': [148],
                },
                'placement_type': 'parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel',
                'placement_cfg': {
                    'capacity': 4,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-27.998458862304688, 16.863752365112305, 0.5623006224632263],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-27.998458862304688, 16.429832458496094, 0.5623006224632263],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-27.998458862304688, 16.832462310791016, 0.9362706542015076],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-27.998458862304688, 16.419113159179688, 0.9368106126785278],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'YellowStorage_10': {
            'key_variables': {
                'type_name': 'Yellow Storage 10 adjacent to Yellow Storage_09, located opposite the semiFinishedArea_02',
                'idx': 10,
                'class_name': 'YellowStorage',
                'capacity': 6,
                'working_area_ids': {
                    'human_working_areas_ids': [148],
                    'robot_parking_areas_ids': [299],
                    'gantry_parking_areas_ids': [148],
                },
                'placement_type': 'parallel, the storage is parallel to the material by default, so their local coordinate xyz is parallel',
                'placement_cfg': {
                    'capacity': 4,
                    'data_type': 'relative',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00',
                            'position': torch.tensor([
                                [-27.998458862304688, 18.696853637695312, 0.5623006820678711],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01',
                            'position': torch.tensor([
                                [-27.998458862304688, 18.2629337310791, 0.5623006820678711],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02',
                            'position': torch.tensor([
                                [-27.998458862304688, 18.665563583374023, 0.9362707138061523],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_03',
                            'position': torch.tensor([
                                [-27.998458862304688, 18.252214431762695, 0.9368107318878174],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'empty',
            'num_material': 0,
            'material_type': None,
            'material_idx_list': [],
        },
        'GroundStorage_00': {
            'key_variables': {
                'type_name': 'Ground Storage 00 adjacent to num08_workbench machine',
                'idx': 0,
                'class_name': 'GroundStorage',
                'capacity': 100,
                'working_area_ids': {
                    'human_working_areas_ids': [45],
                    'robot_parking_areas_ids': [53],
                    'gantry_parking_areas_ids': [52],
                },
                'placement_type': 'grid: The storage consists of a grid of placements arranged within a rectangular area.',
                'placement_cfg': {
                    'num_columns': 6,
                    'num_rows': 3,
                    'capacity': 18,
                    'data_type': 'absolute',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00_00',
                            'position': torch.tensor([
                                [33.72357177734375, -4.7042999267578125, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_00_01',
                            'position': torch.tensor([
                                [33.11420822143555, -3.881119966506958, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_00_02',
                            'position': torch.tensor([
                                [32.50484848022461, -3.0579400062561035, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_00_03',
                            'position': torch.tensor([
                                [31.895490646362305, -2.234760046005249, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_00_04',
                            'position': torch.tensor([
                                [31.286130905151367, -1.411579966545105, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_00_05',
                            'position': torch.tensor([
                                [30.676769256591797, -0.5884000062942505, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_00',
                            'position': torch.tensor([
                                [33.72357177734375, -3.3202600479125977, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_01',
                            'position': torch.tensor([
                                [33.11420822143555, -2.497080087661743, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_02',
                            'position': torch.tensor([
                                [32.50484848022461, -1.6739000082015991, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_03',
                            'position': torch.tensor([
                                [31.895490646362305, -0.8507199883460999, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_04',
                            'position': torch.tensor([
                                [31.286130905151367, -0.027540000155568123, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_05',
                            'position': torch.tensor([
                                [30.676769256591797, 0.7956399917602539, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_00',
                            'position': torch.tensor([
                                [33.72357177734375, -1.9362200498580933, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_01',
                            'position': torch.tensor([
                                [33.11420822143555, -1.1130399703979492, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_02',
                            'position': torch.tensor([
                                [32.50484848022461, -0.2898600101470947, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_03',
                            'position': torch.tensor([
                                [31.895490646362305, 0.5333200097084045, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_04',
                            'position': torch.tensor([
                                [31.286130905151367, 1.3565000295639038, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_05',
                            'position': torch.tensor([
                                [30.676769256591797, 2.179680109024048, 0.30000001192092896],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'partial',
            'num_material': 5,
            'material_type': 'product_00_elbow',
            'material_idx_list': [0, 1, 2, 3, 4],
        },
        'GroundStorage_01': {
            'key_variables': {
                'type_name': 'Ground Storage 01 adjacent to num08_workbench machine',
                'idx': 1,
                'class_name': 'GroundStorage',
                'capacity': 100,
                'working_area_ids': {
                    'human_working_areas_ids': [49],
                    'robot_parking_areas_ids': [54],
                    'gantry_parking_areas_ids': [51],
                },
                'placement_type': 'grid: The storage consists of a grid of placements arranged within a rectangular area.',
                'placement_cfg': {
                    'num_columns': 6,
                    'num_rows': 3,
                    'capacity': 18,
                    'data_type': 'absolute',
                    'pose_list': [
                        {
                            'placement_name': 'storage_pose_00_00',
                            'position': torch.tensor([
                                [19.209720611572266, -4.331299781799316, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_00_01',
                            'position': torch.tensor([
                                [19.881149291992188, -3.4560201168060303, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_00_02',
                            'position': torch.tensor([
                                [20.552579879760742, -2.580739974975586, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_00_03',
                            'position': torch.tensor([
                                [21.224010467529297, -1.7054599523544312, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_00_04',
                            'position': torch.tensor([
                                [21.89543914794922, -0.8301799893379211, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_00_05',
                            'position': torch.tensor([
                                [22.566869735717773, 0.045099999755620956, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_00',
                            'position': torch.tensor([
                                [19.184980392456055, -2.8376100063323975, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_01',
                            'position': torch.tensor([
                                [19.856409072875977, -1.9623299837112427, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_02',
                            'position': torch.tensor([
                                [20.52783966064453, -1.087049961090088, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_03',
                            'position': torch.tensor([
                                [21.199270248413086, -0.21176999807357788, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_04',
                            'position': torch.tensor([
                                [21.87070083618164, 0.6635100245475769, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_01_05',
                            'position': torch.tensor([
                                [22.542129516601562, 1.538789987564087, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_00',
                            'position': torch.tensor([
                                [19.160240173339844, -1.3439199924468994, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_01',
                            'position': torch.tensor([
                                [19.8316707611084, -0.468639999628067, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_02',
                            'position': torch.tensor([
                                [20.50309944152832, 0.4066399931907654, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_03',
                            'position': torch.tensor([
                                [21.174530029296875, 1.2819199562072754, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_04',
                            'position': torch.tensor([
                                [21.84596061706543, 2.1572000980377197, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                        {
                            'placement_name': 'storage_pose_02_05',
                            'position': torch.tensor([
                                [22.51738929748535, 3.032480001449585, 0.21770000457763672],
                            ], dtype=torch.float32),
                            'orientation': torch.tensor([
                                [1.0, 0.0, 0.0, 0.0],
                            ], dtype=torch.float32),
                        },
                    ],
                },
            },
            'state': 'partial',
            'num_material': 5,
            'material_type': 'product_00_flange',
            'material_idx_list': [0, 1, 2, 3, 4],
        },
    },
    'rigid_prims': {
        'num_00_product_00_pipe': {
            'position': torch.tensor([
                [30.76917266845703, 1.8350062370300293, 0.5622950792312622],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_00_product_00_flange': {
            'position': torch.tensor([
                [19.209720611572266, -4.331299781799316, 0.21770000457763672],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_00_product_00_elbow': {
            'position': torch.tensor([
                [33.72357177734375, -4.7042999267578125, 0.30000001192092896],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_00_product_00_semi': {
            'position': torch.tensor([
                [36.842899322509766, 25.53795051574707, -100.0],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_00_product_00_maded': {
            'position': torch.tensor([
                [29.443740844726562, 25.79178237915039, -100.0],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_01_product_00_pipe': {
            'position': torch.tensor([
                [30.76917266845703, 1.4010863304138184, 0.5622950792312622],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_01_product_00_flange': {
            'position': torch.tensor([
                [19.881149291992188, -3.4560201168060303, 0.21770000457763672],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_01_product_00_elbow': {
            'position': torch.tensor([
                [33.11420822143555, -3.881119966506958, 0.30000001192092896],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_01_product_00_semi': {
            'position': torch.tensor([
                [36.84450149536133, 26.705078125, -100.0],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_01_product_00_maded': {
            'position': torch.tensor([
                [29.443740844726562, 26.79300880432129, -100.0],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_02_product_00_pipe': {
            'position': torch.tensor([
                [30.76917266845703, 0.9660962820053101, 0.5622950792312622],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_02_product_00_flange': {
            'position': torch.tensor([
                [20.552579879760742, -2.580739974975586, 0.21770000457763672],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_02_product_00_elbow': {
            'position': torch.tensor([
                [32.50484848022461, -3.0579400062561035, 0.30000001192092896],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_02_product_00_semi': {
            'position': torch.tensor([
                [36.84450149536133, 27.810775756835938, -100.0],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_02_product_00_maded': {
            'position': torch.tensor([
                [29.443740844726562, 27.964052200317383, -100.0],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_03_product_00_pipe': {
            'position': torch.tensor([
                [30.76917266845703, 1.609276294708252, 0.9362651109695435],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_03_product_00_flange': {
            'position': torch.tensor([
                [21.224010467529297, -1.7054599523544312, 0.21770000457763672],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_03_product_00_elbow': {
            'position': torch.tensor([
                [31.895490646362305, -2.234760046005249, 0.30000001192092896],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_03_product_00_semi': {
            'position': torch.tensor([
                [36.84450149536133, 29.004966735839844, -100.0],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_03_product_00_maded': {
            'position': torch.tensor([
                [29.443740844726562, 29.024538040161133, -100.0],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_04_product_00_pipe': {
            'position': torch.tensor([
                [30.76917266845703, 1.195926308631897, 0.9368050694465637],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_04_product_00_flange': {
            'position': torch.tensor([
                [21.89543914794922, -0.8301799893379211, 0.21770000457763672],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_04_product_00_elbow': {
            'position': torch.tensor([
                [31.286130905151367, -1.411579966545105, 0.30000001192092896],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_04_product_00_semi': {
            'position': torch.tensor([
                [36.84450149536133, 29.97551918029785, -100.0],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_04_product_00_maded': {
            'position': torch.tensor([
                [29.443740844726562, 30.120006561279297, -100.0],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_00_NormalHuman': {
            'position': torch.tensor([
                [28.849273681640625, 3.419907569885254, 0.13394999504089355],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [0.707099974155426, 0.0, 0.0, 0.707099974155426],
            ], dtype=torch.float32),
        },
        'num_01_NormalHuman': {
            'position': torch.tensor([
                [-2.5795578956604004, 8.669281959533691, 0.13394999504089355],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [0.25746282935142517, 0.0, 0.0, 0.9662781953811646],
            ], dtype=torch.float32),
        },
        'num_02_NormalHuman': {
            'position': torch.tensor([
                [-2.272568702697754, 4.325856685638428, 0.13394999504089355],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [0.9996333122253418, 0.0, 0.0, -0.026721596717834473],
            ], dtype=torch.float32),
        },
        'num_03_NormalHuman': {
            'position': torch.tensor([
                [0.4831497073173523, 4.1441330909729, 0.13394999504089355],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [0.9996335506439209, 0.0, 0.0, -0.026712507009506226],
            ], dtype=torch.float32),
        },
        'num_04_NormalHuman': {
            'position': torch.tensor([
                [-0.08364272862672806, 4.325831890106201, 0.13394999504089355],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [0.9176362752914429, 0.0, 0.0, 0.3973970413208008],
            ], dtype=torch.float32),
        },
        'num_00_AGV': {
            'position': torch.tensor([
                [22.03065299987793, 6.2314863204956055, 0.13394999504089355],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float32),
        },
        'num_01_AGV': {
            'position': torch.tensor([
                [17.098709106445312, 8.431485176086426, 0.13394999504089355],
            ], dtype=torch.float32),
            'orientation': torch.tensor([
                [0.7071068286895752, 0.0, 0.0, 0.7071067690849304],
            ], dtype=torch.float32),
        },
    },
    'progress': {
        'product_order': {
            'ProductWaterPipe': 5,
        },
        'not_started': {
            'ProductWaterPipe': 4,
        },
        'next_product': 'ProductWaterPipe',
        'next_product_index': 1,
        'producing': ['ProductWaterPipe'],
        'producing_indexs': [0],
        'finished': {},
        'production_done': False,
        'ongoing_task_records': {
            '0': {
                'task_done': False,
                'task': 'logistic_for_pipe_cutting',
                'task_index': 1,
                'task_type': 'logistic',
                'is_final_task': False,
                'product': 'ProductWaterPipe',
                'product_index': 0,
                'new_product_selected': False,
                'submaterials': {
                    'product_00_pipe': {
                        'storage_name': 'BlackStorage_00',
                    },
                    'product_00_flange': {
                        'storage_name': 'GroundStorage_01',
                    },
                    'product_00_elbow': {
                        'storage_name': 'GroundStorage_00',
                    },
                    'product_00_semi': {
                        'storage_name': 'disappear',
                    },
                    'product_00_maded': {
                        'storage_name': 'disappear',
                    },
                },
                'logistic_submaterial': 'product_00_pipe',
                'processing_submaterials': None,
                'processed_material': None,
                'human': 'num_00_NormalHuman',
                'human_index': 0,
                'robot': 'num_00_AGV',
                'robot_index': 0,
                'target_machine': 'num02_rollerbedCNCPipeIntersectionCuttingMachine',
                'chosen_machine_workstation': 'num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station',
                'chosen_workstation_index': 0,
                'logistic_machine': 'num07_gantry_group',
                'chosen_gantry_index': 0,
                'task_start_time_step': 1,
                'subtasks_dict': {
                    'ongoing': ['go_to_material', 'go_to_material', 'wait', 'go_to_material'],
                    'ongoing_index': 0,
                    'required_logistic_material': 'product_00_pipe',
                    'material_start_area': 'BlackStorage_00',
                    'material_goal_area': 'num02_rollerbedCNCPipeIntersectionCuttingMachine',
                    'goal_area_workstation_key': 'num02_rollerbedCNCPipeIntersectionCuttingMachine_part01_station',
                    'start_area_ids': {
                        'human_working_areas_ids': [39],
                        'robot_parking_areas_ids': [38],
                        'gantry_parking_areas_ids': [39],
                    },
                    'goal_area_ids': {
                        'human_working_areas_ids': [90],
                        'robot_parking_areas_ids': [78],
                        'gantry_parking_areas_ids': [78],
                    },
                    'num_subtasks': 9,
                    'finished': [False, True, True, False],
                    'subtasks': [
                        ['go_to_material', 'go_to_material', 'wait', 'go_to_material'],
                        ['material_on_gantry', 'wait', 'wait', 'wait'],
                        ['control_gantry', 'carry_to_robot', 'wait', 'wait'],
                        ['material_on_robot', 'wait', 'wait', 'wait'],
                        ['go_to_goal_area', 'move_to_goal_area', 'wait', 'carry_to_goal_area'],
                        ['material_on_gantry', 'wait', 'wait', 'wait'],
                        ['control_gantry', 'carry_to_goal_area', 'wait', 'wait'],
                        ['material_on_goal_area', 'wait', 'wait', 'done'],
                        ['done', 'done', 'done', 'done'],
                    ],
                    'material_states_in_subtasks': {
                        'product_00_pipe': ['on_start_area', 'on_start_area', 'on_gantry', 'on_gantry', 'on_robot', 'on_robot', 'on_gantry', 'on_gantry', 'on_goal_area'],
                        'product_00_flange': ['on_start_area', 'on_start_area', 'on_start_area', 'on_start_area', 'on_start_area', 'on_start_area', 'on_start_area', 'on_start_area', 'on_start_area'],
                        'product_00_elbow': ['on_start_area', 'on_start_area', 'on_start_area', 'on_start_area', 'on_start_area', 'on_start_area', 'on_start_area', 'on_start_area', 'on_start_area'],
                        'product_00_semi': ['disappear', 'disappear', 'disappear', 'disappear', 'disappear', 'disappear', 'disappear', 'disappear', 'disappear'],
                        'product_00_maded': ['disappear', 'disappear', 'disappear', 'disappear', 'disappear', 'disappear', 'disappear', 'disappear', 'disappear'],
                    },
                },
                'next_processing_task': 'pipe_cutting',
                'next_logistic_task': 'logistic_for_pipe_grooving',
                'next_target_machine': 'num02_rollerbedCNCPipeIntersectionCuttingMachine',
                'next_chosen_machine_workstation': None,
                'next_chosen_workstation_index': None,
                'already_done_next_logistic_task': None,
            },
        },
    },
    'agent_action_mask': {
        'agent_A_product_sequencer': torch.tensor([0], dtype=torch.int32),
        'agent_B_product_selector': torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.int32),
        'agent_C_process_task_planner': torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.int32),
        'agent_D_human_robot_allocator': {
            'human_mask': torch.tensor([0, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int32),
            'robot_mask': torch.tensor([0, 1], dtype=torch.int32),
        },
        'human': {
            'task_availability_mask': torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32),
            'self_availability_mask': torch.tensor([0, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int32),
        },
        'robot': {
            'task_availability_mask': torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32),
            'self_availability_mask': torch.tensor([0, 1], dtype=torch.int32),
        },
        'machine': {
            'task_availability_mask': torch.tensor([1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.int32),
        },
        'material': {
            'task_availability_mask': torch.tensor([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=torch.int32),
        },
    },
    'action': {
        'product_sequencing': torch.tensor([0], dtype=torch.int32),
        'product_selection': torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.int32),
        'process_task_planning': torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        'human_robot_allocation': {
            'human': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
            'robot': torch.tensor([0, 0], dtype=torch.int32),
        },
    },
}
