# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
HC Factory Task Configuration
集中管理机器、人、AGV、材料的任务、动作、状态字典
"""

from typing import Dict, List
from dataclasses import dataclass


# ========================================
# 任务定义 Task Definitions
# ========================================

class TaskConfig:
    """任务配置类 - 定义所有高层任务"""
    
    # 高层任务字典 (High-level tasks) - DEMO版本：只有一个任务
    TASK_DICT = {
        -1: 'none',
        0: 'pipe_production',  # 水管生产 - 由num03机器完成
    }
    
    # 反向映射
    TASK_DICT_INVERSE = {v: k for k, v in TASK_DICT.items()}
    
    # 任务需要的资源类型 - DEMO版本：不需要人工、AGV和箱子
    TASK_RESOURCE_REQUIREMENTS = {
        'pipe_production': {'human': False, 'agv': False, 'box': False, 'machine': 'num03'},
    }


# ========================================
# 人类工人状态和任务 Human Worker States & Tasks
# ========================================

class HumanConfig:
    """人类工人配置类"""
    
    # 工人状态字典
    STATE_DICT = {
        0: 'idle',              # 空闲
        1: 'approaching',       # 接近目标
        2: 'waiting_agv',       # 等待AGV
        3: 'loading',           # 装载物料
        4: 'unloading',         # 卸载物料
        5: 'operating_machine', # 操作机器
        6: 'welding',           # 焊接
        7: 'cutting',           # 切割
        8: 'transporting',      # 运输
    }
    
    # 低层任务字典 (对应具体动作)
    LOW_LEVEL_TASK_DICT = {
        -1: 'none',
        0: 'idle',
        1: 'pick_material_A',
        2: 'pick_material_B',
        3: 'place_material_A',
        4: 'place_material_B',
        5: 'load_to_machine_01',
        6: 'load_to_machine_02',
        7: 'unload_from_machine_01',
        8: 'unload_from_machine_02',
        9: 'operate_machine',
        10: 'transport_to_station',
        # ... 根据实际需求扩展
    }
    
    # 高层任务到低层任务的映射
    HIGH_TO_LOW_TASK_MAPPING = {
        'task_placeholder_00': [1, 3],  # 示例：包含pick和place两个低层任务
        'task_placeholder_01': [5, 9],
        # ... 根据实际需求扩展
    }
    
    # 任务操作时间 (单位: 时间步)
    TASK_OPERATION_TIME = {
        'pick_material_A': 5,
        'pick_material_B': 5,
        'place_material_A': 5,
        'place_material_B': 5,
        'load_to_machine_01': 8,
        'operate_machine': 25,
        # ... 根据实际需求扩展
    }
    
    # 任务对应的目标位置键
    TASK_TARGET_POSE = {
        'pick_material_A': 'material_A_depot',
        'pick_material_B': 'material_B_depot',
        'load_to_machine_01': 'machine_01_loading_zone',
        # ... 根据实际需求扩展
    }


# ========================================
# AGV机器人状态和任务 AGV Robot States & Tasks
# ========================================

class AgvConfig:
    """AGV机器人配置类"""
    
    # AGV状态字典
    STATE_DICT = {
        0: 'idle',              # 空闲
        1: 'navigating',        # 导航中
        2: 'waiting_human',     # 等待工人
        3: 'docked',            # 已停靠
        4: 'transporting',      # 运输中
        5: 'returning',         # 返回中
    }
    
    # AGV任务字典
    TASK_DICT = {
        -1: 'none',
        0: 'idle',
        1: 'transport_material_A',
        2: 'transport_material_B',
        3: 'transport_to_station_01',
        4: 'transport_to_station_02',
        5: 'return_to_depot',
        # ... 根据实际需求扩展
    }
    
    # 导航速度配置
    NAVIGATION_SPEED = 1.0  # m/s
    
    # 停靠精度要求
    DOCKING_PRECISION = 0.1  # meters


# ========================================
# 运输箱配置 Transport Box Configuration
# ========================================

class BoxConfig:
    """运输箱配置类"""
    
    # 箱子容量配置
    @dataclass
    class Capacity:
        material_A: int = 4
        material_B: int = 2
        product: int = 1
        
    # 箱子状态
    STATE_DICT = {
        0: 'empty',         # 空
        1: 'partial',       # 部分装载
        2: 'full',          # 满载
        3: 'transporting',  # 运输中
    }


# ========================================
# 材料状态 Material States
# ========================================

class MaterialConfig:
    """材料配置类"""
    
    # 材料类型
    MATERIAL_TYPES = [
        'material_A',
        'material_B', 
        'material_C',
        'semi_product',
        'final_product',
    ]
    
    # 材料状态字典 (通用)
    COMMON_STATE_DICT = {
        -1: 'done',             # 已完成/已使用
        0: 'raw',               # 原始状态
        1: 'in_box',            # 在运输箱中
        2: 'at_depot',          # 在仓库/中转站
        3: 'in_queue',          # 在队列中
        4: 'processing',        # 加工中
        5: 'processed',         # 已加工
    }
    
    # Material A 特定状态
    MATERIAL_A_STATE_DICT = {
        -1: 'done',
        0: 'raw',               # 原料
        1: 'in_box',            # 在运输箱
        2: 'at_depot',          # 在仓库
        3: 'loading',           # 装载中
        4: 'loaded_machine',    # 已装载到机器
        5: 'processing',        # 加工中
        6: 'processed',         # 已加工
    }
    
    # Material B 特定状态
    MATERIAL_B_STATE_DICT = {
        -1: 'done',
        0: 'raw',
        1: 'in_box',
        2: 'at_depot',
        3: 'loading',
        4: 'loaded_machine',
        5: 'processing',
        6: 'processed',
    }
    
    # 产品状态
    PRODUCT_STATE_DICT = {
        0: 'waiting',           # 等待生产
        1: 'in_production',     # 生产中
        2: 'completed',         # 已完成
        3: 'collected',         # 已收集
        4: 'shipped',           # 已发货
    }


# ========================================
# 机器状态和配置 Machine States & Configuration
# ========================================

class MachineConfig:
    """机器配置类 - 8台主要机器"""
    
    # 机器列表 - DEMO版本：只使用num03
    MACHINE_LIST = [
        'num03_rollerbedCNCPipeIntersectionCuttingMachine',  # 只使用这一台
        # 'num01_rotaryPipeAutomaticWeldingMachine',
        # 'num02_weldingRobot',
        # 'num04_laserCuttingMachine',
        # 'num05_groovingMachineLarge',
        # 'num06_groovingMachineSmall',
        # 'num07_highPressureFoamingMachine',
        # 'num08_gantry_group',
    ]
    
    # 通用机器状态
    COMMON_STATE_DICT = {
        0: 'idle',              # 空闲
        1: 'resetting',         # 复位中
        2: 'ready',             # 就绪
        3: 'loading',           # 装载中
        4: 'processing',        # 加工中
        5: 'unloading',         # 卸载中
        6: 'error',             # 故障
        7: 'maintenance',       # 维护中
    }
    
    # 机器操作时间配置
    OPERATION_TIME = {
        'num01_rotaryPipeAutomaticWeldingMachine': {
            'welding_time': 25,
            'loading_time': 8,
            'unloading_time': 5,
            'reset_time': 100,
        },
        'num02_weldingRobot': {
            'welding_time': 25,
            'loading_time': 8,
            'unloading_time': 5,
            'reset_time': 100,
        },
        'num03_rollerbedCNCPipeIntersectionCuttingMachine': {
            'cutting_time': 30,
            'loading_time': 10,
            'unloading_time': 5,
            'reset_time': 100,
        },
        'num04_laserCuttingMachine': {
            'cutting_time': 25,
            'loading_time': 8,
            'unloading_time': 5,
            'reset_time': 100,
        },
        'num05_groovingMachineLarge': {
            'grooving_time': 20,
            'loading_time': 10,
            'unloading_time': 5,
            'reset_time': 100,
        },
        'num06_groovingMachineSmall': {
            'grooving_time': 15,
            'loading_time': 8,
            'unloading_time': 5,
            'reset_time': 100,
        },
        'num07_highPressureFoamingMachine': {
            'foaming_time': 30,
            'loading_time': 10,
            'unloading_time': 8,
            'reset_time': 100,
        },
        'num08_gantry_group': {
            'operation_time': 50,
            'reset_time': 100,
        },
    }
    
    # 每台机器的部件及其运动配置
    MACHINE_PARTS = {
        'num01_rotaryPipeAutomaticWeldingMachine': [
            'part_01_station',
            'part_02_station',
        ],
        'num02_weldingRobot': [
            'part02_robot_arm_and_base',
            'part04_mobile_base_for_material',
        ],
        'num03_rollerbedCNCPipeIntersectionCuttingMachine': [
            'part01_station',
            'part05_cutting_machine',
        ],
        'num04_laserCuttingMachine': [
            'main_unit',
        ],
        'num05_groovingMachineLarge': [
            'part01_large_fixed_base',
            'part02_large_mobile_base',
        ],
        'num06_groovingMachineSmall': [
            'part01_small_fixed_base',
            'part02_small_mobile_handle',
        ],
        'num07_highPressureFoamingMachine': [
            'main_unit',
        ],
        'num08_gantry_group': [
            'gantry_main',
        ],
    }


# ========================================
# 工艺流程定义 Process Flow Definitions
# ========================================

class ProcessConfig:
    """工艺流程配置"""
    
    # 产品生产流程 (示例)
    PRODUCT_WORKFLOW = [
        {
            'step': 1,
            'machine': 'num04_laserCuttingMachine',
            'input_materials': ['material_A'],
            'output_materials': ['processed_material_A'],
            'required_human_tasks': ['load_to_machine_01', 'unload_from_machine_01'],
        },
        {
            'step': 2,
            'machine': 'num05_groovingMachineLarge',
            'input_materials': ['processed_material_A'],
            'output_materials': ['grooved_material_A'],
            'required_human_tasks': ['load_to_machine_02', 'unload_from_machine_02'],
        },
        {
            'step': 3,
            'machine': 'num01_rotaryPipeAutomaticWeldingMachine',
            'input_materials': ['grooved_material_A', 'material_B'],
            'output_materials': ['semi_product'],
            'required_human_tasks': ['load_materials', 'unload_product'],
        },
        # ... 根据实际工艺流程扩展
    ]
    
    # 材料依赖关系
    MATERIAL_DEPENDENCIES = {
        'semi_product': ['processed_material_A', 'material_B'],
        'final_product': ['semi_product', 'material_C'],
    }


# ========================================
# 位置和路径配置 Position & Path Configuration
# ========================================

class LocationConfig:
    """位置配置类"""
    
    # 关键位置字典 (placeholder - 实际坐标需要从环境中获取)
    LOCATIONS = {
        'material_A_depot': {'x': 0, 'y': 0, 'z': 0},
        'material_B_depot': {'x': 0, 'y': 0, 'z': 0},
        'machine_01_loading_zone': {'x': 0, 'y': 0, 'z': 0},
        'machine_02_loading_zone': {'x': 0, 'y': 0, 'z': 0},
        'product_collection_zone': {'x': 0, 'y': 0, 'z': 0},
        'human_initial_pos': {'x': 0, 'y': 0, 'z': 0},
        'agv_initial_pos': {'x': 0, 'y': 0, 'z': 0},
        # ... 根据实际环境扩展
    }
    
    # 路径节点 (用于导航)
    PATH_NODES = {
        'node_01': {'x': 0, 'y': 0},
        'node_02': {'x': 0, 'y': 0},
        # ... 根据地图扩展
    }


# ========================================
# 时间配置 Timing Configuration
# ========================================

class TimingConfig:
    """时间配置类"""
    
    # 随机时间范围
    MACHINE_TIME_RANDOM = 3  # 机器操作时间随机范围
    HUMAN_TIME_RANDOM = 2    # 人工操作时间随机范围
    
    # 标准操作时间
    STANDARD_TIMES = {
        'human_loading': 8,
        'human_unloading': 5,
        'human_picking': 5,
        'human_placing': 5,
        'cutting_operation': 25,
        'welding_operation': 25,
        'grooving_operation': 20,
        'foaming_operation': 30,
    }


# ========================================
# 疲劳模型配置 Fatigue Model Configuration
# ========================================

class FatigueConfig:
    """疲劳模型配置类"""
    
    # 疲劳阈值
    PHYSICAL_FATIGUE_THRESHOLD = 0.95
    PSYCHOLOGICAL_FATIGUE_THRESHOLD = 0.8
    
    # 疲劳恢复系数
    RECOVERY_COEFFICIENT = {
        'physical': 0.1,
        'psychological': 0.15,
    }
    
    # 任务疲劳系数 (不同任务导致的疲劳累积)
    TASK_FATIGUE_COEFFICIENTS = {
        'pick_material_A': {'physical': 0.02, 'psychological': 0.01},
        'load_to_machine_01': {'physical': 0.05, 'psychological': 0.03},
        'operate_machine': {'physical': 0.03, 'psychological': 0.04},
        # ... 根据实际任务扩展
    }


# ========================================
# 观察空间配置 Observation Space Configuration
# ========================================

class ObservationConfig:
    """观察空间配置"""
    
    # 观察空间维度
    DIMENSIONS = {
        'material_states': 30,      # 物料状态
        'machine_states': 16,       # 机器状态
        'task_mask': 10,            # 任务掩码
        'human_states': 12,         # 工人状态
        'agv_states': 9,            # AGV状态
        'fatigue_states': 6,        # 疲劳状态
    }
    
    # 总观察空间维度
    TOTAL_DIM = sum(DIMENSIONS.values())


# ========================================
# 奖励配置 Reward Configuration
# ========================================

class RewardConfig:
    """奖励配置类"""
    
    # 奖励权重
    WEIGHTS = {
        'task_completion': 1.0,
        'time_penalty': -0.002,
        'action_penalty': -0.1,
        'progress_reward': 0.2,
        'fatigue_penalty': -0.1,
        'overwork_penalty': -0.5,
    }
    
    # 完成奖励
    COMPLETION_REWARDS = {
        'task_finished_early': 1.0,
        'task_finished_on_time': 0.6,
        'task_incomplete': -1.5,
    }


# ========================================
# 辅助函数 Helper Functions
# ========================================

def get_task_name(task_id: int) -> str:
    """根据任务ID获取任务名称"""
    return TaskConfig.TASK_DICT.get(task_id, 'unknown')

def get_task_id(task_name: str) -> int:
    """根据任务名称获取任务ID"""
    return TaskConfig.TASK_DICT_INVERSE.get(task_name, -1)

def get_human_state_name(state_id: int) -> str:
    """根据状态ID获取人类状态名称"""
    return HumanConfig.STATE_DICT.get(state_id, 'unknown')

def get_agv_state_name(state_id: int) -> str:
    """根据状态ID获取AGV状态名称"""
    return AgvConfig.STATE_DICT.get(state_id, 'unknown')

def get_material_state_name(material_type: str, state_id: int) -> str:
    """根据材料类型和状态ID获取状态名称"""
    if material_type == 'material_A':
        return MaterialConfig.MATERIAL_A_STATE_DICT.get(state_id, 'unknown')
    elif material_type == 'material_B':
        return MaterialConfig.MATERIAL_B_STATE_DICT.get(state_id, 'unknown')
    elif material_type == 'product':
        return MaterialConfig.PRODUCT_STATE_DICT.get(state_id, 'unknown')
    else:
        return MaterialConfig.COMMON_STATE_DICT.get(state_id, 'unknown')

def get_machine_operation_time(machine_name: str, operation_type: str) -> int:
    """获取机器操作时间"""
    machine_times = MachineConfig.OPERATION_TIME.get(machine_name, {})
    return machine_times.get(operation_type, 0)


# ========================================
# 配置验证 Configuration Validation
# ========================================

def validate_config():
    """验证配置的一致性 - DEMO版本"""
    errors = []
    
    # 检查任务字典是否有重复
    if len(TaskConfig.TASK_DICT) != len(set(TaskConfig.TASK_DICT.values())):
        errors.append("TaskConfig.TASK_DICT has duplicate values")
    
    # 检查反向字典是否正确
    if TaskConfig.TASK_DICT_INVERSE != {v: k for k, v in TaskConfig.TASK_DICT.items()}:
        errors.append("TaskConfig.TASK_DICT_INVERSE is inconsistent")
    
    # DEMO版本：机器数量可以是1或8
    if len(MachineConfig.MACHINE_LIST) not in [1, 8]:
        errors.append(f"Expected 1 (DEMO) or 8 (FULL) machines, got {len(MachineConfig.MACHINE_LIST)}")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return True


if __name__ == "__main__":
    # 运行配置验证
    try:
        validate_config()
        print("✓ Configuration validation passed!")
        print(f"\nConfiguration Summary:")
        print(f"  - Tasks defined: {len(TaskConfig.TASK_DICT)}")
        print(f"  - Human states: {len(HumanConfig.STATE_DICT)}")
        print(f"  - AGV states: {len(AgvConfig.STATE_DICT)}")
        print(f"  - Machines: {len(MachineConfig.MACHINE_LIST)}")
        print(f"  - Material types: {len(MaterialConfig.MATERIAL_TYPES)}")
        print(f"  - Total observation dimensions: {ObservationConfig.TOTAL_DIM}")
    except ValueError as e:
        print(f"✗ Configuration validation failed: {e}")
