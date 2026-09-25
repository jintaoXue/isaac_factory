"""Configuration for the engine-free backend (no USD/Isaac dependency)."""
import copy
from types import SimpleNamespace


class HcLogicEnvCfg:
    single_env_parallel_producing_limit = 10
    human_number_upper_bound = 6
    robot_upper_bound = 4
    material_batch_upper_bound = 16
    max_episodic_steps = 64000
    rl_step_penalty = 0.08
    rl_finish_bonus = 2.0
    rl_task_bonus = 0.1
    rl_success_bonus = 50.0
    cuda_device_str = "cuda:0"
    seed = 42
    train_cfg = None
    ui_window_class_type = None

    def __init__(self, **kwargs):
        for k, v in vars(type(self)).items():
            if not k.startswith("_") and not callable(v) and not isinstance(v, (classmethod, staticmethod)):
                setattr(self, k, copy.deepcopy(v))
        self.scene = SimpleNamespace(num_envs=1)
        self.sim = SimpleNamespace(device="cuda:0")
        for k, v in kwargs.items():
            setattr(self, k, SimpleNamespace(**v) if k in ("scene", "sim") else v)

    def to_dict(self):
        return {k: vars(v) if isinstance(v, SimpleNamespace) else v for k, v in vars(self).items()}

    def _valid_train_cfg(self):
        return self.train_cfg is not None
