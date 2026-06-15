from __future__ import annotations

import torch


class MARLObsEncoder:
    """Encode env_state_action_dict into fixed-size float vectors for each agent."""

    def __init__(self, cuda_device: torch.device, parallel_producing_limit: int = 5):
        self.cuda_device = cuda_device
        self.parallel_producing_limit = parallel_producing_limit

    def _progress_features(self, env_state_action_dict: dict) -> torch.Tensor:
        progress = env_state_action_dict["progress"]
        producing_count = float(len(progress["producing"]))
        finished_count = float(sum(len(v) for v in progress["finished"].values()))
        not_started_count = float(sum(len(v) for v in progress["not_started"].values()))
        ongoing_count = float(len(progress["ongoing_task_records"]))
        has_next = float(progress["next_product"] is not None)
        time_step = float(env_state_action_dict["time_step"]) / 10000.0
        return torch.tensor(
            [producing_count, finished_count, not_started_count, ongoing_count, has_next, time_step],
            dtype=torch.float32,
            device=self.cuda_device,
        )

    def _resource_summary(self, env_state_action_dict: dict) -> torch.Tensor:
        human_free = sum(1 for h in env_state_action_dict["human"].values() if h["state"] == "free")
        robot_free = sum(1 for r in env_state_action_dict["robot"].values() if r["state"] == "free")
        return torch.tensor(
            [float(human_free), float(robot_free)],
            dtype=torch.float32,
            device=self.cuda_device,
        )

    def encode_A(self, env_state_action_dict: dict) -> torch.Tensor:
        mask = env_state_action_dict["agent_action_mask"]["agent_A_product_sequencer"].float()
        return torch.cat([self._progress_features(env_state_action_dict), self._resource_summary(env_state_action_dict), mask])

    def encode_B(self, env_state_action_dict: dict, product_sequencing_action: torch.Tensor | None) -> torch.Tensor:
        mask = env_state_action_dict["agent_action_mask"]["agent_B_product_selector"].float()
        a_dim = env_state_action_dict["agent_action_mask"]["agent_A_product_sequencer"].shape[0]
        if product_sequencing_action is not None:
            a_action = product_sequencing_action.float()
        else:
            a_action = torch.zeros(a_dim, dtype=torch.float32, device=self.cuda_device)
        return torch.cat([self._progress_features(env_state_action_dict), self._resource_summary(env_state_action_dict), a_action, mask])

    def encode_C(
        self,
        env_state_action_dict: dict,
        product_selection_action: torch.Tensor,
    ) -> torch.Tensor:
        task_mask = env_state_action_dict["agent_action_mask"]["agent_C_process_task_planner"].float()
        human_task_mask = env_state_action_dict["agent_action_mask"]["human"]["task_availability_mask"].float()
        machine_task_mask = env_state_action_dict["agent_action_mask"]["machine"]["task_availability_mask"].float()
        b_action = product_selection_action.float()
        return torch.cat(
            [
                self._progress_features(env_state_action_dict),
                self._resource_summary(env_state_action_dict),
                b_action,
                task_mask.flatten(),
                human_task_mask,
                machine_task_mask,
            ]
        )

    def encode_D(
        self,
        env_state_action_dict: dict,
        process_task_planning_action: torch.Tensor,
    ) -> torch.Tensor:
        human_mask = env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"].float()
        robot_mask = env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"].float()
        c_action = process_task_planning_action.float()
        return torch.cat(
            [
                self._progress_features(env_state_action_dict),
                self._resource_summary(env_state_action_dict),
                c_action,
                human_mask,
                robot_mask,
            ]
        )

    def get_obs_dim_A(self, env_state_action_dict: dict) -> int:
        return self.encode_A(env_state_action_dict).shape[0]

    def get_obs_dim_B(self, env_state_action_dict: dict) -> int:
        return self.encode_B(env_state_action_dict, None).shape[0]

    def get_obs_dim_C(self, env_state_action_dict: dict, product_selection_action: torch.Tensor) -> int:
        return self.encode_C(env_state_action_dict, product_selection_action).shape[0]

    def get_obs_dim_D(self, env_state_action_dict: dict, process_task_planning_action: torch.Tensor) -> int:
        return self.encode_D(env_state_action_dict, process_task_planning_action).shape[0]
