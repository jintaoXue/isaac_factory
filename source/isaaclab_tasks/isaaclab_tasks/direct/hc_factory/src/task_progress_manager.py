
from ..env_asset_cfg.cfg_storage import CfgStorage
from ..env_asset_cfg.cfg_material_product import CfgProductOrder, CfgProductProcess
from ..env_asset_cfg.cfg_process_task_gallery import CfgProcessTaskGalleryInAll, CfgProcessTaskGalleryDetailedClassified, CfgSubtaskGallery, TaskRecordTemplate
from ..env_asset_cfg.cfg_machine import CfgMachine
from ..env_asset_cfg.cfg_hc_env import HcVectorEnvCfg
from ..env_asset_cfg.cfg_gantry_zone import get_gantry_zone
from .material import find_free_storage, reserve_storage_slot, finalize_material_batch_task_done
import torch
import copy


def staging_slot_index(parallel_producing_limit: int | None = None) -> int:
    """Last B/C slot index reserved for ``next_product`` staging."""
    if parallel_producing_limit is None:
        parallel_producing_limit = HcVectorEnvCfg().single_env_parallel_producing_limit
    return int(parallel_producing_limit)


def find_free_gantry_index(
    gantry_states: list,
    logistic_machine: str = "num07_gantry_group",
    preferred_zone: int | None = None,
) -> int | None:
    """Return a free active gantry. Prefer preferred_zone when it is active."""
    active_indices = CfgMachine[logistic_machine]["active_gantry_indices"]
    if preferred_zone is not None and preferred_zone in active_indices:
        if gantry_states[preferred_zone] == "free":
            return preferred_zone
        return None  # wait for the preferred active crane
    for gantry_index in active_indices:
        if gantry_states[gantry_index] == "free":
            return gantry_index
    return None


def find_free_robot_name(env_state_action_dict: dict) -> str | None:
    for robot_name, robot_state in env_state_action_dict["robot"].items():
        if robot_state["state"] == "free":
            return robot_name
    return None


def find_workstation_index_for_task(states: list, task_type: str, task_name: str) -> int | None:
    if task_type == "logistic":
        try:
            return states.index("free")
        except ValueError:
            return None
    if task_type == "processing":
        for i, state in enumerate(states):
            if state == "free":
                continue
            pre_name = state.split("_")[0]
            state_task = state.split("_", 1)[1]
            if pre_name == "materialReadyFor" and state_task == task_name:
                return i
    return None


class TaskManager:
    def __init__(
        self,
        cuda_device: torch.device,
        max_episodic_steps: int = 45000,
        step_penalty: float = 0.01,
        finish_bonus: float = 2.0,
        task_bonus: float = 0.1,
        success_bonus: float = 20.0,
    ):
        self.cuda_device = cuda_device
        self.cfg_storage = CfgStorage
        self.inverse_index_to_task_name = {v: k for k, v in CfgProcessTaskGalleryInAll.items()}
        self.max_episodic_steps = int(max_episodic_steps)
        self.step_penalty = float(step_penalty)
        self.finish_bonus = float(finish_bonus)
        self.task_bonus = float(task_bonus)
        self.success_bonus = float(success_bonus)

    def reset(self, env_state_action_dict) -> dict:
        #production progress reset
        env_state_action_dict["progress"]["product_order"] = copy.deepcopy(CfgProductOrder)
        env_state_action_dict["progress"]["not_started"] = copy.deepcopy(CfgProductOrder)
        env_state_action_dict["progress"]["next_product"] = None
        env_state_action_dict["progress"]["next_product_index"] = None
        env_state_action_dict["progress"]["producing"] = []
        env_state_action_dict["progress"]["producing_indexs"] = []
        env_state_action_dict["progress"]["finished"] = {}
        env_state_action_dict["progress"]["ongoing_task_records"] = {}
        env_state_action_dict["progress"]["production_done"] = False
        env_state_action_dict["progress"]["stage_wip_cap"] = env_state_action_dict["progress"].get(
            "stage_wip_cap", int(HcVectorEnvCfg().single_env_parallel_producing_limit)
        )
        env_state_action_dict["rl"] = {
            "reward": 0.0,
            "done": False,
            "truncated": False,
            "success": False,
            "reward_parts": {
                "step": 0.0,
                "finish": 0.0,
                "task": 0.0,
                "success": 0.0,
            },
        }

    def step(self, env_state_action_dict: dict) -> dict:

        self.decode_action_product_sequencing(env_state_action_dict)

        action = env_state_action_dict.get("action") or {}
        dispatch_list = action.get("dispatch_list")
        if dispatch_list:
            for item in dispatch_list:
                new_task_record = copy.deepcopy(TaskRecordTemplate)
                if not self.decode_dispatch_item(env_state_action_dict, item, new_task_record):
                    continue
                if self.decode_process_task_planning_tensor(
                    item["process_task_planning"], new_task_record
                ):
                    self.decode_human_robot_allocation_dict(
                        item["human_robot_allocation"], env_state_action_dict, new_task_record
                    )
                    # May return False (no AGV / preferred gantry busy) — skip, retry later.
                    self.update_new_task_record(env_state_action_dict, new_task_record)
        else:
            new_task_record = copy.deepcopy(TaskRecordTemplate)
            if self.decode_action_product_selection(env_state_action_dict, new_task_record):
                have_new_task = self.decode_action_process_task_planning(env_state_action_dict, new_task_record)
                if have_new_task:
                    self.decode_action_human_robot_allocation(env_state_action_dict, new_task_record)
                    self.update_new_task_record(env_state_action_dict, new_task_record)

        step_stats = self.step_task_records(env_state_action_dict)
        self.check_done_production(env_state_action_dict)
        # time_step 在 HcSingleEnvBase 里于 managers 之后 +1；此处用 +1 判断超时
        self.update_rl_signals(env_state_action_dict, step_stats)
        return env_state_action_dict

    def update_rl_signals(self, env_state_action_dict: dict, step_stats: dict | None = None) -> dict:
        """Write per-step reward / done into ``env_state_action_dict["rl"]``.

        v2 reward:
            ``-step_penalty``
            ``+ finish_bonus * Δfinished_products``
            ``+ task_bonus * Δcompleted_process_tasks``
            ``+ success_bonus * (1 - t/T_budget)`` on segment success
        Terminal: ``n_finished >= segment_target_nfin`` → success;
        ``time_step+1 >= max_episodic_steps`` (segment T_budget) → truncated.
        """
        if "rl" not in env_state_action_dict or not isinstance(env_state_action_dict["rl"], dict):
            env_state_action_dict["rl"] = {
                "reward": 0.0,
                "done": False,
                "truncated": False,
                "success": False,
                "reward_parts": {},
            }
        rl = env_state_action_dict["rl"]
        progress = env_state_action_dict.get("progress") or {}
        step_stats = step_stats or {}
        n_finished = int(step_stats.get("n_product_finished", 0) or 0)
        n_task_done = int(step_stats.get("n_task_done", 0) or 0)
        # managers 步进时 time_step 尚未 +1，本步结束后的步号为 time_step+1
        t_after = int(env_state_action_dict.get("time_step", 0) or 0) + 1

        part_step = -self.step_penalty
        part_finish = self.finish_bonus * n_finished
        part_task = self.task_bonus * n_task_done
        part_success = 0.0

        done = False
        truncated = False
        success = False

        if bool(progress.get("production_done")):
            done = True
            success = True
            # earlier finish → larger bonus (segment T_budget = max_episodic_steps)
            remain = max(0.0, 1.0 - float(t_after) / float(max(1, self.max_episodic_steps)))
            part_success = self.success_bonus * remain
        elif t_after >= self.max_episodic_steps:
            done = True
            truncated = True

        reward = part_step + part_finish + part_task + part_success
        rl["reward"] = float(reward)
        rl["done"] = bool(done)
        rl["truncated"] = bool(truncated)
        rl["success"] = bool(success)
        rl["reward_parts"] = {
            "step": float(part_step),
            "finish": float(part_finish),
            "task": float(part_task),
            "success": float(part_success),
        }
        return env_state_action_dict

    def check_done_production(self, env_state_action_dict: dict) -> bool:
        """True when the current segment target (or full order) is reached."""
        progress = env_state_action_dict["progress"]
        finished = progress.get("finished", {})
        n_fin = 0
        for v in finished.values():
            n_fin += len(v) if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) else int(v or 0)

        target = progress.get("segment_target_nfin")
        if target is not None:
            if n_fin >= int(target):
                progress["production_done"] = True
                return True
            progress["production_done"] = False
            return False

        product_order = progress["product_order"]
        for product_type, required in product_order.items():
            if len(finished.get(product_type, [])) < required:
                progress["production_done"] = False
                return False
        progress["production_done"] = True
        return True

    def decode_action_product_sequencing(self, env_state_action_dict):
        action_product_sequencing = env_state_action_dict["action"]["product_sequencing"]
        # Return a torch tensor filled with zeros that matches the shape of action_product_sequencing,
        # placed on the correct device.
        if action_product_sequencing.sum() != 0:
            product_type_index = action_product_sequencing.nonzero()[0][0]
            product_type = list(CfgProductProcess.keys())[product_type_index]
            #shape is (num_material_batch, )
            material_states_dict = env_state_action_dict["material"]
            #find the next_product_index in material_batch list that is the right product type
            for material_name, material_state in material_states_dict.items():
                key_variables = material_state["key_variables"]
                finished_task = material_state["finished_task"]
                ongoing_task_record_index = material_state["ongoing_task_record_index"]
                if key_variables["type_name"] == product_type and finished_task == "none" and ongoing_task_record_index is None:
                    env_state_action_dict["progress"]["next_product"] = product_type
                    env_state_action_dict["progress"]["next_product_index"] = key_variables["idx"]
                    break

    def decode_dispatch_item(self, env_state_action_dict: dict, item: dict, new_task_record: dict) -> bool:
        slot_index = int(item["slot_index"])
        return self.resolve_slot_to_product(env_state_action_dict, slot_index, new_task_record)

    def resolve_slot_to_product(self, env_state_action_dict: dict, slot_index: int, new_task_record: dict) -> bool:
        staging = staging_slot_index()
        progress = env_state_action_dict["progress"]
        producing = progress["producing"]
        producing_indexs = progress["producing_indexs"]
        if slot_index == staging:
            if progress["next_product"] is None or progress["next_product_index"] is None:
                return False
            new_task_record["product"] = progress["next_product"]
            new_task_record["product_index"] = progress["next_product_index"]
            new_task_record["from_staging_slot"] = True
        elif slot_index < len(producing):
            new_task_record["product"] = producing[slot_index]
            new_task_record["product_index"] = producing_indexs[slot_index]
            new_task_record["from_staging_slot"] = False
        else:
            return False
        material_name = f"num_{new_task_record['product_index']:02d}_{new_task_record['product']}"
        new_task_record["submaterials"] = env_state_action_dict["material"][material_name]["submaterials"]
        return True

    def decode_action_product_selection(self, env_state_action_dict, new_task_record):
        action_product_selection = env_state_action_dict["action"]["product_selection"]
        if action_product_selection.sum() == 0:
            return False
        _index = int(action_product_selection.nonzero()[0][0].item())
        return self.resolve_slot_to_product(env_state_action_dict, _index, new_task_record)

    def decode_process_task_planning_tensor(self, action_process_task_planning, new_task_record: dict) -> bool:
        if action_process_task_planning.sum() == 0:
            action_process_task_planning = action_process_task_planning.clone()
            action_process_task_planning[0] = 1
        _index = int(action_process_task_planning.nonzero()[0][0].item())
        new_task_record["task"] = self.inverse_index_to_task_name[_index]
        new_task_record["task_index"] = _index
        new_task_record["task_type"] = CfgProcessTaskGalleryDetailedClassified[new_task_record["product"]][
            new_task_record["task"]
        ]["task_type"]
        return new_task_record["task"] != "none"

    def decode_action_process_task_planning(self, env_state_action_dict, new_task_record):
        return self.decode_process_task_planning_tensor(
            env_state_action_dict["action"]["process_task_planning"], new_task_record
        )

    def decode_human_robot_allocation_dict(
        self, action_human_robot_allocation: dict, env_state_action_dict: dict, new_task_record: dict
    ) -> dict:
        if new_task_record["task"] is None or new_task_record["task"] == "none":
            return new_task_record
        action_human = action_human_robot_allocation["human"]
        action_robot = action_human_robot_allocation["robot"]
        human_keys = list(env_state_action_dict["human"].keys())
        robot_keys = list(env_state_action_dict["robot"].keys())
        if action_human.sum() == 1:
            _index = action_human.nonzero()[0][0].item()
            if _index < len(human_keys):
                new_task_record["human"] = human_keys[_index]
                new_task_record["human_index"] = _index
        if action_robot.sum() != 0 and new_task_record["task_type"] in ("logistic", "processing"):
            _index = action_robot.nonzero()[0][0].item()
            if _index < len(robot_keys):
                new_task_record["robot"] = robot_keys[_index]
                new_task_record["robot_index"] = _index
        return new_task_record

    def decode_action_human_robot_allocation(self, env_state_action_dict, new_task_record):
        if new_task_record["task"] is None or new_task_record["task"] == "none":
            return
        return self.decode_human_robot_allocation_dict(
            env_state_action_dict["action"]["human_robot_allocation"],
            env_state_action_dict,
            new_task_record,
        )

    def update_new_task_record(self, env_state_action_dict, new_task_record: dict):
        assert new_task_record["human"] != None, "Human availablity should be check by mask before the task can be selected"
        assert new_task_record["product_index"] not in env_state_action_dict["progress"]["ongoing_task_records"], "The product index should not be in the ongoing task records"
        assert new_task_record["task"] != "none", "The task should not be none"
        product_type = new_task_record["product"]
        new_task_record.update(copy.deepcopy(CfgProcessTaskGalleryDetailedClassified[product_type][new_task_record["task"]]))
        ##machine information
        states = env_state_action_dict["machine"][new_task_record["target_machine"]]["state"]
        workstation_index = find_workstation_index_for_task(
            states, new_task_record["task_type"], new_task_record["task"]
        )
        if workstation_index is None:
            raise ValueError(
                f"No workstation for task {new_task_record['task']} on {new_task_record['target_machine']}"
            )
        new_task_record["chosen_machine_workstation"] = \
            list(env_state_action_dict["machine"][new_task_record["target_machine"]]["key_variables"]["working_area_ids"].keys())[workstation_index]
        new_task_record["chosen_workstation_index"] = workstation_index
        new_task_record["task_start_time_step"] = int(env_state_action_dict["time_step"])
        # Build subtasks first (sets preferred_gantry_zone / may drop AGV on same-zone).
        subtasks = self.initialze_subtasks(env_state_action_dict, new_task_record)
        if subtasks is None:
            # Cross-zone logistic with no free AGV: skip start (wait for a later dispatch).
            return False
        new_task_record["subtasks_dict"] = subtasks
        if new_task_record["task_type"] == "logistic":
            chosen_gantry_index = self._find_free_gantry(env_state_action_dict, new_task_record)
            if chosen_gantry_index is None:
                # Preferred-zone crane busy (mask only checks "any free gantry"). Wait.
                return False
            new_task_record["chosen_gantry_index"] = chosen_gantry_index

        env_state_action_dict["progress"]["ongoing_task_records"][new_task_record["product_index"]] = new_task_record
        self.apply_new_task_record_to_human_robot_machine_material(env_state_action_dict, new_task_record)
        if new_task_record.get("from_staging_slot"):
            self._commit_new_product_to_producing(env_state_action_dict, new_task_record)
        return True

    def _commit_new_product_to_producing(self, env_state_action_dict: dict, task_record: dict) -> None:
        """5.2b: dispatch from staging slot immediately enters WIP."""
        product_type = task_record["product"]
        product_index = task_record["product_index"]
        progress = env_state_action_dict["progress"]
        progress["producing"].append(product_type)
        progress["producing_indexs"].append(product_index)
        progress["next_product"] = None
        progress["next_product_index"] = None
        progress["not_started"][product_type] -= 1
        task_record["new_product_selected"] = False

    def _find_free_gantry(self, env_state_action_dict, task_record):
        gantry_states: list[str] = env_state_action_dict["machine"][task_record["logistic_machine"]]["state"]
        return find_free_gantry_index(
            gantry_states,
            task_record["logistic_machine"],
            preferred_zone=task_record.get("preferred_gantry_zone"),
        )

    def initialze_subtasks(self, env_state_action_dict, task_record):
        if task_record["task_type"] == "logistic":
            product_name = f"num_{task_record['product_index']:02d}_{task_record['product']}"
            required_logistic_material = task_record["logistic_submaterial"]
            state_material = env_state_action_dict["material"][product_name]["submaterials"][required_logistic_material]
            assert state_material["storage_name"] is not None, "The storage name should be initialized in material.py"
            assert state_material["storage_name"] != "disappear", "The material still not appeared"
            material_start_area = state_material["storage_name"]
            gallery_task = CfgSubtaskGallery[task_record["product"]][task_record["task"]]
            # Peek goal from template (before deepcopy) to choose same/cross template.
            peek_goal = gallery_task["have_AGV"]["material_goal_area"]
            machine_workstation_key = task_record["chosen_machine_workstation"]
            start_zone = get_gantry_zone(material_start_area, None)
            goal_zone = get_gantry_zone(peek_goal, machine_workstation_key)
            same_zone = start_zone is not None and start_zone == goal_zone

            # Hard rule: same-zone = gantry only; cross-zone = AGV required (never single-crane cross).
            if same_zone:
                subtasks = copy.deepcopy(gallery_task["only_have_gantry"])
                task_record["robot"] = None
                task_record["robot_index"] = None
                task_record["goal_gantry_zone"] = None
                task_record["preferred_gantry_zone"] = start_zone
            else:
                if task_record.get("robot") is None:
                    # Grab a free AGV if Agent D did not assign one.
                    robot_name = find_free_robot_name(env_state_action_dict)
                    if robot_name is None:
                        # Hard wait: do not start cross-zone logistic without AGV.
                        return None
                    robot_keys = list(env_state_action_dict["robot"].keys())
                    task_record["robot"] = robot_name
                    task_record["robot_index"] = robot_keys.index(robot_name)
                subtasks = copy.deepcopy(gallery_task["have_AGV"])
                task_record["preferred_gantry_zone"] = start_zone
                task_record["goal_gantry_zone"] = goal_zone

            assert subtasks["material_start_area"] != subtasks["material_goal_area"], "The material start area and goal area should be different, \
                otherwise dont need to logistic this material"
            subtasks["material_start_area"] = material_start_area
            if subtasks["material_start_area"] in env_state_action_dict["machine"]:
                chosen_machine_workstation_key = task_record["chosen_machine_workstation"]
                subtasks["start_area_ids"] = env_state_action_dict["machine"][subtasks["material_start_area"]]["key_variables"]["working_area_ids"][chosen_machine_workstation_key]
            elif subtasks["material_start_area"] in env_state_action_dict["storage"]:
                subtasks["start_area_ids"] = env_state_action_dict["storage"][subtasks["material_start_area"]]["key_variables"]["working_area_ids"]
            else:
                raise ValueError(f"Invalid material start area: {subtasks['material_start_area']}")
            subtasks["goal_area_ids"] = subtasks["goal_area_ids"][machine_workstation_key]
            subtasks["goal_area_workstation_key"] = machine_workstation_key
        elif task_record["task_type"] == "processing":
            # Goal (machine/storage) is unknown until after process → start with prefix only.
            # outbound_same_zone / outbound_cross_zone are appended in _attach_processing_outbound.
            subtasks = copy.deepcopy(
                CfgSubtaskGallery[task_record["product"]][task_record["task"]]["process_prefix"]
            )
            assert subtasks["material_start_area"] in env_state_action_dict["machine"]
            machine_workstation_key = task_record["chosen_machine_workstation"]
            subtasks["start_area_ids"] = subtasks["start_area_ids"][machine_workstation_key]
            task_record["preferred_gantry_zone"] = get_gantry_zone(
                subtasks["material_start_area"], machine_workstation_key
            )
            task_record["goal_gantry_zone"] = None
            subtasks["outbound_attached"] = False
        else:
            raise ValueError(f"Invalid task type: {task_record['task_type']}")
        return subtasks

    def _release_task_robot(self, env_state_action_dict, task_record):
        """Free a robot that was reserved but is not needed (same-zone outbound)."""
        robot = task_record.get("robot")
        if robot is None:
            return
        robot_state = env_state_action_dict["robot"][robot]
        if robot_state.get("ongoing_task_record_index") == task_record["product_index"]:
            robot_state["ongoing_task_record_index"] = None
            robot_state["state"] = "free"
        task_record["robot"] = None
        task_record["robot_index"] = None

    def _allocate_task_robot(self, env_state_action_dict, task_record) -> bool:
        """Occupy a free AGV for cross-zone outbound. Returns False if none free."""
        if task_record.get("robot") is not None:
            return True
        robot_name = find_free_robot_name(env_state_action_dict)
        if robot_name is None:
            return False
        robot_keys = list(env_state_action_dict["robot"].keys())
        task_record["robot"] = robot_name
        task_record["robot_index"] = robot_keys.index(robot_name)
        env_state_action_dict["robot"][robot_name]["ongoing_task_record_index"] = task_record["product_index"]
        env_state_action_dict["robot"][robot_name]["state"] = "working_" + task_record["task"]
        return True

    def _attach_processing_outbound(self, env_state_action_dict, task_record):
        """After goal is known, append same-zone or cross-zone outbound tail."""
        subtasks = task_record["subtasks_dict"]
        if subtasks.get("outbound_attached") or subtasks.get("goal_area_ids") is None:
            return

        gallery = CfgSubtaskGallery[task_record["product"]][task_record["task"]]
        start_zone = get_gantry_zone(
            subtasks["material_start_area"], task_record.get("chosen_machine_workstation")
        )
        goal_zone = get_gantry_zone(
            subtasks["material_goal_area"], subtasks.get("goal_area_workstation_key")
        )
        same_zone = start_zone is not None and start_zone == goal_zone

        if same_zone:
            self._release_task_robot(env_state_action_dict, task_record)
            tail = copy.deepcopy(gallery["outbound_same_zone"])
            subtasks["outbound_mode"] = "same_zone"
        else:
            # Hard rule: cross-zone must use AGV; wait (do not attach) until one is free.
            if not self._allocate_task_robot(env_state_action_dict, task_record):
                return
            tail = copy.deepcopy(gallery["outbound_cross_zone"])
            subtasks["outbound_mode"] = "cross_zone"
            # Next finding_free_gantry should take the goal-zone crane.
            task_record["preferred_gantry_zone"] = goal_zone
            task_record["goal_gantry_zone"] = goal_zone
            # Do not expand finished[] yet — still on a 3-col prefix row; pad on advance.

        subtasks["subtasks"].extend(tail["subtasks"])
        for mat_name, mat_states in tail["material_states_in_subtasks"].items():
            prefix_states = subtasks["material_states_in_subtasks"].setdefault(
                mat_name, ["disappear"] * subtasks["num_subtasks"]
            )
            subtasks["material_states_in_subtasks"][mat_name] = list(prefix_states) + list(mat_states)
        subtasks["num_subtasks"] = len(subtasks["subtasks"])
        subtasks["outbound_attached"] = True
    
    def apply_new_task_record_to_human_robot_machine_material(self, env_state_action_dict, task_record):
        #apply the new task record to the human, robot, and machine
        #machine
        machine_type = task_record["target_machine"]
        if machine_type != None:
            chosen_workstation_index = task_record["chosen_workstation_index"]
            ongoing_task_record_index = env_state_action_dict["machine"][machine_type]["ongoing_task_record_index"]
            assert ongoing_task_record_index[chosen_workstation_index] is None, "The ongoing task record should be empty"
            ongoing_task_record_index[chosen_workstation_index] = task_record["product_index"]
            env_state_action_dict["machine"][machine_type]["state"][chosen_workstation_index] = "working_" + task_record["task"]
        elif task_record["task"] == "logistic_for_paint_rust_proof":
            env_state_action_dict["machine"]["num07_gantry_group"]["state"][5] = "working_" + task_record["task"]
        else:
            raise ValueError(f"Invalid task: {task_record['task']}")
        #human
        human = task_record["human"]
        if human != None:
            assert env_state_action_dict["human"][human]["ongoing_task_record_index"] == None, "The ongoing task record should be empty"
            env_state_action_dict["human"][human]["ongoing_task_record_index"] = task_record["product_index"]
            env_state_action_dict["human"][human]["state"] = "working_" + task_record["task"]
            env_state_action_dict["human"][human]["generated_route"] = []
            env_state_action_dict["human"][human]["route_index"] = 0
            env_state_action_dict["human"][human]["route_length"] = 0
            env_state_action_dict["human"][human]["target_area_id"] = None
        #robot: logistic / processing 均在开工时按 action 定死并占用
        robot = task_record["robot"]
        if robot != None:
            assert env_state_action_dict["robot"][robot]["ongoing_task_record_index"] == None, "The ongoing task record should be empty"
            assert env_state_action_dict["robot"][robot]["state"] == "free", "Allocated robot must be free at task start"
            env_state_action_dict["robot"][robot]["ongoing_task_record_index"] = task_record["product_index"]
            env_state_action_dict["robot"][robot]["state"] = "working_" + task_record["task"]
        logistic_machine_type = task_record["logistic_machine"]
        if logistic_machine_type != None:
            if task_record["task_type"] == "logistic":
                assert logistic_machine_type == "num07_gantry_group", "now only num07_gantry_group is valid logistic machine"
                chosen_gantry_index = task_record["chosen_gantry_index"]
                env_state_action_dict["machine"][logistic_machine_type]["ongoing_task_record_index"][chosen_gantry_index] = task_record["product_index"]
                env_state_action_dict["machine"][logistic_machine_type]["state"][chosen_gantry_index] = "working_" + task_record["task"]
            elif task_record["task_type"] == "processing":
                pass #processing task dont need to update logistic machine now, will be updated after the during the processing task
            else:
                raise ValueError(f"Invalid task type: {task_record['task_type']}")
        #material
        material_type = task_record["product"]
        assert material_type != None, "The material type should not be none"
        material_name = f"num_{task_record['product_index']:02d}_{material_type}"
        assert env_state_action_dict["material"][material_name]["ongoing_task_record_index"] == None, "The ongoing task record should be empty"
        env_state_action_dict["material"][material_name]["ongoing_task_record_index"] = task_record["product_index"]        

    def step_task_records(self, env_state_action_dict):

        ongoing_task_records: dict = env_state_action_dict["progress"]["ongoing_task_records"]
        completed_task_records_indexs: list = []
        n_task_done = 0
        n_product_finished = 0

        for product_index, task_record in ongoing_task_records.items():
            product_type = task_record["product"]
            if self._check_subtask_and_task_done(env_state_action_dict, task_record):
                n_task_done += 1
                material_name = f"num_{product_index:02d}_{product_type}"
                finalize_material_batch_task_done(
                    env_state_action_dict["material"][material_name],
                    task_record,
                )
                if task_record["is_final_task"] == True:
                    # Remove this batch by product_index (not product_type — duplicates share the same type).
                    producing_list = env_state_action_dict["progress"]["producing"]
                    producing_indexs = env_state_action_dict["progress"]["producing_indexs"]
                    assert len(producing_list) == len(producing_indexs), (
                        "The length of producing list and producing indexs should be the same"
                    )
                    assert product_index in producing_indexs, (
                        f"Product index {product_index} not found in producing_indexs"
                    )
                    i = producing_indexs.index(product_index)
                    del producing_list[i]
                    del producing_indexs[i]
                    finished = env_state_action_dict["progress"]["finished"]
                    if product_type not in finished:
                        finished[product_type] = []
                    finished[product_type].append(product_index)
                    n_product_finished += 1
                completed_task_records_indexs.append(product_index)

        for task_record_index in completed_task_records_indexs:
            del ongoing_task_records[task_record_index]

        return {
            "n_task_done": n_task_done,
            "n_product_finished": n_product_finished,
        }

    def _check_subtask_and_task_done(self, env_state_action_dict, task_record):
        self._update_task_record_when_doing_subtask(env_state_action_dict, task_record)
        finished : list[bool] =  task_record["subtasks_dict"]["finished"]
        subtasks = task_record["subtasks_dict"]
        if all(finished) == True:
            if subtasks["ongoing_index"] == subtasks["num_subtasks"] - 1:
                # Processing: goal known but cross-zone AGV not ready → wait, never finish on prefix alone.
                if (
                    task_record["task_type"] == "processing"
                    and subtasks.get("goal_area_ids") is not None
                    and not subtasks.get("outbound_attached")
                ):
                    self._attach_processing_outbound(env_state_action_dict, task_record)
                    if not subtasks.get("outbound_attached"):
                        return False
                    # Outbound just attached: advance into first outbound row.
                    subtasks["ongoing_index"] += 1
                    subtasks["ongoing"] = subtasks["subtasks"][subtasks["ongoing_index"]]
                    ongoing = subtasks["ongoing"]
                    while len(finished) < len(ongoing):
                        finished.append(False)
                    del finished[len(ongoing):]
                    for sub_task_name, index in zip(ongoing, range(len(finished))):
                        finished[index] = sub_task_name in ("done", "none")
                    return False
                ## all subtasks are done
                task_record["task_done"] = True
                return True
            else:
                subtasks["ongoing_index"] += 1
                subtasks["ongoing"] = subtasks["subtasks"][subtasks["ongoing_index"]]
                ongoing = subtasks["ongoing"]
                # Pad/trim finished when switching 3-col prefix <-> 4-col cross outbound.
                while len(finished) < len(ongoing):
                    finished.append(False)
                del finished[len(ongoing):]
                for sub_task_name, index in zip(ongoing, range(len(finished))):
                    if sub_task_name == "done" or sub_task_name == "none":
                        finished[index] = True
                    else:
                        finished[index] = False
                return False

    def _update_task_record_when_doing_subtask(self, env_state_action_dict, task_record):
        subtasks = task_record["subtasks_dict"]
        ongoing = subtasks["ongoing"]
        finished = subtasks["finished"]

        # Robot freed early (subtask "done") → later wait/done/none must not block.
        if len(ongoing) > 3 and not finished[3] and ongoing[3] in ("wait", "done", "none"):
            robot_name = task_record.get("robot")
            unbound = robot_name is None
            if not unbound:
                r_idx = env_state_action_dict["robot"][robot_name].get("ongoing_task_record_index")
                unbound = r_idx != task_record["product_index"]
            if unbound:
                finished[3] = True

        # Mid-task gantry acquire/release (logistic cross-zone + processing).
        if len(ongoing) > 1:
            if ongoing[1] == "finding_free_gantry" and not finished[1]:
                if task_record["chosen_gantry_index"] is None:
                    task_record["chosen_gantry_index"] = self._find_free_gantry(env_state_action_dict, task_record)
                chosen_gantry_index = task_record["chosen_gantry_index"]
                if chosen_gantry_index is not None:
                    if task_record.get("task_start_time_step") is None:
                        task_record["task_start_time_step"] = int(env_state_action_dict["time_step"])
                    env_state_action_dict["machine"]["num07_gantry_group"]["ongoing_task_record_index"][chosen_gantry_index] = task_record["product_index"]
                    env_state_action_dict["machine"]["num07_gantry_group"]["state"][chosen_gantry_index] = "working_" + task_record["task"]
                    finished[1] = True
            elif (
                task_record.get("chosen_gantry_index") is None
                and ongoing[1] in ("wait", "none")
                and not finished[1]
            ):
                finished[1] = True

        if task_record["task_type"] != "processing":
            return

        # where the processed material will be put on
        if (
            subtasks["ongoing_index"] >= subtasks["index_to_decide_goal_area"]
            and subtasks["goal_area_ids"] is None
        ):
            if task_record["is_final_task"] == True:
                #no processing task after this task, so the processed material will be put on a storage
                goal_storage_name = find_free_storage(
                    env_state_action_dict["storage"],
                    task_record["processed_material"],
                )
                reserve_storage_slot(
                    env_state_action_dict["storage"][goal_storage_name],
                    task_record["processed_material"],
                    task_record["product_index"],
                )
                subtasks["goal_area_ids"] = env_state_action_dict["storage"][goal_storage_name]["key_variables"]["working_area_ids"]
                subtasks["material_goal_area"] = goal_storage_name
            else:
                next_target_machine = task_record["next_target_machine"]
                machine_state = env_state_action_dict["machine"][next_target_machine]["state"]
                if "free" in machine_state:
                    # the processed material will be put on the free workstation of the next target machine, no need to do logistic task for next processing task
                    task_record["already_done_next_logistic_task"] = True
                    task_record["next_chosen_workstation_index"] = machine_state.index("free")
                    machine_state[task_record["next_chosen_workstation_index"]] = "waiting_" + task_record["task"]
                    workstation_key = list(env_state_action_dict["machine"][next_target_machine]["key_variables"]["working_area_ids"].keys())[task_record["next_chosen_workstation_index"]]
                    task_record["next_chosen_machine_workstation"] = workstation_key
                    subtasks["goal_area_ids"] = env_state_action_dict["machine"][next_target_machine]["key_variables"]["working_area_ids"][workstation_key]
                    subtasks["material_goal_area"] = next_target_machine
                    subtasks["goal_area_workstation_key"] = workstation_key
                else:
                    task_record["already_done_next_logistic_task"] = False
                    goal_storage_name = find_free_storage(
                        env_state_action_dict["storage"],
                        task_record["processed_material"],
                    )
                    reserve_storage_slot(
                        env_state_action_dict["storage"][goal_storage_name],
                        task_record["processed_material"],
                        task_record["product_index"],
                    )
                    subtasks["goal_area_ids"] = env_state_action_dict["storage"][goal_storage_name]["key_variables"]["working_area_ids"]
                    subtasks["material_goal_area"] = goal_storage_name

        # Attach outbound as soon as goal is known (before prefix last step completes).
        self._attach_processing_outbound(env_state_action_dict, task_record)
