from __future__ import annotations

import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim

from .hier_buffer import ReplayBuffer, Transition
from .hier_networks import QNetwork
from .hier_utils import index_to_one_hot, masked_select_action, one_hot_to_index


class MaskedDQNAgent:
    """Single-agent masked DQN for discrete one-hot actions."""

    def __init__(
        self,
        name: str,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        target_update_interval: int = 500,
    ):
        self.name = name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.train_steps = 0

        self.q_net = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_net = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, obs: torch.Tensor, mask: torch.Tensor, epsilon: float) -> int | None:
        with torch.no_grad():
            q_values = self.q_net(obs.unsqueeze(0)).squeeze(0)
        return masked_select_action(q_values, mask, epsilon)

    def act_tensor(self, obs: torch.Tensor, mask: torch.Tensor, epsilon: float) -> torch.Tensor:
        action_idx = self.select_action(obs, mask, epsilon)
        if action_idx is None:
            # no valid action → all-zero one-hot (env decode treats sum==0 as skip)
            return torch.zeros(self.action_dim, dtype=torch.int32, device=self.device)
        return index_to_one_hot(action_idx, self.action_dim, self.device)

    def store(
        self,
        obs: torch.Tensor,
        action_idx: int,
        reward: float,
        next_obs: torch.Tensor,
        mask: torch.Tensor,
        next_mask: torch.Tensor,
        done: bool,
    ) -> None:
        """Legacy flat path: store detached obs tensors."""
        self.buffer.push(
            Transition(
                action=action_idx,
                reward=reward,
                mask=mask.detach().cpu(),
                next_mask=next_mask.detach().cpu(),
                done=done,
                obs=obs.detach().cpu(),
                next_obs=next_obs.detach().cpu(),
            )
        )

    def store_pre(
        self,
        pre: dict,
        action_idx: int,
        reward: float,
        next_pre: dict,
        mask: torch.Tensor,
        next_mask: torch.Tensor,
        done: bool,
    ) -> None:
        from .hier_utils import detach_pre_to_cpu

        self.buffer.push(
            Transition(
                action=action_idx,
                reward=reward,
                mask=mask.detach().cpu(),
                next_mask=next_mask.detach().cpu(),
                done=done,
                pre=detach_pre_to_cpu(pre),
                next_pre=detach_pre_to_cpu(next_pre),
            )
        )

    def _encode_batch(
        self,
        transitions: list[Transition],
        encode_fn: Callable[[dict], torch.Tensor] | None,
        *,
        use_next: bool,
    ) -> torch.Tensor:
        from .hier_utils import pre_to_device

        if transitions[0].pre is not None:
            assert encode_fn is not None
            encoded = []
            for transition in transitions:
                pre = pre_to_device(
                    transition.next_pre if use_next else transition.pre,
                    self.device,
                )
                encoded.append(encode_fn(pre))
            return torch.stack(encoded)

        obs_key = "next_obs" if use_next else "obs"
        return torch.stack([getattr(transition, obs_key) for transition in transitions]).to(self.device)

    def compute_loss(
        self,
        encode_fn: Callable[[dict], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        obs_batch = self._encode_batch(batch, encode_fn, use_next=False)
        action_batch = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        reward_batch = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_obs_batch = self._encode_batch(batch, encode_fn, use_next=True)
        mask_batch = torch.stack([t.mask for t in batch]).to(self.device)
        next_mask_batch = torch.stack([t.next_mask for t in batch]).to(self.device)
        done_batch = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_net(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_obs_batch)
            next_q[next_mask_batch == 0] = -float("inf")
            max_next_q = next_q.max(dim=1).values
            max_next_q[torch.isinf(max_next_q)] = 0.0
            target = reward_batch + self.gamma * max_next_q * (1.0 - done_batch)

        return nn.functional.mse_loss(q_values, target)

    def register_train_step(self) -> None:
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def step_optimizer(self) -> None:
        self.optimizer.step()

    def learn(self, encode_fn: Callable[[dict], torch.Tensor] | None = None) -> float | None:
        loss = self.compute_loss(encode_fn)
        if loss is None:
            return None
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.register_train_step()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"q_net": self.q_net.state_dict(), "name": self.name}, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=True)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(self.q_net.state_dict())


class RLProductSequencingAgent:
    """Agent A: product type sequencing — action dim = num_product_types."""

    AGENT_KEY = "agent_A_product_sequencer"
    ACTION_KEY = "product_sequencing"

    def __init__(self, obs_encoder, device: torch.device, **dqn_kwargs):
        self.obs_encoder = obs_encoder
        self.device = device
        self.dqn: MaskedDQNAgent | None = None
        self.dqn_kwargs = dqn_kwargs

    def _ensure_dqn(self, env_state_action_dict: dict) -> None:
        if self.dqn is not None:
            return
        obs_dim = self.obs_encoder.get_obs_dim_A(env_state_action_dict)
        action_dim = env_state_action_dict["agent_action_mask"][self.AGENT_KEY].shape[0]
        self.dqn = MaskedDQNAgent("agent_A", obs_dim, action_dim, self.device, **self.dqn_kwargs)

    def act(self, env_state_action_dict: dict, epsilon: float, *, pre: dict | None = None) -> torch.Tensor:
        self._ensure_dqn(env_state_action_dict)
        mask = env_state_action_dict["agent_action_mask"][self.AGENT_KEY]
        obs = self.obs_encoder.encode_A(env_state_action_dict, pre=pre)
        return self.dqn.act_tensor(obs, mask, epsilon)

    def observe_step(
        self, env_state_action_dict, action, reward, next_env_state_action_dict, done, epsilon, *, learn=True
    ):
        del epsilon
        if action.sum() == 0:
            return None
        self._ensure_dqn(env_state_action_dict)
        mask = env_state_action_dict["agent_action_mask"][self.AGENT_KEY].float()
        next_mask = next_env_state_action_dict["agent_action_mask"][self.AGENT_KEY].float()
        self.dqn.store_pre(
            env_state_action_dict,
            one_hot_to_index(action),
            reward,
            next_env_state_action_dict,
            mask,
            next_mask,
            done,
        )
        if not learn:
            return None
        return self.dqn.compute_loss(lambda pre: self.obs_encoder.encode_A(pre))


class RLProductSelectionAgent:
    """Agent B: RL priority ranking over eligible WIP / staging slots."""

    AGENT_KEY = "agent_B_product_selector"
    ACTION_KEY = "product_selection"

    def __init__(self, obs_encoder, device: torch.device, **dqn_kwargs):
        self.obs_encoder = obs_encoder
        self.device = device
        self.dqn: MaskedDQNAgent | None = None
        self.dqn_kwargs = dqn_kwargs
        from .agent_B_product_priority import ProductPriorityAgent

        self._priority = ProductPriorityAgent(device)

    def rank_slots(
        self,
        env_state_action_dict: dict,
        eligible_mask: torch.Tensor,
        epsilon: float,
        product_sequencing_action: torch.Tensor,
        *,
        pre: dict | None = None,
    ) -> list[int]:
        indices = (eligible_mask == 1).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return []

        if self.dqn is None:
            return self._priority.rank_slots(eligible_mask)

        self._ensure_dqn(env_state_action_dict, product_sequencing_action)
        if random.random() < epsilon:
            ordered = [int(i.item()) for i in indices]
            random.shuffle(ordered)
            staging = int(eligible_mask.shape[0] - 1)
            if staging in ordered:
                ordered.remove(staging)
                ordered.append(staging)
            return ordered

        mask = env_state_action_dict["agent_action_mask"][self.AGENT_KEY]
        with torch.no_grad():
            obs = self.obs_encoder.encode_B(env_state_action_dict, product_sequencing_action, pre=pre)
            q = self.dqn.q_net(obs.unsqueeze(0)).squeeze(0).clone()
            q[mask == 0] = -float("inf")

        staging = int(eligible_mask.shape[0] - 1)
        producing = [int(i.item()) for i in indices if int(i.item()) != staging]
        producing.sort(key=lambda slot: float(q[slot].item()), reverse=True)
        ordered = producing
        if eligible_mask[staging].item() == 1:
            ordered.append(staging)
        return ordered

    def scores_from_order(
        self,
        eligible_mask: torch.Tensor,
        slot_order: list[int],
        dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self._priority.scores_from_order(eligible_mask, slot_order, dim, device)

    def _ensure_dqn(self, env_state_action_dict: dict, product_sequencing_action: torch.Tensor) -> None:
        if self.dqn is not None:
            return
        obs_dim = self.obs_encoder.get_obs_dim_B(env_state_action_dict)
        action_dim = env_state_action_dict["agent_action_mask"][self.AGENT_KEY].shape[0]
        self.dqn = MaskedDQNAgent("agent_B", obs_dim, action_dim, self.device, **self.dqn_kwargs)

    def act(
        self,
        env_state_action_dict: dict,
        product_sequencing_action: torch.Tensor,
        epsilon: float,
        *,
        pre: dict | None = None,
    ) -> torch.Tensor:
        self._ensure_dqn(env_state_action_dict, product_sequencing_action)
        mask = env_state_action_dict["agent_action_mask"][self.AGENT_KEY]
        obs = self.obs_encoder.encode_B(env_state_action_dict, product_sequencing_action, pre=pre)
        return self.dqn.act_tensor(obs, mask, epsilon)

    def observe_step(
        self,
        env_state_action_dict,
        product_sequencing_action,
        action,
        reward,
        next_env_state_action_dict,
        done,
        epsilon,
        *,
        learn=True,
    ):
        del epsilon
        if action.sum() == 0:
            return None
        self._ensure_dqn(env_state_action_dict, product_sequencing_action)
        a_seq = product_sequencing_action
        mask = env_state_action_dict["agent_action_mask"][self.AGENT_KEY].float()
        next_mask = next_env_state_action_dict["agent_action_mask"][self.AGENT_KEY].float()
        self.dqn.store_pre(
            env_state_action_dict,
            one_hot_to_index(action),
            reward,
            next_env_state_action_dict,
            mask,
            next_mask,
            done,
        )
        if not learn:
            return None
        return self.dqn.compute_loss(lambda pre: self.obs_encoder.encode_B(pre, a_seq))


class RLProcessTaskPlanningAgent:
    """Agent C: plan next process/logistic task — action dim = len(CfgProcessTaskGalleryInAll)."""

    AGENT_KEY = "agent_C_process_task_planner"
    ACTION_KEY = "process_task_planning"

    def __init__(self, obs_encoder, device: torch.device, **dqn_kwargs):
        self.obs_encoder = obs_encoder
        self.device = device
        self.dqn: MaskedDQNAgent | None = None
        self.dqn_kwargs = dqn_kwargs

    def _get_task_mask(self, env_state_action_dict: dict, product_selection_action: torch.Tensor) -> torch.Tensor:
        task_mask_2d = env_state_action_dict["agent_action_mask"][self.AGENT_KEY]
        count = (product_selection_action == 1).sum().item()
        if count == 0:
            mask = torch.zeros(task_mask_2d.shape[1], dtype=torch.int32, device=self.device)
            mask[0] = 1
            return mask
        row_idx = int(product_selection_action.nonzero()[0][0].item())
        return task_mask_2d[row_idx]

    def _ensure_dqn(self, env_state_action_dict: dict, product_selection_action: torch.Tensor) -> None:
        if self.dqn is not None:
            return
        obs_dim = self.obs_encoder.get_obs_dim_C(env_state_action_dict, product_selection_action)
        action_dim = env_state_action_dict["agent_action_mask"][self.AGENT_KEY].shape[1]
        self.dqn = MaskedDQNAgent("agent_C", obs_dim, action_dim, self.device, **self.dqn_kwargs)

    def act(
        self,
        env_state_action_dict: dict,
        product_selection_action: torch.Tensor,
        epsilon: float,
        *,
        pre: dict | None = None,
    ) -> torch.Tensor:
        self._ensure_dqn(env_state_action_dict, product_selection_action)
        mask = self._get_task_mask(env_state_action_dict, product_selection_action)
        obs = self.obs_encoder.encode_C(env_state_action_dict, product_selection_action, pre=pre)
        return self.dqn.act_tensor(obs, mask, epsilon)

    def act_with_mask(
        self,
        env_state_action_dict: dict,
        product_selection_action: torch.Tensor,
        task_mask: torch.Tensor,
        epsilon: float,
        *,
        pre: dict | None = None,
    ) -> torch.Tensor:
        self._ensure_dqn(env_state_action_dict, product_selection_action)
        obs = self.obs_encoder.encode_C(env_state_action_dict, product_selection_action, pre=pre)
        return self.dqn.act_tensor(obs, task_mask, epsilon)

    def observe_step(
        self,
        env_state_action_dict,
        product_selection_action,
        action,
        reward,
        next_env_state_action_dict,
        done,
        epsilon,
        *,
        learn=True,
    ):
        del epsilon
        if action.sum() == 0:
            return None
        self._ensure_dqn(env_state_action_dict, product_selection_action)
        b_sel = product_selection_action
        mask = self._get_task_mask(env_state_action_dict, product_selection_action).float()
        next_mask = self._get_task_mask(next_env_state_action_dict, product_selection_action).float()
        self.dqn.store_pre(
            env_state_action_dict,
            one_hot_to_index(action),
            reward,
            next_env_state_action_dict,
            mask,
            next_mask,
            done,
        )
        if not learn:
            return None
        return self.dqn.compute_loss(lambda pre: self.obs_encoder.encode_C(pre, b_sel))


class RLHumanRobotAllocatorAgent:
    """Agent D: allocate human and robot — two masked DQN heads (human + robot)."""

    ACTION_KEY = "human_robot_allocation"

    def __init__(self, obs_encoder, device: torch.device, **dqn_kwargs):
        self.obs_encoder = obs_encoder
        self.device = device
        self.human_dqn: MaskedDQNAgent | None = None
        self.robot_dqn: MaskedDQNAgent | None = None
        self.dqn_kwargs = dqn_kwargs

    def _ensure_dqn(self, env_state_action_dict: dict, process_task_planning_action: torch.Tensor) -> None:
        if self.human_dqn is not None:
            return
        obs_dim = self.obs_encoder.get_obs_dim_D(env_state_action_dict, process_task_planning_action)
        human_dim = env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"].shape[0]
        robot_dim = env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"].shape[0]
        self.human_dqn = MaskedDQNAgent("agent_D_human", obs_dim, human_dim, self.device, **self.dqn_kwargs)
        self.robot_dqn = MaskedDQNAgent("agent_D_robot", obs_dim, robot_dim, self.device, **self.dqn_kwargs)

    def act(
        self,
        env_state_action_dict: dict,
        product_selection_action: torch.Tensor,
        process_task_planning_action: torch.Tensor,
        epsilon: float,
        *,
        pre: dict | None = None,
    ) -> dict:
        self._ensure_dqn(env_state_action_dict, process_task_planning_action)
        human_mask = env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"]
        robot_mask = env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"]
        return self._act_with_masks_impl(
            env_state_action_dict, process_task_planning_action, human_mask, robot_mask, epsilon, pre=pre
        )

    def act_with_masks(
        self,
        env_state_action_dict: dict,
        process_task_planning_action: torch.Tensor,
        d_masks: dict,
        epsilon: float,
        *,
        pre: dict | None = None,
    ) -> dict:
        self._ensure_dqn(env_state_action_dict, process_task_planning_action)
        return self._act_with_masks_impl(
            env_state_action_dict,
            process_task_planning_action,
            d_masks["human"],
            d_masks["robot"],
            epsilon,
            pre=pre,
        )

    def _act_with_masks_impl(
        self,
        env_state_action_dict: dict,
        process_task_planning_action: torch.Tensor,
        human_mask: torch.Tensor,
        robot_mask: torch.Tensor,
        epsilon: float,
        *,
        pre: dict | None = None,
    ) -> dict:
        if process_task_planning_action[0] == 1:
            return {
                "human": torch.zeros(human_mask.shape[0], dtype=torch.int32, device=self.device),
                "robot": torch.zeros(robot_mask.shape[0], dtype=torch.int32, device=self.device),
            }

        obs = self.obs_encoder.encode_D(env_state_action_dict, process_task_planning_action, pre=pre)
        return {
            "human": self.human_dqn.act_tensor(obs, human_mask, epsilon),
            "robot": self.robot_dqn.act_tensor(obs, robot_mask, epsilon),
        }

    def observe_step(
        self,
        env_state_action_dict,
        process_task_planning_action,
        action,
        reward,
        next_env_state_action_dict,
        done,
        epsilon,
        *,
        learn=True,
    ):
        del epsilon
        if process_task_planning_action[0] == 1:
            return None, None

        self._ensure_dqn(env_state_action_dict, process_task_planning_action)
        c_plan = process_task_planning_action
        encode_d = lambda pre: self.obs_encoder.encode_D(pre, c_plan)

        human_mask = env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"].float()
        robot_mask = env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"].float()
        next_human_mask = next_env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"].float()
        next_robot_mask = next_env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"].float()

        human_loss = None
        robot_loss = None
        if action["human"].sum() > 0:
            self.human_dqn.store_pre(
                env_state_action_dict,
                one_hot_to_index(action["human"]),
                reward,
                next_env_state_action_dict,
                human_mask,
                next_human_mask,
                done,
            )
            if learn:
                human_loss = self.human_dqn.compute_loss(encode_d)
        if action["robot"].sum() > 0:
            self.robot_dqn.store_pre(
                env_state_action_dict,
                one_hot_to_index(action["robot"]),
                reward,
                next_env_state_action_dict,
                robot_mask,
                next_robot_mask,
                done,
            )
            if learn:
                robot_loss = self.robot_dqn.compute_loss(encode_d)
        return human_loss, robot_loss
