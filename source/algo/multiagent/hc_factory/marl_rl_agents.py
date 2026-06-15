from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from .marl_buffer import ReplayBuffer, Transition
from .marl_networks import QNetwork
from .marl_utils import index_to_one_hot, masked_select_action, one_hot_to_index


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

    def select_action(self, obs: torch.Tensor, mask: torch.Tensor, epsilon: float) -> int:
        with torch.no_grad():
            q_values = self.q_net(obs.unsqueeze(0)).squeeze(0)
        return masked_select_action(q_values, mask, epsilon)

    def act_tensor(self, obs: torch.Tensor, mask: torch.Tensor, epsilon: float) -> torch.Tensor:
        action_idx = self.select_action(obs, mask, epsilon)
        return index_to_one_hot(action_idx, self.action_dim, self.device)

    def store(self, obs, action_idx, reward, next_obs, mask, next_mask, done) -> None:
        self.buffer.push(
            Transition(obs=obs, action=action_idx, reward=reward, next_obs=next_obs, mask=mask, next_mask=next_mask, done=done)
        )

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        obs_batch = torch.stack([t.obs for t in batch])
        action_batch = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        reward_batch = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_obs_batch = torch.stack([t.next_obs for t in batch])
        mask_batch = torch.stack([t.mask for t in batch])
        next_mask_batch = torch.stack([t.next_mask for t in batch])
        done_batch = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_net(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_obs_batch)
            next_q[next_mask_batch == 0] = -float("inf")
            max_next_q = next_q.max(dim=1).values
            max_next_q[torch.isinf(max_next_q)] = 0.0
            target = reward_batch + self.gamma * max_next_q * (1.0 - done_batch)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
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

    def act(self, env_state_action_dict: dict, epsilon: float) -> torch.Tensor:
        self._ensure_dqn(env_state_action_dict)
        mask = env_state_action_dict["agent_action_mask"][self.AGENT_KEY]
        obs = self.obs_encoder.encode_A(env_state_action_dict)
        return self.dqn.act_tensor(obs, mask, epsilon)

    def observe_step(self, env_state_action_dict, action, reward, next_env_state_action_dict, done, epsilon):
        self._ensure_dqn(env_state_action_dict)
        obs = self.obs_encoder.encode_A(env_state_action_dict)
        next_obs = self.obs_encoder.encode_A(next_env_state_action_dict)
        mask = env_state_action_dict["agent_action_mask"][self.AGENT_KEY].float()
        next_mask = next_env_state_action_dict["agent_action_mask"][self.AGENT_KEY].float()
        self.dqn.store(obs, one_hot_to_index(action), reward, next_obs, mask, next_mask, done)
        return self.dqn.learn()


class RLProductSelectionAgent:
    """Agent B: select focal product from producing list — action dim = parallel_producing_limit + 1."""

    AGENT_KEY = "agent_B_product_selector"
    ACTION_KEY = "product_selection"

    def __init__(self, obs_encoder, device: torch.device, **dqn_kwargs):
        self.obs_encoder = obs_encoder
        self.device = device
        self.dqn: MaskedDQNAgent | None = None
        self.dqn_kwargs = dqn_kwargs

    def _ensure_dqn(self, env_state_action_dict: dict, product_sequencing_action: torch.Tensor) -> None:
        if self.dqn is not None:
            return
        obs_dim = self.obs_encoder.get_obs_dim_B(env_state_action_dict)
        action_dim = env_state_action_dict["agent_action_mask"][self.AGENT_KEY].shape[0]
        self.dqn = MaskedDQNAgent("agent_B", obs_dim, action_dim, self.device, **self.dqn_kwargs)

    def act(self, env_state_action_dict: dict, product_sequencing_action: torch.Tensor, epsilon: float) -> torch.Tensor:
        self._ensure_dqn(env_state_action_dict, product_sequencing_action)
        mask = env_state_action_dict["agent_action_mask"][self.AGENT_KEY]
        obs = self.obs_encoder.encode_B(env_state_action_dict, product_sequencing_action)
        return self.dqn.act_tensor(obs, mask, epsilon)

    def observe_step(self, env_state_action_dict, product_sequencing_action, action, reward, next_env_state_action_dict, done, epsilon):
        self._ensure_dqn(env_state_action_dict, product_sequencing_action)
        obs = self.obs_encoder.encode_B(env_state_action_dict, product_sequencing_action)
        next_obs = self.obs_encoder.encode_B(next_env_state_action_dict, product_sequencing_action)
        mask = env_state_action_dict["agent_action_mask"][self.AGENT_KEY].float()
        next_mask = next_env_state_action_dict["agent_action_mask"][self.AGENT_KEY].float()
        self.dqn.store(obs, one_hot_to_index(action), reward, next_obs, mask, next_mask, done)
        return self.dqn.learn()


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

    def act(self, env_state_action_dict: dict, product_selection_action: torch.Tensor, epsilon: float) -> torch.Tensor:
        self._ensure_dqn(env_state_action_dict, product_selection_action)
        mask = self._get_task_mask(env_state_action_dict, product_selection_action)
        obs = self.obs_encoder.encode_C(env_state_action_dict, product_selection_action)
        return self.dqn.act_tensor(obs, mask, epsilon)

    def observe_step(self, env_state_action_dict, product_selection_action, action, reward, next_env_state_action_dict, done, epsilon):
        self._ensure_dqn(env_state_action_dict, product_selection_action)
        obs = self.obs_encoder.encode_C(env_state_action_dict, product_selection_action)
        next_obs = self.obs_encoder.encode_C(next_env_state_action_dict, product_selection_action)
        mask = self._get_task_mask(env_state_action_dict, product_selection_action).float()
        next_mask = self._get_task_mask(next_env_state_action_dict, product_selection_action).float()
        self.dqn.store(obs, one_hot_to_index(action), reward, next_obs, mask, next_mask, done)
        return self.dqn.learn()


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
    ) -> dict:
        self._ensure_dqn(env_state_action_dict, process_task_planning_action)
        human_mask = env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"]
        robot_mask = env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"]

        if process_task_planning_action[0] == 1:
            return {
                "human": torch.zeros(human_mask.shape[0], dtype=torch.int32, device=self.device),
                "robot": torch.zeros(robot_mask.shape[0], dtype=torch.int32, device=self.device),
            }

        obs = self.obs_encoder.encode_D(env_state_action_dict, process_task_planning_action)
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
    ):
        if process_task_planning_action[0] == 1:
            return None, None

        self._ensure_dqn(env_state_action_dict, process_task_planning_action)
        obs = self.obs_encoder.encode_D(env_state_action_dict, process_task_planning_action)
        next_obs = self.obs_encoder.encode_D(next_env_state_action_dict, process_task_planning_action)

        human_mask = env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"].float()
        robot_mask = env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"].float()
        next_human_mask = next_env_state_action_dict["agent_action_mask"]["human"]["self_availability_mask"].float()
        next_robot_mask = next_env_state_action_dict["agent_action_mask"]["robot"]["self_availability_mask"].float()

        human_loss = None
        robot_loss = None
        if action["human"].sum() > 0:
            self.human_dqn.store(
                obs, one_hot_to_index(action["human"]), reward, next_obs, human_mask, next_human_mask, done
            )
            human_loss = self.human_dqn.learn()
        if action["robot"].sum() > 0:
            self.robot_dqn.store(
                obs, one_hot_to_index(action["robot"]), reward, next_obs, robot_mask, next_robot_mask, done
            )
            robot_loss = self.robot_dqn.learn()
        return human_loss, robot_loss
