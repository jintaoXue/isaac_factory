from __future__ import annotations

import copy
import random
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from .hier_buffer import PrioritizedReplayBuffer, ReplayBuffer, Transition
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
        *,
        double_dqn: bool = True,
        target_tau: float = 0.005,
        huber_delta: float = 1.0,
        reward_clip: float | None = 100.0,
        q_target_clip: float | None = 500.0,
        prioritized_replay: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 1_000_000,
        per_eps: float = 1e-6,
        dueling: bool = False,
        noisy: bool = False,
        noisy_std: float = 0.5,
    ):
        self.name = name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = max(1, int(target_update_interval))
        self.double_dqn = bool(double_dqn)
        self.target_tau = float(target_tau)
        self.huber_delta = float(huber_delta)
        self.reward_clip = None if reward_clip is None else float(reward_clip)
        self.q_target_clip = None if q_target_clip is None else float(q_target_clip)
        self.prioritized_replay = bool(prioritized_replay)
        self.dueling = bool(dueling)
        self.noisy = bool(noisy)
        self.train_steps = 0

        q_kwargs = dict(dueling=self.dueling, noisy=self.noisy, noisy_std=float(noisy_std))
        self.q_net = QNetwork(obs_dim, action_dim, hidden_dim, **q_kwargs).to(device)
        self.target_net = QNetwork(obs_dim, action_dim, hidden_dim, **q_kwargs).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        if self.prioritized_replay:
            self.buffer: ReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(
                buffer_capacity,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_frames=per_beta_frames,
                eps=per_eps,
            )
        else:
            self.buffer = ReplayBuffer(buffer_capacity)

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        self.q_net.reset_noise()
        self.target_net.reset_noise()

    def select_action(self, obs: torch.Tensor, mask: torch.Tensor, epsilon: float) -> int | None:
        # Noisy nets provide exploration; keep a small ε fallback only when not noisy.
        act_eps = 0.0 if self.noisy else float(epsilon)
        with torch.no_grad():
            if self.noisy and self.q_net.training:
                self.q_net.reset_noise()
            q_values = self.q_net(obs.unsqueeze(0)).squeeze(0)
        return masked_select_action(q_values, mask, act_eps)

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
        discount: float | None = None,
    ) -> None:
        """Legacy flat path: store detached obs tensors."""
        self.buffer.push(
            Transition(
                action=action_idx,
                reward=reward,
                mask=mask.detach().cpu(),
                next_mask=next_mask.detach().cpu(),
                done=done,
                discount=discount,
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
        discount: float | None = None,
        context: torch.Tensor | None = None,
    ) -> None:
        from .hier_utils import detach_pre_to_cpu

        self.buffer.push(
            Transition(
                action=action_idx,
                reward=reward,
                mask=mask.detach().cpu(),
                next_mask=next_mask.detach().cpu(),
                done=done,
                discount=discount,
                context=None if context is None else context.detach().cpu(),
                pre=detach_pre_to_cpu(pre),
                next_pre=detach_pre_to_cpu(next_pre),
            )
        )

    def _encode_batch(
        self,
        transitions: list[Transition],
        encode_fn: Callable[[dict, Transition], torch.Tensor] | None,
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
                encoded.append(encode_fn(pre, transition))
            return torch.stack(encoded)

        obs_key = "next_obs" if use_next else "obs"
        return torch.stack([getattr(transition, obs_key) for transition in transitions]).to(self.device)

    def _sample_train_batch(
        self,
        *,
        offline_buffer: ReplayBuffer | PrioritizedReplayBuffer | None = None,
        mix_ratio: float = 0.0,
    ) -> tuple[list[Transition], object | None, torch.Tensor | None] | None:
        """Sample ``batch_size`` transitions, mixing online + optional offline ORU buffer.

        Returns ``(batch, per_indices_or_None, is_weights_or_None)``.
        PER only applies to the **online** portion; offline ORU stays uniform.
        """
        mix_ratio = max(0.0, min(1.0, float(mix_ratio)))
        offline_n = 0
        if offline_buffer is not None and len(offline_buffer) > 0 and mix_ratio > 0.0:
            offline_n = min(len(offline_buffer), int(round(self.batch_size * mix_ratio)))
        online_n = self.batch_size - offline_n
        if online_n > len(self.buffer):
            short = online_n - len(self.buffer)
            online_n = len(self.buffer)
            if offline_buffer is not None:
                offline_n = min(len(offline_buffer), offline_n + short)
        total = online_n + offline_n
        if total <= 0:
            return None
        # Warmup / early online: allow slightly smaller batches once we have ≥1/4 batch.
        if total < self.batch_size and total < max(1, self.batch_size // 4):
            return None
        batch: list[Transition] = []
        per_indices = None
        is_weights: torch.Tensor | None = None
        if online_n > 0:
            if isinstance(self.buffer, PrioritizedReplayBuffer):
                online_batch, per_indices, weights = self.buffer.sample(online_n)
                batch.extend(online_batch)
                is_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            else:
                batch.extend(self.buffer.sample(online_n))
        if offline_n > 0 and offline_buffer is not None:
            # Offline catalog dumps use plain ReplayBuffer; sample uniformly.
            if isinstance(offline_buffer, PrioritizedReplayBuffer):
                off_batch, _, off_w = offline_buffer.sample(offline_n)
                batch.extend(off_batch)
                off_w_t = torch.as_tensor(off_w, dtype=torch.float32, device=self.device)
                if is_weights is None:
                    is_weights = off_w_t
                else:
                    is_weights = torch.cat([is_weights, off_w_t], dim=0)
            else:
                batch.extend(offline_buffer.sample(offline_n))
                if is_weights is not None:
                    # Online PER + offline uniform: pad offline weights with 1.0
                    ones = torch.ones(offline_n, dtype=torch.float32, device=self.device)
                    is_weights = torch.cat([is_weights, ones], dim=0)
        return batch, per_indices, is_weights

    def compute_loss(
        self,
        encode_fn: Callable[[dict, Transition], torch.Tensor] | None = None,
        *,
        offline_buffer: ReplayBuffer | PrioritizedReplayBuffer | None = None,
        mix_ratio: float = 0.0,
    ) -> torch.Tensor | None:
        sampled = self._sample_train_batch(offline_buffer=offline_buffer, mix_ratio=mix_ratio)
        if sampled is None:
            return None
        batch, per_indices, is_weights = sampled
        if self.noisy:
            self.reset_noise()
        obs_batch = self._encode_batch(batch, encode_fn, use_next=False)
        action_batch = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        reward_batch = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_obs_batch = self._encode_batch(batch, encode_fn, use_next=True)
        mask_batch = torch.stack([t.mask for t in batch]).to(self.device)
        next_mask_batch = torch.stack([t.next_mask for t in batch]).to(self.device)
        done_batch = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        discount_batch = torch.tensor(
            [self.gamma if t.discount is None else t.discount for t in batch],
            dtype=torch.float32,
            device=self.device,
        )

        q_values = self.q_net(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.q_net(next_obs_batch)
            next_q_online[next_mask_batch == 0] = -float("inf")
            if self.double_dqn:
                best_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.target_net(next_obs_batch)
                next_q_target[next_mask_batch == 0] = -float("inf")
                max_next_q = next_q_target.gather(1, best_actions).squeeze(1)
            else:
                next_q_target = self.target_net(next_obs_batch)
                next_q_target[next_mask_batch == 0] = -float("inf")
                max_next_q = next_q_target.max(dim=1).values
            max_next_q[torch.isinf(max_next_q)] = 0.0
            rewards = reward_batch
            if self.reward_clip is not None:
                rewards = rewards.clamp(-self.reward_clip, self.reward_clip)
            target = rewards + discount_batch * max_next_q * (1.0 - done_batch)
            if self.q_target_clip is not None:
                target = target.clamp(-self.q_target_clip, self.q_target_clip)

        td_error = target - q_values
        elementwise = nn.functional.smooth_l1_loss(
            q_values, target, beta=self.huber_delta, reduction="none"
        )
        if is_weights is not None and is_weights.numel() == elementwise.numel():
            loss = (is_weights * elementwise).mean()
        else:
            loss = elementwise.mean()

        if (
            per_indices is not None
            and isinstance(self.buffer, PrioritizedReplayBuffer)
            and is_weights is not None
        ):
            # Only update priorities for the online PER slice (leading online_n entries).
            online_n = len(per_indices)
            self.buffer.update_priorities(
                per_indices,
                td_error[:online_n].detach().abs().cpu().numpy(),
            )

        return loss

    def register_train_step(self) -> None:
        self.train_steps += 1
        if self.target_tau > 0.0:
            tau = self.target_tau
            with torch.no_grad():
                for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
        elif self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def step_optimizer(self) -> None:
        self.optimizer.step()

    def learn(self, encode_fn: Callable[[dict, Transition], torch.Tensor] | None = None) -> float | None:
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
        self, env_state_action_dict, action, reward, next_env_state_action_dict, done, epsilon, *, discount=None, learn=True,
        offline_buffer=None, mix_ratio: float = 0.0,
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
            discount=discount,
        )
        if not learn:
            return None
        return self.dqn.compute_loss(
            lambda pre, transition: self.obs_encoder.encode_A(pre),
            offline_buffer=offline_buffer,
            mix_ratio=mix_ratio,
        )


class RLProductSelectionAgent:
    """Agent B: RL priority ranking over eligible WIP / staging slots."""

    AGENT_KEY = "agent_B_product_selector"
    ACTION_KEY = "product_selection"

    def __init__(self, obs_encoder, device: torch.device, **dqn_kwargs):
        self.obs_encoder = obs_encoder
        self.device = device
        self.dqn: MaskedDQNAgent | None = None
        self.dqn_kwargs = dqn_kwargs
        # T1RH: when True, use lower ε on B ranking so Q-scores dominate earlier.
        self.b_score_rl = False
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
        rank_eps = float(epsilon) * (0.5 if self.b_score_rl else 1.0)
        if self.dqn is not None and getattr(self.dqn, "noisy", False):
            rank_eps = 0.0
        if random.random() < rank_eps:
            ordered = [int(i.item()) for i in indices]
            random.shuffle(ordered)
            staging = int(eligible_mask.shape[0] - 1)
            if staging in ordered:
                ordered.remove(staging)
                ordered.append(staging)
            return ordered

        mask = env_state_action_dict["agent_action_mask"][self.AGENT_KEY]
        with torch.no_grad():
            if self.dqn.noisy and self.dqn.q_net.training:
                self.dqn.q_net.reset_noise()
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
        discount=None,
        learn=True,
        offline_buffer=None,
        mix_ratio: float = 0.0,
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
            discount=discount,
            context=a_seq,
        )
        if not learn:
            return None
        return self.dqn.compute_loss(
            lambda pre, transition: self.obs_encoder.encode_B(pre, transition.context.to(self.device)),
            offline_buffer=offline_buffer,
            mix_ratio=mix_ratio,
        )


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
        discount=None,
        learn=True,
        offline_buffer=None,
        mix_ratio: float = 0.0,
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
            discount=discount,
            context=b_sel,
        )
        if not learn:
            return None
        return self.dqn.compute_loss(
            lambda pre, transition: self.obs_encoder.encode_C(pre, transition.context.to(self.device)),
            offline_buffer=offline_buffer,
            mix_ratio=mix_ratio,
        )


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
        discount=None,
        learn=True,
        offline_buffer_human=None,
        offline_buffer_robot=None,
        mix_ratio: float = 0.0,
    ):
        del epsilon
        if process_task_planning_action[0] == 1:
            return None, None

        self._ensure_dqn(env_state_action_dict, process_task_planning_action)
        c_plan = process_task_planning_action
        encode_d = lambda pre, transition: self.obs_encoder.encode_D(
            pre, transition.context.to(self.device)
        )

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
                discount=discount,
                context=c_plan,
            )
            if learn:
                human_loss = self.human_dqn.compute_loss(
                    encode_d, offline_buffer=offline_buffer_human, mix_ratio=mix_ratio
                )
        if action["robot"].sum() > 0:
            self.robot_dqn.store_pre(
                env_state_action_dict,
                one_hot_to_index(action["robot"]),
                reward,
                next_env_state_action_dict,
                robot_mask,
                next_robot_mask,
                done,
                discount=discount,
                context=c_plan,
            )
            if learn:
                robot_loss = self.robot_dqn.compute_loss(
                    encode_d, offline_buffer=offline_buffer_robot, mix_ratio=mix_ratio
                )
        return human_loss, robot_loss
