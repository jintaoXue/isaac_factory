# -*- coding: utf-8 -*-
"""HierarchicalTPA: env orchestration for hierarchical A→B→C→D agents.

Online: raw ``env_state_action_dict`` ↔ env.
Learning: one preprocess per env step (cached s_t); act reuses it; s_{t+1} becomes next step's cache.
``rl.reward/done`` are written by env ``TaskManager.update_rl_signals``.
"""
from __future__ import division

import copy
import os
import time
from collections import deque

import torch
import torch.optim as optim
import wandb
from rl_games.common import vecenv

from .hc_factory_imports import import_hc_module
from .hier_obs import HierObsEncoder
from .hier_rl_agents import (
    RLHumanRobotAllocatorAgent,
    RLProcessTaskPlanningAgent,
    RLProductSelectionAgent,
    RLProductSequencingAgent,
)
from .hier_utils import compute_team_reward, count_busy_agents, crossed_interval, env_steps, read_rl_done, steps_per_min
from .horizon_hooks import HorizonHooks
from .wandb_metrics import (
    HumanFatigueMonitor,
    LocalMetricsWriter,
    axis_payload,
    define_shared_metrics,
    episode_metrics,
    fullorder_core_metrics,
    fullorder_peak_metrics,
    log_metrics,
    shop_metrics,
    train_metrics,
)

_curr = import_hc_module("src.curriculum")


def _clear_rl(env_dict: dict) -> None:
    """Drop stale terminal flags after TPA has consumed them (env already reset)."""
    env_dict["rl"] = {
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


def _count_finished(env_dict: dict) -> int:
    fin = env_dict.get("progress", {}).get("finished", {})
    if not isinstance(fin, dict):
        return 0
    return sum(
        len(v) if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) else int(v or 0)
        for v in fin.values()
    )


def _segment_fields(
    progress: dict,
    spec,
    *,
    curriculum_enabled: bool,
    explore: bool,
    finished: int,
    episode_n_finished: int = 0,
    explore_n_target: int | None = None,
) -> tuple[int, int, int]:
    """Terminal start/target/remain for step logs."""
    if curriculum_enabled:
        start_n = int(progress.get("segment_start_nfin", spec.start_nfin))
        target_n = int(progress.get("segment_target_nfin", spec.target_nfin))
        n_done = int(episode_n_finished)
        finished_abs = start_n + n_done
        remain_n = max(0, target_n - finished_abs)
        return start_n, target_n, remain_n
    if explore:
        start_n = 0
        target_n = int(explore_n_target) if explore_n_target is not None else _curr.N_FULL_ORDER
        remain_n = max(0, target_n - int(finished))
        return start_n, target_n, remain_n
    start_n = 0
    order = progress.get("product_order") or {}
    if isinstance(order, dict) and order:
        target_n = sum(int(v or 0) for v in order.values())
    else:
        target_n = int(spec.n_products)
    remain_n = max(0, target_n - int(finished))
    return start_n, target_n, remain_n


def _mean_per_product_span_str(ep_t: int, *, episode_n_finished: int, finished: int) -> str:
    """makespan / n_finished; show n/a when nothing completed yet."""
    n_done = int(episode_n_finished)
    if n_done <= 0:
        n_done = int(finished)
    if n_done <= 0:
        return "n/a"
    return f"{float(ep_t + 1) / float(n_done):.1f}"


def _buffer_len(agent) -> int:
    dqn = getattr(agent, "dqn", None)
    if dqn is None or getattr(dqn, "buffer", None) is None:
        return 0
    return len(dqn.buffer)


class HierarchicalTPA:
    """Four-layer hierarchical decision stack (Masked DQN backend first; swappable later).

    Decision order (same as rule_based):
        A product sequencing → B product selection → C task planning → D allocation
    """

    def __init__(self, base_name, params):
        config = params["config"]
        self.config = config
        self.env_config = config.get("env_config", {})
        # Align with HcVectorEnvCfg.scene.num_envs (default 3)
        self.num_actors = int(config.get("num_actors", 3))
        self.env_name = config["env_name"]
        print("Env name:", self.env_name)

        self.env_info = config.get("env_info")
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()
        else:
            self.vec_env = None

        # Prefer live env count if wrapper exposes it
        if self.vec_env is not None:
            n = getattr(self.vec_env, "num_envs", None)
            if n is None and hasattr(self.vec_env, "env_list"):
                n = len(self.vec_env.env_list)
            if n is None and isinstance(self.env_info, dict):
                n = self.env_info.get("num_envs")
            if n is not None and int(n) != self.num_actors:
                print(
                    f"[Hier] warn: num_actors={self.num_actors} != env.num_envs={n}; "
                    f"using env.num_envs"
                )
                self.num_actors = int(n)

        self.cuda_device = self.env_info["cuda_device"]
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay_steps = config.get("epsilon_decay_steps", 100000)
        self.gamma = float(config.get("gamma", 0.99))
        self.decision_reward_scale = float(config.get("decision_reward_scale", 0.01))
        self.learn_interval = config.get("learn_interval", 1)
        self.save_interval = config.get("save_interval", 1000)
        self.log_interval = config.get("log_interval", 100)
        self.grad_clip_norm = float(config.get("grad_clip_norm", 10.0))
        self.global_step = 0
        self._late_stability_applied = False
        self.max_episodic_steps = int(config.get("max_episodic_steps", _curr.T_MAX_ANCHOR))

        dqn_kwargs = {
            "hidden_dim": config.get("hidden_dim", 128),
            "lr": config.get("learning_rate", 1e-4),
            "gamma": config.get("gamma", 0.99),
            "buffer_capacity": config.get("replay_buffer_size", 50000),
            "batch_size": config.get("batch_size", 64),
            "target_update_interval": config.get("target_update_interval", 500),
            "double_dqn": bool(config.get("double_dqn", True)),
            "target_tau": float(config.get("target_tau", 0.005)),
            "huber_delta": float(config.get("huber_delta", 1.0)),
            "reward_clip": config.get("reward_clip", 100.0),
            "q_target_clip": config.get("q_target_clip", 500.0),
        }

        parallel_limit = config.get("parallel_producing_limit", 10)
        self.max_parallel_cd_dispatch = int(config.get("max_parallel_cd_dispatch", 1))
        self.max_sim_episodes = config.get("max_sim_episodes")
        if self.max_sim_episodes is not None:
            self.max_sim_episodes = int(self.max_sim_episodes)
        self.episodes_done = 0
        self._train_t0 = None
        self.peak_producing = 0
        self.peak_ongoing = 0
        self.peak_ongoing_human = 0
        self.peak_ongoing_robot = 0
        self._ep_peak_producing: list[int] = []
        self._ep_peak_ongoing: list[int] = []
        self._ep_peak_ongoing_human: list[int] = []
        self._ep_peak_ongoing_robot: list[int] = []
        self._fatigue = HumanFatigueMonitor(num_envs=max(1, self.num_actors))
        self.obs_encoder = HierObsEncoder(
            self.cuda_device,
            parallel_producing_limit=parallel_limit,
            state_dim=int(config.get("state_dim", 256)),
        )
        encoder_lr = float(config.get("encoder_learning_rate", config.get("learning_rate", 1e-4)))
        self.encoder_optimizer = optim.Adam(self.obs_encoder.parameters(), lr=encoder_lr)

        dqn_kwargs_A = dict(dqn_kwargs)
        dqn_kwargs_A["batch_size"] = int(config.get("batch_size_A", 16))
        dqn_kwargs_A["buffer_capacity"] = int(config.get("replay_buffer_size_A", 5000))
        self.agent_A = RLProductSequencingAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs_A)
        self.agent_B = RLProductSelectionAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)
        self.agent_C = RLProcessTaskPlanningAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)
        self.agent_D = RLHumanRobotAllocatorAgent(self.obs_encoder, self.cuda_device, **dqn_kwargs)

        self.train_dir = config.get("train_dir", "runs")
        self.experiment_dir = os.path.join(self.train_dir, config["full_experiment_name"])
        self.nn_dir = os.path.join(self.experiment_dir, "nn")
        os.makedirs(self.nn_dir, exist_ok=True)
        self.local_metrics = LocalMetricsWriter(self.experiment_dir)
        print("Hierarchical experiment dir:", self.experiment_dir, "num_actors:", self.num_actors)

        self.use_wandb = config.get("wandb_activate", False)
        # DQN has no actor loss; rolling TD (critic) losses + makespan windows
        self._loss_window = {
            "A": deque(maxlen=200),
            "B": deque(maxlen=200),
            "C": deque(maxlen=200),
            "D_human": deque(maxlen=200),
            "D_robot": deque(maxlen=200),
        }
        self.makespan_success = deque(maxlen=int(config.get("makespan_window", 50)))
        self.makespan_all = deque(maxlen=int(config.get("makespan_window", 50)))
        self.success_hist = deque(maxlen=int(config.get("makespan_window", 50)))
        self.fullorder_makespan_success = deque(maxlen=int(config.get("makespan_window", 50)))
        self.fullorder_makespan_all = deque(maxlen=int(config.get("makespan_window", 50)))
        self.fullorder_success_hist = deque(maxlen=int(config.get("makespan_window", 50)))

        self.horizon = HorizonHooks(config)
        if self.horizon.explore:
            self.max_episodic_steps = self.horizon.explore_t_max()
            print(
                f"[Hier] explore mode: epsilon=1, no DQN backward, "
                f"N={self.horizon.explore_n_products} T_max={self.max_episodic_steps}"
            )
        elif not self.horizon.curriculum.enabled:
            # Hard train: same N/T as curriculum final stage (N_TRAIN_TARGET), not N16 anchor.
            anchor = int(config.get("t_max_anchor", _curr.T_MAX_ANCHOR))
            self.max_episodic_steps = _curr.t_max_for(_curr.N_TRAIN_TARGET, anchor)
            print(
                f"[Hier] hard train: N={_curr.N_TRAIN_TARGET} T_max={self.max_episodic_steps} "
                f"(no curriculum)"
            )

    def init_wandb_logger(self):
        define_shared_metrics(
            rl=True,
            curriculum=True,
            test=bool(self.config.get("test")),
        )

    def _log_metrics(self, payload: dict) -> None:
        log_metrics(payload, local=self.local_metrics, use_wandb=self.use_wandb)

    def _wall_time_sec(self) -> float:
        if self._train_t0 is None:
            return 0.0
        return float(time.time() - self._train_t0)

    def _update_concurrency_peaks(self, env_id: int, env_dict: dict) -> tuple[int, int, int, int]:
        progress = env_dict.get("progress") or {}
        n_producing = len(progress.get("producing") or [])
        n_ongoing = len(progress.get("ongoing_task_records") or {})
        n_human = count_busy_agents(env_dict.get("human"))
        n_robot = count_busy_agents(env_dict.get("robot"))
        while env_id >= len(self._ep_peak_producing):
            self._ep_peak_producing.append(0)
            self._ep_peak_ongoing.append(0)
            self._ep_peak_ongoing_human.append(0)
            self._ep_peak_ongoing_robot.append(0)
        self._ep_peak_producing[env_id] = max(self._ep_peak_producing[env_id], n_producing)
        self._ep_peak_ongoing[env_id] = max(self._ep_peak_ongoing[env_id], n_ongoing)
        self._ep_peak_ongoing_human[env_id] = max(self._ep_peak_ongoing_human[env_id], n_human)
        self._ep_peak_ongoing_robot[env_id] = max(self._ep_peak_ongoing_robot[env_id], n_robot)
        self.peak_producing = max(self.peak_producing, n_producing)
        self.peak_ongoing = max(self.peak_ongoing, n_ongoing)
        self.peak_ongoing_human = max(self.peak_ongoing_human, n_human)
        self.peak_ongoing_robot = max(self.peak_ongoing_robot, n_robot)
        return n_producing, n_ongoing, n_human, n_robot

    def get_epsilon(self) -> float:
        if getattr(self, "horizon", None) is not None and self.horizon.explore:
            return 1.0
        ratio = min(1.0, self.global_step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio
    def _all_dqns(self):
        return [
            self.agent_A.dqn,
            self.agent_B.dqn,
            self.agent_C.dqn,
            self.agent_D.human_dqn,
            self.agent_D.robot_dqn,
        ]

    def _maybe_apply_late_stability(self) -> None:
        """Lower LR and target-network tau once exploration reaches its floor."""
        threshold = int(self.config.get("late_stability_step", 0) or 0)
        if self._late_stability_applied or threshold <= 0 or self.global_step < threshold:
            return
        dqn_lr = float(self.config.get("late_learning_rate", 2.0e-5))
        encoder_lr = float(self.config.get("late_encoder_learning_rate", dqn_lr))
        target_tau = float(self.config.get("late_target_tau", 0.001))
        for group in self.encoder_optimizer.param_groups:
            group["lr"] = encoder_lr
        for dqn in self._all_dqns():
            if dqn is None:
                continue
            for group in dqn.optimizer.param_groups:
                group["lr"] = dqn_lr
            dqn.target_tau = target_tau
        self._late_stability_applied = True
        print(
            f"[Hier] late stability enabled at step={self.global_step}: "
            f"dqn_lr={dqn_lr:g} encoder_lr={encoder_lr:g} target_tau={target_tau:g}"
        )


    def restore(self, checkpoint_path: str) -> None:
        self._checkpoint_path = checkpoint_path

    def _maybe_load_checkpoint(self, env_state_action_dict: dict) -> None:
        ckpt = getattr(self, "_checkpoint_path", None) or self.config.get("load_dir")
        if not ckpt and not self.config.get("load_name"):
            return
        from .tpa_checkpoint import load_hier_checkpoint

        self.act_one_env(env_state_action_dict, epsilon=0.0)
        load_hier_checkpoint(self, self.config, getattr(self, "_checkpoint_path", None))

    def test(self) -> dict:
        assert self.vec_env is not None, "vec_env required for test()"
        seeds = list(self.config.get("test_seeds") or [int(self.config.get("seed", 42))])
        episodes_per_seed = int(self.config.get("test_times", 1))
        eval_epsilon = float(self.config.get("test_epsilon", 0.0))
        output_dir = os.path.join(self.experiment_dir, "eval")
        # Full-order eval: 16 products, T_max=anchor (not incremental curriculum segments).
        eval_horizon = int(self.config.get("t_max_anchor", _curr.T_MAX_ANCHOR))

        obs: list[dict] = self.vec_env.reset()
        self.horizon.bind(self.vec_env, len(obs))
        eval_horizon = self.horizon.apply_full_order_eval(eval_horizon)
        self._maybe_load_checkpoint(obs[0])

        from .tpa_eval import (
            EvalStream,
            build_eval_payload,
            print_eval_summary,
            run_eval_episodes,
            save_eval_results,
        )

        if self.use_wandb:
            self.init_wandb_logger()

        total_eps = len(seeds) * episodes_per_seed
        progress_iv = int(self.config.get("test_progress_log_interval", 500))
        stream = EvalStream(
            output_dir,
            t_budget=eval_horizon,
            algo_name="hier",
            total_episodes=total_eps,
            local=self.local_metrics,
            use_wandb=self.use_wandb,
        )

        print(
            f"[Hier] test full-order N={_curr.N_FULL_ORDER} T_max={eval_horizon} "
            f"eps={eval_epsilon} seeds={seeds} n={episodes_per_seed}"
        )
        results = run_eval_episodes(
            self.vec_env,
            lambda o, eps: self.act(o, epsilon=eps)[0:2],
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            max_episodic_steps=eval_horizon,
            epsilon=eval_epsilon,
            on_reset=lambda: self.horizon.apply_full_order_eval(eval_horizon),
            on_episode_done=stream.on_episode_done,
            on_progress=stream.on_progress,
            progress_log_interval=progress_iv,
        )
        payload = build_eval_payload(
            algo_name="hier",
            results=results,
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            epsilon=eval_epsilon,
            checkpoint=getattr(self, "_checkpoint_path", None),
        )
        json_path, summary_path = save_eval_results(output_dir, payload)
        print_eval_summary(payload["summary"], "hier")
        self.local_metrics.close()
        print(f"[Hier] eval saved: {json_path}\n[Hier] summary: {summary_path}")
        return payload

    def act_one_env(
        self, env_state_action_dict: dict, epsilon: float, pre: dict | None = None
    ) -> tuple[dict, dict]:
        from .hierarchical_dispatch import build_hier_rl_action

        action = build_hier_rl_action(
            env_state_action_dict,
            self.cuda_device,
            self,
            epsilon,
            max_parallel_cd_dispatch=self.max_parallel_cd_dispatch,
            pre=pre,
        )
        return action, {}

    def act(
        self,
        obs: list[dict],
        epsilon: float | None = None,
        prev_pre_list: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        eps = self.get_epsilon() if epsilon is None else epsilon
        actions: list[dict] = []
        actions_extra: list[dict] = []
        for i, env_state_action_dict in enumerate(obs):
            pre_i = prev_pre_list[i] if prev_pre_list is not None else None
            action, action_extra = self.act_one_env(env_state_action_dict, eps, pre=pre_i)
            actions.append(action)
            actions_extra.append(action_extra)
        return actions, actions_extra

    def _had_meaningful_decision(self, action: dict) -> bool:
        # Sequencing without a feasible dispatch does not change the factory.
        return bool(action.get("dispatch_list"))

    def _joint_learn(self, entries: list[tuple[str, torch.Tensor, object | None]]) -> dict[str, float]:
        """Average repeated K-dispatch losses per head, then update each optimizer once."""
        if not entries:
            return {}

        grouped: dict[str, list[torch.Tensor]] = {}
        dqn_by_name: dict[str, object] = {}
        for name, loss, dqn in entries:
            grouped.setdefault(name, []).append(loss)
            if dqn is not None:
                dqn_by_name[name] = dqn

        head_losses = {name: torch.stack(losses).mean() for name, losses in grouped.items()}
        for dqn in dqn_by_name.values():
            dqn.optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        total_loss = sum(head_losses.values())
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.obs_encoder.parameters(), self.grad_clip_norm)
        for dqn in dqn_by_name.values():
            torch.nn.utils.clip_grad_norm_(dqn.q_net.parameters(), self.grad_clip_norm)
        self.encoder_optimizer.step()
        for dqn in dqn_by_name.values():
            dqn.optimizer.step()
            dqn.register_train_step()

        return {name: float(loss.detach().item()) for name, loss in head_losses.items()}

    def observe_one_env(
        self,
        prev_obs: dict,
        action: dict,
        reward: float,
        next_obs: dict,
        done: bool,
        epsilon: float,
        *,
        bootstrap_discount: float | None = None,
        learn: bool = False,
    ) -> dict[str, float]:
        """Store one decision-interval transition, including every K dispatch."""
        losses: dict[str, float] = {}
        if not self._had_meaningful_decision(action):
            return losses

        explore = getattr(self, "horizon", None) is not None and self.horizon.explore
        should_learn = bool(learn) and not explore
        loss_entries: list[tuple[str, torch.Tensor, object | None]] = []

        loss_a = self.agent_A.observe_step(
            prev_obs,
            action["product_sequencing"],
            reward,
            next_obs,
            done,
            epsilon,
            discount=bootstrap_discount,
            learn=should_learn,
        )
        if loss_a is not None:
            loss_entries.append(("A", loss_a, self.agent_A.dqn))

        dispatches = action.get("dispatch_list") or []
        if not dispatches and action.get("product_selection") is not None:
            dispatches = [
                {
                    "product_selection": action["product_selection"],
                    "process_task_planning": action["process_task_planning"],
                    "human_robot_allocation": action["human_robot_allocation"],
                }
            ]

        for dispatch in dispatches:
            selection = dispatch["product_selection"]
            planning = dispatch["process_task_planning"]
            allocation = dispatch["human_robot_allocation"]

            loss_b = self.agent_B.observe_step(
                prev_obs,
                action["product_sequencing"],
                selection,
                reward,
                next_obs,
                done,
                epsilon,
                discount=bootstrap_discount,
                learn=should_learn,
            )
            if loss_b is not None:
                loss_entries.append(("B", loss_b, self.agent_B.dqn))

            loss_c = self.agent_C.observe_step(
                prev_obs,
                selection,
                planning,
                reward,
                next_obs,
                done,
                epsilon,
                discount=bootstrap_discount,
                learn=should_learn,
            )
            if loss_c is not None:
                loss_entries.append(("C", loss_c, self.agent_C.dqn))

            loss_d_h, loss_d_r = self.agent_D.observe_step(
                prev_obs,
                planning,
                allocation,
                reward,
                next_obs,
                done,
                epsilon,
                discount=bootstrap_discount,
                learn=should_learn,
            )
            if loss_d_h is not None:
                loss_entries.append(("D_human", loss_d_h, self.agent_D.human_dqn))
            if loss_d_r is not None:
                loss_entries.append(("D_robot", loss_d_r, self.agent_D.robot_dqn))

        if not should_learn:
            return losses

        losses = self._joint_learn(loss_entries)
        for key, value in losses.items():
            self._loss_window[key].append(value)
        return losses

    def save_checkpoint(self, step: int) -> None:
        """``step`` is summed env-instance steps (global_step × num_envs)."""
        enc_path = os.path.join(self.nn_dir, f"state_encoder_step_{step}.pth")
        import torch

        torch.save(self.obs_encoder.state_dict(), enc_path)

        for agent, name in [
            (self.agent_A, "agent_A"),
            (self.agent_B, "agent_B"),
            (self.agent_C, "agent_C"),
        ]:
            if agent.dqn is not None:
                agent.dqn.save(os.path.join(self.nn_dir, f"{name}_step_{step}.pth"))
        if self.agent_D.human_dqn is not None:
            self.agent_D.human_dqn.save(os.path.join(self.nn_dir, f"agent_D_human_step_{step}.pth"))
        if self.agent_D.robot_dqn is not None:
            self.agent_D.robot_dqn.save(os.path.join(self.nn_dir, f"agent_D_robot_step_{step}.pth"))

    def train(self):
        if self.config.get("test"):
            return self.test()
        assert self.vec_env is not None, "vec_env required for train()"
        if self.use_wandb:
            self.init_wandb_logger()
        obs: list[dict] = self.vec_env.reset()
        self.horizon.bind(self.vec_env, len(obs))
        episode_reward = [0.0 for _ in range(len(obs))]
        episode_len = [0 for _ in range(len(obs))]
        self._ep_peak_producing = [0 for _ in range(len(obs))]
        self._ep_peak_ongoing = [0 for _ in range(len(obs))]
        self._ep_peak_ongoing_human = [0 for _ in range(len(obs))]
        self._ep_peak_ongoing_robot = [0 for _ in range(len(obs))]
        # Completed products count within the current episode/segment.
        # Env may reset on done, so we accumulate per-step n_product_finished.
        episode_n_finished = [0 for _ in range(len(obs))]
        prev_pre_list = [self.obs_encoder.preprocess(o) for o in obs]
        pending_decisions = [None for _ in range(len(obs))]
        self._train_t0 = time.time()
        stop = False
        last_saved_env_steps = 0
        last_logged_env_steps = 0
        last_learned_env_steps = 0

        while not stop:
            epsilon = self.get_epsilon()
            actions, actions_extra = self.act(obs, prev_pre_list=prev_pre_list)
            for i, a in enumerate(actions):
                obs[i]["action"] = a

            # Close the previous decision interval at the current pre-action state,
            # then start a pending interval for each new meaningful decision.
            next_env_step = env_steps(self.global_step + 1, self.num_actors)
            do_learn = crossed_interval(last_learned_env_steps, next_env_step, self.learn_interval)
            learn_event = False
            for env_id, action in enumerate(actions):
                if not self._had_meaningful_decision(action):
                    continue
                old = pending_decisions[env_id]
                if old is not None:
                    self.observe_one_env(
                        old["pre"],
                        old["action"],
                        old["reward"] * self.decision_reward_scale,
                        prev_pre_list[env_id],
                        done=False,
                        epsilon=old["epsilon"],
                        bootstrap_discount=old["discount"],
                        learn=do_learn,
                    )
                    learn_event = True
                pending_decisions[env_id] = {
                    "pre": prev_pre_list[env_id],
                    "action": copy.deepcopy(action),
                    "reward": 0.0,
                    "discount": 1.0,
                    "epsilon": epsilon,
                }

            # Env writes rl and resets itself when done.
            next_obs = self.vec_env.step(actions, actions_extra)
            self.global_step += 1
            self._maybe_apply_late_stability()
            env_step = env_steps(self.global_step, self.num_actors)

            for env_id in range(len(obs)):
                self._update_concurrency_peaks(env_id, next_obs[env_id])
                self.horizon.on_decision(env_id, actions[env_id], next_obs[env_id])
                stall = self.horizon.after_step(env_id, next_obs[env_id])
                restored = stall in ("L2_RESTORE", "L3_RESTORE")
                reward = -0.05 if restored else compute_team_reward(obs[env_id], next_obs[env_id])
                done, truncated, success = read_rl_done(next_obs[env_id])
                if restored:
                    # Checkpoint jumps are not environment dynamics: discard the pending
                    # transition and restart credit assignment from the restored state.
                    pending_decisions[env_id] = None
                    done = False
                    truncated = False
                    success = False
                # When done, env has already reset next_obs; use pre-step obs for last fatigue frame.
                self._fatigue.update(env_id, obs[env_id] if done else next_obs[env_id])
                episode_reward[env_id] += reward
                episode_len[env_id] += 1
                rl_step = next_obs[env_id].get("rl") or {}
                if restored:
                    spec_now = self.horizon.curriculum.spec
                    episode_n_finished[env_id] = max(
                        0, _count_finished(next_obs[env_id]) - int(spec_now.start_nfin)
                    )
                else:
                    episode_n_finished[env_id] += int(rl_step.get("n_product_finished", 0) or 0)
                next_pre = self.obs_encoder.preprocess(next_obs[env_id])
                pending = pending_decisions[env_id]
                if pending is not None and not restored:
                    pending["reward"] += pending["discount"] * reward
                    pending["discount"] *= self.gamma
                    if done:
                        self.observe_one_env(
                            pending["pre"],
                            pending["action"],
                            pending["reward"] * self.decision_reward_scale,
                            next_pre,
                            done=True,
                            epsilon=pending["epsilon"],
                            bootstrap_discount=pending["discount"],
                            learn=do_learn,
                        )
                        learn_event = True
                        pending_decisions[env_id] = None
                prev_pre_list[env_id] = next_pre
                if done:
                    self.episodes_done += 1
                    # env already reset: episode_num in next_obs is the *new* episode index
                    completed_ep = int(next_obs[env_id].get("episode_num", 0) or 0)
                    del completed_ep
                    ep_len = episode_len[env_id]
                    spec = self.horizon.curriculum.spec
                    n_fin = int(episode_n_finished[env_id])
                    finished_abs = (
                        int(spec.start_nfin) + n_fin
                        if self.horizon.curriculum.enabled
                        else n_fin
                    )
                    t_budget = (
                        spec.t_max
                        if self.horizon.curriculum.enabled
                        else self.max_episodic_steps
                    )
                    wall = self._wall_time_sec()
                    is_fullorder = (not self.horizon.curriculum.enabled) or (
                        spec.start_nfin == 0 and spec.target_nfin == spec.n_products
                    )
                    if success:
                        self.makespan_success.append(ep_len)
                        self.makespan_all.append(ep_len)
                    elif truncated:
                        self.makespan_all.append(ep_len)
                    self.success_hist.append(float(success))
                    if is_fullorder:
                        if success:
                            self.fullorder_makespan_success.append(ep_len)
                            self.fullorder_makespan_all.append(ep_len)
                        elif truncated:
                            self.fullorder_makespan_all.append(ep_len)
                        self.fullorder_success_hist.append(float(success))
                    mean_ms = (
                        sum(self.makespan_all) / len(self.makespan_all) if self.makespan_all else None
                    )
                    mean_ms_ok = (
                        sum(self.makespan_success) / len(self.makespan_success)
                        if self.makespan_success
                        else None
                    )
                    success_rate = (
                        sum(self.success_hist) / len(self.success_hist) if self.success_hist else None
                    )
                    success_rate_str = f"{success_rate:.2f}" if success_rate is not None else "n/a"
                    nmk = float(ep_len) / float(max(1, t_budget))
                    mpps_str = _mean_per_product_span_str(
                        ep_len - 1,
                        episode_n_finished=n_fin,
                        finished=n_fin,
                    )
                    print(
                        f"[Hier] EP_DONE episode={self.episodes_done} "
                        f"len={ep_len} finished={finished_abs} ep_done_finished={n_fin} "
                        f"success_rate={success_rate_str} nmk={nmk:.3f} mpps={mpps_str} "
                        f"stalls={self.horizon.stall_counts_str()}"
                    )
                    human_ep = self._fatigue.on_episode_done(env_id, episode=self.episodes_done)
                    payload = axis_payload(
                        env_steps(self.global_step, self.num_actors), wall
                    )
                    core_payload = episode_metrics(
                        episode=self.episodes_done,
                        success=success,
                        truncated=truncated,
                        makespan=ep_len,
                        n_finished=n_fin,
                        finished_abs=finished_abs,
                        ep_return=episode_reward[env_id],
                        t_budget=t_budget,
                        success_rate=success_rate,
                        mean_makespan=mean_ms,
                        mean_makespan_success=mean_ms_ok,
                    )
                    peak_payload = shop_metrics(
                        peak_producing=self.peak_producing,
                        peak_ongoing=self.peak_ongoing,
                        peak_ongoing_human=self.peak_ongoing_human,
                        peak_ongoing_robot=self.peak_ongoing_robot,
                    )
                    payload.update(core_payload)
                    payload.update(peak_payload)
                    payload.update(human_ep)
                    if is_fullorder:
                        full_core_payload = dict(core_payload)
                        if self.fullorder_success_hist:
                            full_core_payload["MetricCore/03_success_rate"] = (
                                sum(self.fullorder_success_hist) / len(self.fullorder_success_hist)
                            )
                        if self.fullorder_makespan_all:
                            full_core_payload["MetricCore/09_mean_makespan"] = (
                                sum(self.fullorder_makespan_all) / len(self.fullorder_makespan_all)
                            )
                        if self.fullorder_makespan_success:
                            full_core_payload["MetricCore/10_mean_makespan_success"] = (
                                sum(self.fullorder_makespan_success) / len(self.fullorder_makespan_success)
                            )
                        payload.update(fullorder_core_metrics(full_core_payload))
                        payload.update(fullorder_peak_metrics(peak_payload))
                    if self.horizon.curriculum.enabled:
                        payload.update(
                            {
                                "Curriculum/01_stage": spec.stage,
                                "Curriculum/02_target_nfin": spec.target_nfin,
                                "Curriculum/03_start_nfin": spec.start_nfin,
                                "Curriculum/04_delta_n": spec.delta_n,
                                "Curriculum/05_t_budget": spec.t_max,
                            }
                        )
                    payload["Stagnation/resets_per_episode"] = float(
                        self.horizon.ep_stalled[env_id]
                    )
                    self._log_metrics(payload)
                    episode_reward[env_id] = 0.0
                    episode_len[env_id] = 0
                    self._ep_peak_producing[env_id] = 0
                    self._ep_peak_ongoing[env_id] = 0
                    self._ep_peak_ongoing_human[env_id] = 0
                    self._ep_peak_ongoing_robot[env_id] = 0
                    _clear_rl(next_obs[env_id])
                    self.horizon.on_episode_end(env_id, success=success, ep_len=ep_len)
                    prev_pre_list[env_id] = self.obs_encoder.preprocess(next_obs[env_id])
                    pending_decisions[env_id] = None
                    episode_n_finished[env_id] = 0
                    if (
                        self.max_sim_episodes is not None
                        and self.episodes_done >= self.max_sim_episodes
                    ):
                        stop = True

            if do_learn and learn_event:
                last_learned_env_steps = env_step

            obs = next_obs

            if crossed_interval(last_logged_env_steps, env_step, self.log_interval):
                last_logged_env_steps = env_step
                finished = _count_finished(next_obs[0])
                rl0 = next_obs[0].get("rl", {})
                progress0 = next_obs[0].get("progress") or {}
                producing0 = len(progress0.get("producing") or [])
                ongoing0 = len(progress0.get("ongoing_task_records") or {})
                ep0 = int(next_obs[0].get("episode_num", 0) or 0)
                t0 = int(next_obs[0].get("time_step", 0) or 0)
                wall = self._wall_time_sec()
                spm = steps_per_min(self.global_step, wall, self.num_actors)
                spm_str = f"{spm:.1f}" if spm is not None else "n/a"
                spec_now = self.horizon.curriculum.spec
                if self.horizon.explore:
                    self.horizon.ensure_explore_episode(0)
                    t_budget = self.horizon.explore_t_max()
                else:
                    t_budget = (
                        spec_now.t_max if self.horizon.curriculum.enabled else self.max_episodic_steps
                    )
                nmk = float(t0 + 1) / float(max(1, t_budget))
                mpps_str = _mean_per_product_span_str(
                    t0,
                    episode_n_finished=int(episode_n_finished[0]),
                    finished=finished,
                )
                start_n, target_n, remain_n = _segment_fields(
                    progress0,
                    spec_now,
                    curriculum_enabled=self.horizon.curriculum.enabled,
                    explore=self.horizon.explore,
                    finished=finished,
                    episode_n_finished=int(episode_n_finished[0]),
                    explore_n_target=(
                        self.horizon.explore_n_products if self.horizon.explore else None
                    ),
                )
                print(
                    f"[Hier] step={env_steps(self.global_step, self.num_actors)} episode={ep0} ep_t={t0} "
                    f"start={start_n} target={target_n} remain={remain_n} "
                    f"nmk={nmk:.3f} mpps={mpps_str} "
                    f"steps/min={spm_str} n_envs={len(obs)} "
                    f"stalls={self.horizon.stall_counts_str()}"
                )
                buf_d_h = (
                    len(self.agent_D.human_dqn.buffer)
                    if self.agent_D.human_dqn is not None
                    else 0
                )
                buf_d_r = (
                    len(self.agent_D.robot_dqn.buffer)
                    if self.agent_D.robot_dqn is not None
                    else 0
                )
                loss_payload = {}
                critic_vals = []
                for name, window in self._loss_window.items():
                    if window:
                        m = sum(window) / len(window)
                        loss_key = {
                            "A": "MetricLoss/02_critic_A",
                            "B": "MetricLoss/03_critic_B",
                            "C": "MetricLoss/04_critic_C",
                            "D_human": "MetricLoss/05_critic_D_human",
                            "D_robot": "MetricLoss/06_critic_D_robot",
                        }[name]
                        loss_payload[loss_key] = m
                        critic_vals.append(m)
                if critic_vals:
                    loss_payload["MetricLoss/01_critic_mean"] = sum(critic_vals) / len(critic_vals)
                payload = axis_payload(
                    env_steps(self.global_step, self.num_actors), wall, spm
                )
                peak_payload = shop_metrics(
                    producing=producing0,
                    ongoing=ongoing0,
                    peak_producing=self.peak_producing,
                    peak_ongoing=self.peak_ongoing,
                    peak_ongoing_human=self.peak_ongoing_human,
                    peak_ongoing_robot=self.peak_ongoing_robot,
                )
                payload.update(peak_payload)
                is_fullorder = (not self.horizon.curriculum.enabled) or (
                    spec_now.start_nfin == 0 and spec_now.target_nfin == spec_now.n_products
                )
                if is_fullorder:
                    payload.update(fullorder_peak_metrics(peak_payload))
                payload.update(self._fatigue.step_payload(0))
                payload.update(
                    train_metrics(
                        epsilon=epsilon,
                        buffer_A=_buffer_len(self.agent_A),
                        buffer_B=_buffer_len(self.agent_B),
                        buffer_C=_buffer_len(self.agent_C),
                        buffer_D_human=buf_d_h,
                        buffer_D_robot=buf_d_r,
                    )
                )
                payload.update(loss_payload)
                self._log_metrics(payload)

            if crossed_interval(last_saved_env_steps, env_step, self.save_interval):
                self.save_checkpoint(env_step)
                last_saved_env_steps = env_step
                print(f"[Hier] checkpoint saved at step {env_step}")

        env_step = env_steps(self.global_step, self.num_actors)
        if env_step > last_saved_env_steps:
            self.save_checkpoint(env_step)
            print(f"[Hier] checkpoint saved at step {env_step} (final)")

        wall = self._wall_time_sec()
        spm = steps_per_min(self.global_step, wall, self.num_actors)
        spm_str = f"{spm:.1f}" if spm is not None else "n/a"
        print(
            f"[Hier] train finished episodes_done={self.episodes_done} "
            f"steps={env_steps(self.global_step, self.num_actors)} steps/min={spm_str} "
            f"peak_prod={self.peak_producing} peak_ong={self.peak_ongoing} "
            f"peak_human={self.peak_ongoing_human} peak_robot={self.peak_ongoing_robot} "
            f"stalls={self.horizon.stall_counts_str()}"
        )
        finish_payload = axis_payload(
            env_steps(self.global_step, self.num_actors), wall, spm
        )
        self._log_metrics(finish_payload)
        self.local_metrics.close()
        if self.use_wandb:
            wandb.finish()
