"""Opt-in R series: feasible serial dispatch and per-head decision-time replay.

Other heads are part of each head's changing behaviour policy. This is still
independent Q learning, not a proof of joint optimality or a centralized critic.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import torch

from .hc_factory_imports import import_hc_module
from .hier_buffer import Transition
from .hier_utils import detach_pre_to_cpu, index_to_one_hot
from .human_match import human_task_match_features
from .tpa_info_pool import TpaInfoPool

SCHEMA = "R-per-head-decisions-v1"
HEADS = ("A", "B", "C", "D_human", "D_robot")


def validate_config(config):
    unsupported = ("oru", "teacher_explore", "autoregressive", "curriculum", "explore",
                   "explore_catalog", "catalog_collect", "teacher_collect", "human_pair_head",
                   "task_pair_head", "task_match_head", "human_duration_aux", "noisy_net",
                   "hierarchical_credit")
    enabled = [key for key in unsupported if config.get(key, False)]
    if enabled:
        raise ValueError(f"R replay does not support these legacy extensions: {enabled}")
    if config.get("c_forbid_none_mode", "always") != "always":
        raise ValueError("R series dispatches feasible tasks; c_forbid_none_mode must be always")
    if config.get("decision_target_encoder", False) and float(config.get("target_tau", .005)) <= 0:
        raise ValueError("R target encoder requires a positive EMA target_tau")


class DecisionPool(TpaInfoPool):
    """Use the environment's record preparation/commit on private shadow resources."""

    def __init__(self, state, device):
        super().__init__(state, device)
        self._working = {**state, **self._working, "rl": {}}
        module = import_hc_module("src.task_progress_manager")
        self.manager = module.TaskManager(device)
        self.template = module.TaskRecordTemplate

    def record(self, slot, task, human, robot):
        rec = copy.deepcopy(self.template)
        if not self.manager.resolve_slot_to_product(self._working, slot, rec):
            return None
        if not self.manager.decode_process_task_planning_tensor(task, rec):
            return None
        self.manager.decode_human_robot_allocation_dict(
            {"human": human, "robot": robot}, self._working, rec)
        return rec if self.manager.prepare_new_task_record(self._working, rec) else None

    def feasible_masks(self):
        eligible = self.compute_b_eligible_mask()
        d = self.get_d_masks()
        rows = {}
        if not bool(d["human"].any()):
            return torch.zeros_like(eligible), rows
        h = index_to_one_hot(int(d["human"].nonzero()[0]), d["human"].numel(), self.device)
        r = torch.zeros_like(d["robot"])
        if bool(d["robot"].any()):
            r = index_to_one_hot(int(d["robot"].nonzero()[0]), r.numel(), self.device)
        for slot in eligible.nonzero().flatten().tolist():
            row = self.get_c_mask_for_slot(slot).clone()
            row[0] = 0  # same set for acting, replay and next-state targets
            for task_id in row.nonzero().flatten().tolist():
                task = index_to_one_hot(task_id, row.numel(), self.device)
                if self.record(slot, task, h, r) is None:
                    row[task_id] = 0
            eligible[slot] = int(bool(row.any()))
            rows[slot] = row
        return eligible, rows

    def commit(self, slot, record):
        # Exactly mirror accepted startup, including preferred-zone gantry and
        # the environment dropping unnecessary same-zone robot reservations.
        p = record["product_index"]
        self.progress["ongoing_task_records"][p] = record
        self.manager.apply_new_task_record_to_human_robot_machine_material(self._working, record)
        if record.get("from_staging_slot"):
            self.manager._commit_new_product_to_producing(self._working, record)
        self.served_slots.add(slot)
        self.dispatched_product_indices.add(p)
        self._refresh_masks()

    def observation(self, encoder, eligible, rows):
        pre = encoder.preprocess(self._working)
        aam = pre["agent_action_mask"]
        aam["agent_B_product_selector"] = eligible.clone()
        for slot, row in rows.items():
            aam["agent_C_process_task_planner"][slot] = row.clone()
        return pre


def event(head, pre, action, mask, context=None, dispatch=None):
    return dict(head=head, pre=copy.deepcopy(detach_pre_to_cpu(pre)), action=int(action.argmax()),
                mask=mask.detach().cpu().clone(),
                context=None if context is None else context.detach().cpu().clone(), dispatch=dispatch)


@torch.no_grad()
def build_decision_action(state, agents, epsilon):
    device, encoder = agents.cuda_device, agents.obs_encoder
    pool = DecisionPool(state, device)
    pre = encoder.preprocess(state)
    aam = pre["agent_action_mask"]
    bdim = aam["agent_B_product_selector"].numel()
    cdim = aam["agent_C_process_task_planner"].shape[1]
    bzero = torch.zeros(bdim, dtype=torch.int32, device=device)
    none = index_to_one_hot(0, cdim, device)
    # All heads must exist even if the first state has no feasible dispatch.
    agents.agent_A._ensure_dqn(pre)
    azero = torch.zeros_like(aam["agent_A_product_sequencer"])
    agents.agent_B._ensure_dqn(pre, azero)
    agents.agent_C._ensure_dqn(pre, bzero)
    agents.agent_D._ensure_dqn(pre, none)
    if agents.agent_D.human_match_head and not getattr(agents.agent_D, "_r_prior_initialized", False):
        # R2 learns candidate scoring from scratch; no fixed positive speed prior.
        agents.agent_D.human_dqn.q_net.prior_weight.zero_()
        agents.agent_D.human_dqn.target_net.prior_weight.zero_()
        agents.agent_D._r_prior_initialized = True
    sequence = agents.agent_A.act(pre, epsilon, pre=pre)
    trace = []
    if bool(sequence.any()):
        trace.append(event("A", pre, sequence, aam["agent_A_product_sequencer"]))
    pool.apply_product_sequencing(sequence)
    dispatches, order = [], []
    for _ in range(max(1, agents.max_parallel_cd_dispatch)):
        eligible, rows = pool.feasible_masks()
        if not bool(eligible.any()):
            break
        local = pool.observation(encoder, eligible, rows)
        # B is a sequential slot choice, not a whole ranking receiving duplicated credit.
        b = agents.agent_B.dqn.act_tensor(encoder.encode_B(local, sequence), eligible, epsilon)
        slot = int(b.argmax())
        c = agents.agent_C.act_with_mask(local, b, rows[slot], epsilon, pre=local)
        dmask = pool.get_d_masks()
        d = agents.agent_D.act_with_masks(local, c, dmask, epsilon, pre=local)
        record = pool.record(slot, c, d["human"], d["robot"])
        if record is None:
            raise RuntimeError("R feasibility changed inside a shadow dispatch")
        # Actual environment may discard an unnecessary robot on same-zone moves.
        if record.get("robot_index") is None:
            d["robot"] = torch.zeros_like(d["robot"])
        idx = len(dispatches)
        trace.extend([event("B", local, b, eligible, sequence, idx),
                      event("C", local, c, rows[slot], b, idx)])
        if agents.agent_D.human_policy == "rl":
            trace.append(event("D_human", local, d["human"], dmask["human"], c, idx))
        if bool(d["robot"].any()):
            trace.append(event("D_robot", local, d["robot"], dmask["robot"], c, idx))
        dispatches.append(dict(slot_index=slot, product_selection=b,
                               process_task_planning=c, human_robot_allocation=d))
        order.append(slot)
        pool.commit(slot, record)
    hzero = torch.zeros_like(aam["human"]["self_availability_mask"])
    rzero = torch.zeros_like(aam["robot"]["self_availability_mask"])
    first = dispatches[0] if dispatches else dict(product_selection=bzero,
        process_task_planning=none, human_robot_allocation={"human": hzero, "robot": rzero})
    return dict(product_sequencing=sequence, product_priority=agents.agent_B.scores_from_order(
        aam["agent_B_product_selector"], order, bdim, device), dispatch_list=dispatches,
        product_selection=first["product_selection"], process_task_planning=first["process_task_planning"],
        human_robot_allocation=first["human_robot_allocation"], _decision_trace=trace)


class DecisionReplay:
    """Each head accrues reward until its next executed decision, or termination.

    Successive decisions within a tick have dt=0. A rejected proposal is excluded
    from the head's action stream; rewards remain assigned to the prior pending
    action. Unexpected rejects are counted and should be investigated.
    """
    def __init__(self, owner):
        self.owner = owner
        self.pending = {}
        self.counts = Counter()
        self.dirty = False
        self.target_encoder = (copy.deepcopy(owner.obs_encoder).eval().requires_grad_(False)
                               if owner.config.get("decision_target_encoder", False) else None)

    def reset_target(self):
        if self.target_encoder is not None:
            self.target_encoder.load_state_dict(self.owner.obs_encoder.state_dict())

    def dqn(self, head):
        if head.startswith("D_"):
            return getattr(self.owner.agent_D, head[2:] + "_dqn")
        return getattr(self.owner, "agent_" + head).dqn

    def _close(self, old, nxt, done):
        head = old["head"]
        tr = Transition(action=old["action"], reward=old["reward"] * self.owner.decision_reward_scale,
                        mask=old["mask"], next_mask=torch.zeros_like(old["mask"]) if done else nxt["mask"],
                        done=done, discount=old["discount"], context=old["context"],
                        next_context=old["context"] if done else nxt["context"],
                        pre=old["pre"], next_pre=old["pre"] if done else nxt["pre"])
        self.dqn(head).buffer.push(tr)
        self.counts[f"transitions_{head}"] += 1
        if not done and old["context"] is not None and not torch.equal(old["context"], nxt["context"]):
            self.counts[f"context_changes_{head}"] += 1
        self.dirty = True

    def observe(self, env_id, action, rl, reward, done, *, restored=False):
        if restored:
            self.pending.pop(env_id, None)
            self.counts["restores"] += 1
            return
        pending = self.pending.setdefault(env_id, {})
        outcomes = {x["index"]: x for x in rl.get("dispatch_outcomes", [])}
        if action.get("dispatch_list") and "dispatch_outcomes" not in rl:
            raise RuntimeError("R training requires environment dispatch_feedback")
        self.counts["requested"] += len(action.get("dispatch_list", []))
        self.counts["accepted"] += sum(bool(x["accepted"]) for x in outcomes.values())
        for item in action.get("_decision_trace", []):
            head, idx = item["head"], item["dispatch"]
            if idx is not None:
                result = outcomes.get(idx, {})
                if not result.get("accepted", False):
                    continue
                if head.startswith("D_") and result.get(head[2:] + "_index") != item["action"]:
                    self.counts["execution_mismatches"] += 1
                    continue
            elif head == "A" and not action.get("dispatch_list"):
                self.counts["A_only"] += 1
            if head in pending:
                self._close(pending[head], item, False)
            pending[head] = {**item, "reward": 0., "discount": 1.}
            self.counts[f"decisions_{head}"] += 1
            self.counts[f"choices_{head}"] += int(item["mask"].sum() > 1)
        for old in pending.values():
            old["reward"] += old["discount"] * float(reward)
            old["discount"] *= self.owner.gamma
        if done:
            for old in pending.values():
                self._close(old, None, True)
            self.pending.pop(env_id, None)

    def encode(self, head, pre, transition, *, next_state=False, target=False):
        encoder = self.target_encoder if target and self.target_encoder is not None else self.owner.obs_encoder
        ctx = transition.next_context if next_state else transition.context
        if ctx is not None:
            ctx = ctx.to(self.owner.cuda_device)
        if head == "A":
            return encoder.encode_A(pre)
        if head == "B":
            return encoder.encode_B(pre, ctx)
        if head == "C":
            return encoder.encode_C(pre, ctx)
        base = encoder.encode_D(pre, ctx)
        if head == "D_human" and self.owner.agent_D.human_match_head:
            features = human_task_match_features(pre, ctx, self.dqn(head).action_dim)
            return torch.cat((base, features.flatten()))
        return base

    def learn(self):
        if not self.dirty:
            return False
        entries = []
        for head in HEADS:
            if head == "D_human" and self.owner.agent_D.human_policy != "rl":
                continue
            dqn = self.dqn(head)
            if dqn is None:
                continue
            loss = dqn.compute_loss(
                lambda p, t, h=head: self.encode(h, p, t),
                next_encode_fn=lambda p, t, h=head: self.encode(h, p, t, next_state=True),
                target_encode_fn=lambda p, t, h=head: self.encode(h, p, t, next_state=True, target=True))
            if loss is not None:
                entries.append((head, loss, dqn))
        losses = self.owner._joint_learn(entries)
        if not losses:
            return False
        for head, loss in losses.items():
            self.owner._loss_window[head].append(loss)
            self.counts[f"updates_{head}"] += 1
        if self.target_encoder is not None:
            tau = entries[0][2].target_tau  # includes the owner's late-stability schedule
            with torch.no_grad():
                for dst, src in zip(self.target_encoder.parameters(), self.owner.obs_encoder.parameters()):
                    dst.lerp_(src, tau)
        self.dirty = False
        return True

    def metrics(self):
        result = {f"MetricDecision/{k}": v for k, v in self.counts.items()}
        requested = self.counts["requested"]
        result["MetricDecision/accept_rate"] = self.counts["accepted"] / max(1, requested)
        result["MetricDecision/rejected"] = requested - self.counts["accepted"]
        return result

    def save(self, directory, step):
        p = Path(directory)
        metadata = {"schema": SCHEMA, "variant": str(self.owner.config.get("algo_variant", "R")),
                    "target_encoder": self.target_encoder is not None,
                    "human_match_head": self.owner.agent_D.human_match_head,
                    "human_policy": self.owner.agent_D.human_policy}
        # Written last: marks a complete R weight bundle, never a resumable optimizer/replay dump.
        (p / f"decision_schema_step_{step}.json").write_text(json.dumps(metadata, indent=2))


def check_checkpoint(owner, directory, step):
    path = Path(directory) / f"decision_schema_step_{step}.json"
    enabled = bool(getattr(owner, "decision_consistent", False))
    if path.exists() != enabled:
        raise RuntimeError("R and legacy G/E checkpoint semantics differ; use the matching series")
    if enabled:
        meta = json.loads(path.read_text())
        expected = dict(schema=SCHEMA, variant=str(owner.config.get("algo_variant", "R")),
                        target_encoder=bool(owner.config.get("decision_target_encoder", False)),
                        human_match_head=owner.agent_D.human_match_head, human_policy=owner.agent_D.human_policy)
        if meta != expected:
            raise RuntimeError(f"R checkpoint configuration mismatch: {meta} != {expected}")
        names = ("state_encoder", "agent_A", "agent_B", "agent_C", "agent_D_human", "agent_D_robot")
        if not all((Path(directory) / f"{name}_step_{step}.pth").is_file() for name in names):
            raise RuntimeError("Incomplete R checkpoint bundle")
