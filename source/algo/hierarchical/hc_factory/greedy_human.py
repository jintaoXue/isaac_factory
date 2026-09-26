"""Fixed instantaneous eta × task-skill baseline (not a duration oracle)."""
import torch
from .human_match import _skill_eff_clips


@torch.no_grad()
def greedy_human_action(pre: dict, task: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Use the dispatch pool's current mask, not the stale observation availability.

    Fatigue enters through current efficiency eta. Subtask skill, travel, future
    fatigue and downstream contention are deliberately absent from this proxy.
    Ties go to the lowest index, with no extra random-number consumption.
    """
    action = torch.zeros_like(mask, dtype=torch.int32)
    if task[0] == 1 or not bool(mask.any()):
        return action
    human = pre["human"]
    n = mask.numel()
    skill = human["skill_task"][:n].to(mask.device).float() @ task[1:].to(mask.device).float()
    lo, hi = _skill_eff_clips()
    score = human["efficiency"][:n].to(mask.device).flatten() * skill.clamp(lo, hi)
    valid = mask.bool() & human["mask"][:n].to(mask.device).flatten().bool()
    if not bool(valid.any()):
        return action
    if not bool(torch.isfinite(score[valid]).all()):
        raise ValueError("Non-finite human efficiency/skill in greedy evaluation")
    action[score.masked_fill(~valid, -torch.inf).argmax()] = 1
    return action
