"""Leakage and provenance checks for the fixed diagnostic rate probe."""

import importlib.util
from pathlib import Path
import subprocess

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools/diagnose_upcoming_state_transfer.py"
spec = importlib.util.spec_from_file_location("state_transfer", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_physical_rules_match_pinned_main_including_boundaries():
    source = subprocess.check_output(["git", "show", module.MAIN_REFERENCE + ":" + module.MAIN_RULE_PATH])
    namespace = {}; exec(compile(source, "pinned_main_cause_cluster", "exec"), namespace)
    rng = np.random.default_rng(801)
    x = rng.uniform(0, 30, (31, 7, 27)).astype(np.float32)
    for channel, threshold in ((0, 1), (1, 20), (4, .8), (6, 24), (7, 0), (13, 20), (14, 20), (15, .25), (16, .7)):
        boundary = np.zeros((3, 7, 27), dtype=np.float32)
        boundary[:, :, channel] = np.array([np.nextafter(np.float32(threshold), np.float32(-np.inf)), threshold, np.nextafter(np.float32(threshold), np.float32(np.inf))])[:, None]
        x = np.concatenate((x, boundary))
    assert np.array_equal(module.physical_states(x), namespace["seed_cluster_ids"](x, window_size_s=60.))


def test_leave_episode_excludes_all_rows_and_prior():
    key = np.array([1, 1, 2, 1, 3]); ep = np.array([0, 0, 0, 1, 1]); y = np.array([1, 1, 1, 0, 1])
    # The entire first episode is excluded, including its different cell 2.
    score, n, p = module.rate_probe(key, y, ep, np.array([1, 2]), np.array([0, 0]))
    assert np.allclose(score, [.25, .5]) and n.tolist() == [1, 0] and p.tolist() == [0, 0]
    changed = y.copy(); changed[ep == 0] = 0
    score2, _, _ = module.rate_probe(key, changed, ep, np.array([1, 2]), np.array([0, 0]))
    assert np.array_equal(score, score2)


def test_validation_score_is_training_only_and_unseen_cell_uses_prior():
    score, n, p = module.rate_probe(np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 1]), np.array([0, 7]))
    assert np.allclose(score, [4/9, 1/3]) and n.tolist() == [2, 0] and p.tolist() == [1, 0]


def test_observation_padding_and_transition_identity():
    x = np.zeros((2, 3, 27), dtype=np.float32); mean = np.zeros(21, np.float32); std = np.ones(21, np.float32)
    mean[15] = .5  # An unobserved normalized zero must not become observed shortage.
    observation = np.ones((2, 3)); observation[1, 1] = 0
    state = module.observed_states(x, mean, std, observation)
    assert state.tolist() == [[2, 2, 2], [2, 7, 2]]
    node = np.arange(3)
    first = module.keys_for("node_state_transition", node, state[0], state[1])
    second = module.keys_for("node_state_transition", node, state[1], state[0])
    assert first[1] != second[1] and len(set(first.tolist())) == 3
