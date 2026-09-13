"""Whole-episode exclusion and reference-identity tests, without files/directories."""

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "isaaclab_tasks/direct/hc_factory/tools/diagnose_frozen_representation_transfer.py"
spec = importlib.util.spec_from_file_location("representation_transfer", PATH)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_centroid_has_training_scale_and_constant_feature_is_finite():
    x = np.array([[0., 2.], [1., 2.], [4., 2.], [5., 2.]])
    w, b = module.centroid_fit(x, np.array([0, 0, 1, 1]))
    assert np.allclose(w, [4 / 4.25, 0]) and np.isclose(b, -2.5 * w[0])
    scores = module.centroid_scores(x, np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.array([[0., 200.], [5., -200.]]))
    assert np.isfinite(scores).all() and scores[0] < 0 < scores[1]


def test_episode_exclusion_removes_labels_and_standardization_statistics():
    x = np.array([[0.], [10.], [1.], [11.]])
    y, ep, query = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), np.array([[0.], [10.]])
    result = module.centroid_scores(x, y, ep, query, np.array([0, 0]))
    assert np.allclose(result, [-2.4, 1.6])
    changed_x, changed_y = x.copy(), y.copy(); changed_x[:2] = 100000; changed_y[:2] = 1
    assert np.array_equal(result, module.centroid_scores(changed_x, changed_y, ep, query, np.array([0, 0])))


def test_neighbors_exclude_entire_episode_and_identify_near_overlap():
    reference = np.array([[0.], [.1], [.2], [.4]])
    labels, episodes, ids = np.array([1, 1, 0, 1]), np.array([0, 0, 1, 1]), np.array([10, 11, 20, 21])
    result = module.nearest_for_query(reference, np.array([0.]), labels, episodes, ids, 0, 10, True)
    assert result["nearest_training_nonself"]["sample_index"] == 11
    assert result["nearest_eligible_positive"]["sample_index"] == 21
    assert result["nearest_eligible_negative"]["sample_index"] == 20
    val = module.nearest_for_query(reference, np.array([0.]), labels, episodes, ids, 2, 30, False)
    assert val["nearest_eligible_positive"]["sample_index"] == 10


def test_missing_positive_reference_is_explicit():
    result = module.nearest_for_query(np.array([[0.], [1.]]), np.array([0.]), np.array([1, 0]), np.array([0, 1]), np.array([3, 4]), 0, 3, True)
    assert result["nearest_eligible_positive"] is None
    assert result["nearest_eligible_negative"]["sample_index"] == 4
