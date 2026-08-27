"""Remaining-jobs occupancy packing (no GPU)."""

from __future__ import annotations

import numpy as np

from factory_bn.dataset import _build_samples
from factory_bn.remain import (
    ensure_labor_saturated_feature,
    first_done_index,
    jobs_remaining_series,
    labor_saturated_mask,
    node_hot_mask,
    occupancy_node_mask,
    occupancy_to_events,
    pack_remain_target,
    smooth_occupancy_runs,
)


def test_node_hot_includes_score() -> None:
    t_len, n, f = 3, 2, 22
    feats = np.zeros((t_len, n, f), dtype=np.float32)
    scores = np.zeros((t_len, n, 1), dtype=np.float32)
    scores[1, 0, 0] = 0.7
    feats[2, 1, 19] = 1.0  # turning point
    feats[0, 0, 18] = 1.0  # L2 context must not mint occupancy y
    hot = node_hot_mask(feats, scores, score_threshold=0.55, min_hot_windows=1, gap_windows=0)
    assert hot[1, 0] == 1.0
    assert hot[2, 1] == 1.0
    assert hot[0].sum() == 0.0

    feats[0, 1, 15] = 0.8  # shortage
    feats[0, 1, 7] = 40.0  # starved 40/60s
    hot2 = node_hot_mask(
        feats, scores, score_threshold=0.55, window_size_s=60.0, min_hot_windows=1, gap_windows=0
    )
    assert hot2[0, 1] == 1.0
    feats[0, 0, 0] = 2.0  # queue
    feats[0, 0, 6] = 10.0  # blocked
    hot3 = node_hot_mask(
        feats, scores, score_threshold=0.55, window_size_s=60.0, min_hot_windows=1, gap_windows=0
    )
    assert hot3[0, 0] == 1.0

    # Gantry with route delay + stall is process occupancy; starve alone is not.
    gantry = np.zeros((1, 1, 26), dtype=np.float32)
    gantry[0, 0, 7] = 40.0
    gantry[0, 0, 13] = 40.0
    gantry[0, 0, 22] = 1.0  # type = gantry
    g_scores = np.zeros((1, 1, 1), dtype=np.float32)
    assert node_hot_mask(gantry, g_scores, window_size_s=60.0, min_hot_windows=1, gap_windows=0)[0, 0] == 1.0
    gantry_starve = np.zeros((1, 1, 26), dtype=np.float32)
    gantry_starve[0, 0, 7] = 40.0
    gantry_starve[0, 0, 22] = 1.0
    assert node_hot_mask(
        gantry_starve, g_scores, window_size_s=60.0, min_hot_windows=1, gap_windows=0
    )[0, 0] == 0.0

    down = np.zeros((1, 1, 26), dtype=np.float32)
    down[0, 0, 9] = 1.0
    down[0, 0, 18] = 1.0
    down[0, 0, 21] = 1.0  # type = machine
    assert node_hot_mask(down, g_scores, window_size_s=60.0, min_hot_windows=1, gap_windows=0)[0, 0] == 1.0

    agv_stop = np.zeros((2, 1, 26), dtype=np.float32)
    agv_stop[:, 0, 9] = 1.0
    agv_stop[:, 0, 24] = 1.0  # type = AGV
    assert node_hot_mask(agv_stop, np.zeros((2, 1, 1), dtype=np.float32), min_hot_windows=2)[:, 0].sum() == 2.0

    # Machine stall while a human node is STOP is occupancy; stall alone is not.
    wait = np.zeros((2, 2, 26), dtype=np.float32)
    wait[:, 0, 7] = 40.0  # machine starved
    wait[:, 0, 21] = 1.0
    wait[:, 1, 9] = 1.0  # human STOP
    wait[:, 1, 23] = 1.0
    w_hot = node_hot_mask(
        wait, np.zeros((2, 2, 1), dtype=np.float32), window_size_s=60.0, min_hot_windows=2
    )
    assert w_hot[:, 0].sum() == 2.0
    assert w_hot[:, 1].sum() == 0.0
    starve_only = np.zeros((2, 1, 26), dtype=np.float32)
    starve_only[:, 0, 7] = 40.0
    starve_only[:, 0, 21] = 1.0
    assert (
        node_hot_mask(
            starve_only,
            np.zeros((2, 1, 1), dtype=np.float32),
            window_size_s=60.0,
            min_hot_windows=2,
        )[:, 0].sum()
        == 0.0
    )

    # Stall + all on-duty humans busy is occupancy; an idle present human is not.
    sat = np.zeros((2, 3, 26), dtype=np.float32)
    sat[:, 0, 7] = 40.0
    sat[:, 0, 21] = 1.0
    sat[:, 1, 4] = 0.9
    sat[:, 1, 23] = 1.0
    sat[:, 2, 4] = 0.9
    sat[:, 2, 23] = 1.0
    s_hot = node_hot_mask(
        sat, np.zeros((2, 3, 1), dtype=np.float32), window_size_s=60.0, min_hot_windows=2
    )
    assert s_hot[:, 0].sum() == 2.0
    assert s_hot[:, 1].sum() == 0.0
    sat[:, 2, 4] = 0.1
    idle_h = node_hot_mask(
        sat, np.zeros((2, 3, 1), dtype=np.float32), window_size_s=60.0, min_hot_windows=2
    )
    assert idle_h[:, 0].sum() == 0.0


def test_labor_saturated_appended_on_machines_only() -> None:
    # Idle machine still gets X=1 when labor is saturated; humans / unused slots stay 0.
    feats = np.zeros((2, 4, 26), dtype=np.float32)
    feats[:, 0, 21] = 1.0  # idle machine
    feats[:, 1, 22] = 1.0  # gantry
    feats[:, 2, 4] = 0.9
    feats[:, 2, 23] = 1.0  # busy human
    feats[:, 3, 4] = 0.9  # unused slot: high active, no type one-hot
    out = ensure_labor_saturated_feature(feats)
    assert out.shape == (2, 4, 27)
    assert np.allclose(out[:, :, :26], feats)
    assert out[:, 0, 26].tolist() == [1.0, 1.0]
    assert out[:, 1, 26].sum() == 0.0
    assert out[:, 2, 26].sum() == 0.0
    assert out[:, 3, 26].sum() == 0.0
    assert labor_saturated_mask(feats).ravel().tolist() == [True, True]

    feats[:, 2, 4] = 0.1
    idle = ensure_labor_saturated_feature(feats)
    assert idle[:, 0, 26].sum() == 0.0

    already = ensure_labor_saturated_feature(out)
    assert already.shape[-1] == 27
    assert np.allclose(already[:, 0, 26], 1.0)


def test_smooth_occupancy_drops_one_minute_flicker() -> None:
    hot = np.zeros((8, 1), dtype=np.float32)
    hot[1, 0] = 1.0
    hot[3, 0] = 1.0
    hot[4, 0] = 1.0
    hot[6, 0] = 1.0
    out = smooth_occupancy_runs(hot, gap_windows=1, min_windows=2)
    # drop 1-min first, then fill a 1-window hole between remaining runs
    assert out[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    flicker = np.zeros((5, 1), dtype=np.float32)
    flicker[2, 0] = 1.0
    assert float(smooth_occupancy_runs(flicker, gap_windows=1, min_windows=2)[:, 0].sum()) == 0.0


def test_occupancy_node_mask_machine_carrier() -> None:
    feats = np.zeros((2, 5, 26), dtype=np.float32)
    feats[:, 0, 21] = 1.0  # machine
    feats[:, 1, 22] = 1.0  # gantry
    feats[:, 2, 23] = 1.0  # human
    feats[:, 3, 24] = 1.0  # transport_robot
    feats[:, 4, 25] = 1.0  # buffer
    mask = occupancy_node_mask(feats)
    assert mask.tolist() == [1.0, 1.0, 0.0, 1.0, 0.0]
    short = occupancy_node_mask(np.zeros((2, 2, 4), dtype=np.float32))
    assert short.tolist() == [1.0, 1.0]


def test_soft_dice_iou_and_occ_mask() -> None:
    import torch

    from factory_bn.model import occupancy_cell_weight, soft_dice_loss, soft_iou_loss

    logits = torch.tensor([[[8.0, -8.0], [8.0, -8.0]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    step = torch.ones(1, 2)
    node = torch.tensor([[1.0, 0.0]])
    w = occupancy_cell_weight(step, node)
    assert float(w[0, 0, 1]) == 0.0
    assert float(soft_dice_loss(logits, target, w)) < 0.05
    assert float(soft_iou_loss(logits, target, w)) < 0.05
    empty = occupancy_cell_weight(step, torch.zeros(1, 2))
    assert float(soft_dice_loss(logits, target, empty)) == 0.0
    wrong = torch.tensor([[[-8.0, 8.0], [-8.0, 8.0]]])
    assert float(soft_dice_loss(wrong, target, w)) > 0.5
    assert float(soft_dice_loss(wrong, target, w)) > float(soft_dice_loss(logits, target, w))


def test_jobs_remaining_series() -> None:
    kpi = [
        {"complete_s": "100"},
        {"complete_s": "250"},
        {"complete_s": ""},
    ]
    w = np.array([0.0, 60.0, 120.0, 240.0], dtype=np.float32)
    rem, total = jobs_remaining_series(kpi, w)
    assert total == 3.0
    assert rem.tolist() == [3.0, 3.0, 2.0, 2.0]
    assert first_done_index(rem) == 4


def test_pack_remain_and_events() -> None:
    t_len, n = 8, 2
    scores = np.zeros((t_len, n, 1), dtype=np.float32)
    hot = np.zeros((t_len, n), dtype=np.float32)
    hot[3:6, 0] = 1.0
    scores[3:6, 0, 0] = 0.9
    y_s, y_h, mask, remain_len = pack_remain_target(
        scores, hot, t=2, done_ti=7, max_remain_windows=10
    )
    assert remain_len == 5
    assert mask[:5].sum() == 5
    assert mask[5:].sum() == 0
    assert y_h[1:4, 0].sum() == 3
    events = occupancy_to_events(
        y_h[:5],
        resource_ids=["a", "b"],
        first_future_start_s=120.0,
        window_size_s=60.0,
    )
    assert len(events) == 1
    assert events[0]["resource_id"] == "a"
    assert events[0]["n_windows"] == 3
    assert abs(events[0]["duration_s"] - 180.0) < 1e-6


def test_build_samples_remain_horizon() -> None:
    t_len, n_nodes = 20, 3
    feats = np.zeros((t_len, n_nodes, 4), dtype=np.float32)
    scores = np.zeros((t_len, n_nodes, 1), dtype=np.float32)
    scores[15:, 1, 0] = 0.8
    jobs = np.array([3] * 16 + [2, 1, 0, 0], dtype=np.float32)
    ep = {
        "name": "toy",
        "episode_id": 0,
        "features": feats,
        "scores": scores,
        "will": np.zeros((t_len,), dtype=np.float32),
        "mark": np.full((t_len,), -1, dtype=np.int64),
        "cause": np.full((t_len,), -1, dtype=np.int64),
        "tts": np.zeros((t_len,), dtype=np.float32),
        "duration": np.zeros((t_len,), dtype=np.float32),
        "window_start_s": np.arange(t_len, dtype=np.float32) * 60.0,
        "event_node": np.zeros((0,), dtype=np.int64),
        "event_start_s": np.zeros((0,), dtype=np.float32),
        "event_duration_s": np.zeros((0,), dtype=np.float32),
        "event_start_ti": np.zeros((0,), dtype=np.int64),
        "jobs_remaining": jobs,
        "jobs_total": 3.0,
    }
    samples = _build_samples(
        [ep],
        input_window=12,
        output_window=1,
        horizon_windows=3,
        remain_to_jobs_done=True,
        max_remain_windows=16,
        window_size_s=60.0,
        horizon_s=180.0,
    )
    assert samples
    first = samples[0]
    assert first["jobs_remaining"] == 3.0
    assert first["remain_len"] == 6
    assert float(first["remain_mask"].sum()) == 6
    assert first["occ_node_mask"].tolist() == [1.0, 1.0, 1.0]
    last_ok = [s for s in samples if s["jobs_remaining"] > 0]
    assert all(s["remain_len"] > 0 for s in last_ok)
    assert samples[0]["episode_name"] == "toy"
    assert samples[0]["run_dim_id"] == -1
    assert "window_hot" in first


def test_occupancy_horizon_caps_mask_not_remain_len() -> None:
    t_len, n = 30, 2
    scores = np.zeros((t_len, n, 1), dtype=np.float32)
    hot = np.zeros((t_len, n), dtype=np.float32)
    hot[12:20, 0] = 1.0
    y_s, y_h, mask, remain_len = pack_remain_target(
        scores, hot, t=10, done_ti=28, max_remain_windows=15, occupancy_horizon_windows=15
    )
    assert remain_len == 18
    assert int(mask.sum()) == 15
    assert y_h[:15, 0].sum() == 8.0
    assert mask[14] == 1.0


def test_split_episodes_by_name() -> None:
    from factory_bn.dataset import split_episodes_by_name

    names = [f"old_machine2.0__episode_{i:02d}" for i in range(50)]
    names += [f"old_logistics2.0__episode_{i:02d}" for i in range(50)]
    names += [f"new_machine1.0__episode_{i:02d}" for i in range(20)]
    train, val, test = split_episodes_by_name(names, train_ratio=0.7, val_ratio=0.15, seed=42)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train) + len(val) + len(test) == 120
    for prefix, n in (
        ("old_machine2.0", 50),
        ("old_logistics2.0", 50),
        ("new_machine1.0", 20),
    ):
        nt = sum(1 for x in train if x.startswith(prefix))
        nv = sum(1 for x in val if x.startswith(prefix))
        ne = sum(1 for x in test if x.startswith(prefix))
        assert nt + nv + ne == n
        assert nt == int(n * 0.7)
        assert nv >= 1 and ne >= 1

    one_tr, one_va, one_te = split_episodes_by_name(["only"], train_ratio=0.7, val_ratio=0.15)
    assert one_tr == {"only"}
    assert one_va == {"only"}
    assert one_te == {"only"}

    two_tr, two_va, two_te = split_episodes_by_name(
        ["run__0", "run__1"], train_ratio=0.7, val_ratio=0.15, seed=0
    )
    assert len(two_tr) == 1
    assert two_tr.isdisjoint(two_va)
    assert two_va == two_te

    mix_names = [f"n10_machine1.0__episode_{i:02d}" for i in range(10)]
    mix_names += [f"n10_mix_mm1.0__episode_{i:02d}" for i in range(5)]
    tr, va, te = split_episodes_by_name(
        mix_names,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        train_only_contains=["n10_mix_"],
    )
    assert all("n10_mix_" not in x for x in va)
    assert all("n10_mix_" not in x for x in te)
    assert sum(1 for x in tr if "n10_mix_" in x) == 5
    assert tr.isdisjoint(va) and tr.isdisjoint(te)


def test_grouped_embed_and_contrastive() -> None:
    import torch

    from factory_bn.backbone import GroupedTokenEmbedding
    from factory_bn.dataset import run_dim_id
    from factory_bn.model import supervised_contrastive_loss

    assert run_dim_id("n10_machine1.0__episode_00") == 0
    assert run_dim_id("n10_human1.0__episode_01") == 1
    assert run_dim_id("n10_logistics1.0__episode_02") == 2
    assert run_dim_id("n10_material1.0__episode_03") == 3
    assert run_dim_id("n10_none1.0__episode_00") == 4
    assert run_dim_id("toy") == -1

    x = torch.zeros(2, 3, 4, 26)
    y = GroupedTokenEmbedding(26, 64)(x)
    assert y.shape == (2, 3, 4, 64)
    emb26 = GroupedTokenEmbedding(26, 64)
    assert emb26.type_proj[0].in_features == 5
    assert emb26.projs["context"][0].in_features == 6
    x27 = torch.zeros(2, 3, 4, 27)
    y27 = GroupedTokenEmbedding(27, 64)(x27)
    assert y27.shape == (2, 3, 4, 64)
    emb27 = GroupedTokenEmbedding(27, 64)
    assert emb27.type_proj[0].in_features == 5
    assert emb27.projs["context"][0].in_features == 7
    short = GroupedTokenEmbedding(4, 8)(torch.zeros(1, 2, 2, 4))
    assert short.shape == (1, 2, 2, 8)

    z = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    same = supervised_contrastive_loss(z, torch.tensor([0, 0, 1, 1]), temperature=0.2)
    mixed = supervised_contrastive_loss(z, torch.tensor([0, 1, 0, 1]), temperature=0.2)
    assert float(same) < float(mixed)
    assert float(supervised_contrastive_loss(z[:1], torch.tensor([0]))) == 0.0


def test_type_balanced_occupancy_and_contrast_ids() -> None:
    import torch
    import torch.nn.functional as F

    from factory_bn.model import (
        contrastive_class_ids,
        occupied_type_id,
        occupancy_type_node_masks,
        type_balanced_occupancy_losses,
    )

    types = occupancy_type_node_masks(
        ["machine", "gantry", "gantry", "transport_robot"], 4
    )
    assert float(types["machine"].sum()) == 1.0
    assert float(types["gantry"].sum()) == 2.0
    assert float(types["agv"].sum()) == 1.0

    # 1 machine wrong, 20 gantries right: type-mean BCE > cell-mean BCE.
    n = 21
    types_b = occupancy_type_node_masks(["machine"] + ["gantry"] * 20, n)
    logits = torch.zeros(1, 1, n)
    logits[0, 0, 0] = -8.0
    logits[0, 0, 1:] = 8.0
    target = torch.ones(1, 1, n)
    hot_m = torch.ones(1, 1, n)
    loss_hot, loss_dice, _ = type_balanced_occupancy_losses(
        logits, target, hot_m, types_b, hot_pos_weight=1.0, w_dice=0.0, w_iou=0.0
    )
    global_bce = float(F.binary_cross_entropy_with_logits(logits, target))
    assert float(loss_hot) > global_bce + 1.0
    assert float(loss_dice) == 0.0

    empty_agv = dict(types_b)
    empty_agv["agv"] = torch.zeros(n)
    loss_skip, _, _ = type_balanced_occupancy_losses(
        logits, target, hot_m, empty_agv, hot_pos_weight=1.0, w_dice=0.0, w_iou=0.0
    )
    assert abs(float(loss_skip) - float(loss_hot)) < 1e-5

    y_hot = torch.zeros(3, 2, 4)
    y_hot[0, :, 0] = 1.0
    y_hot[1, :, 1] = 1.0
    tid = occupied_type_id(y_hot, types)
    assert tid.tolist() == [1, 2, 0]
    ids = contrastive_class_ids(
        torch.tensor([1, 1, 0]), torch.tensor([0, 0, 1]), tid
    )
    assert ids[0] != ids[1]
    assert int(ids[2]) == 1 * 4 + 0


def test_gantry_fp_costs_more_than_machine_fp() -> None:
    import torch

    from factory_bn.model import occupancy_bce_cell_weight, occupancy_type_node_masks

    types = occupancy_type_node_masks(["machine", "gantry"], 2)
    y = torch.zeros(1, 1, 2)
    same = occupancy_bce_cell_weight(
        y, types, default_pos_weight=4.0, pos_weight_by_type={"gantry": 1.0}
    )
    assert float(same[0, 0, 0]) == 1.0
    assert float(same[0, 0, 1]) == 1.0
    taxed = occupancy_bce_cell_weight(
        y,
        types,
        default_pos_weight=4.0,
        pos_weight_by_type={"machine": 4.0, "gantry": 1.0},
        fp_weight_by_type={"gantry": 2.0},
    )
    assert float(taxed[0, 0, 0]) == 1.0
    assert float(taxed[0, 0, 1]) == 2.0
    y_pos = torch.ones(1, 1, 2)
    pos_w = occupancy_bce_cell_weight(
        y_pos,
        types,
        default_pos_weight=4.0,
        pos_weight_by_type={"machine": 4.0, "gantry": 1.0},
        fp_weight_by_type={"gantry": 2.0},
    )
    assert float(pos_w[0, 0, 0]) == 4.0
    assert float(pos_w[0, 0, 1]) == 1.0


def test_hot_type_affine_identity_and_bias() -> None:
    import torch

    from factory_bn.model import apply_hot_type_affine, occupancy_type_node_masks

    types = occupancy_type_node_masks(["machine", "gantry", "transport_robot"], 3)
    logits = torch.zeros(1, 2, 3)
    ident = apply_hot_type_affine(
        logits, types, torch.ones(3), torch.zeros(3)
    )
    assert torch.equal(ident, logits)
    biased = apply_hot_type_affine(
        logits, types, torch.ones(3), torch.tensor([0.0, -1.0, 0.5])
    )
    assert float(biased[0, 0, 0]) == 0.0
    assert float(biased[0, 0, 1]) == -1.0
    assert float(biased[0, 0, 2]) == 0.5


if __name__ == "__main__":
    test_node_hot_includes_score()
    test_labor_saturated_appended_on_machines_only()
    test_smooth_occupancy_drops_one_minute_flicker()
    test_occupancy_node_mask_machine_carrier()
    test_grouped_embed_and_contrastive()
    test_type_balanced_occupancy_and_contrast_ids()
    test_gantry_fp_costs_more_than_machine_fp()
    test_hot_type_affine_identity_and_bias()
    test_soft_dice_iou_and_occ_mask()
    test_jobs_remaining_series()
    test_pack_remain_and_events()
    test_build_samples_remain_horizon()
    test_occupancy_horizon_caps_mask_not_remain_len()
    test_split_episodes_by_name()
    print("ok")
