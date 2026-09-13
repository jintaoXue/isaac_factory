"""Single width intervention, old configurations and unchanged spatial prefix."""
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys
import torch
from torch.nn import functional as F
TOOLS = Path(__file__).resolve().parents[1] / 'isaaclab_tasks/direct/hc_factory/tools'
sys.path.insert(0, str(TOOLS))
from train_dense_baseline_control import dense_configuration
from factory_baselines.b4_gcn_gru import B4GcnGru, B4ModelConfig
from factory_baselines.b5_gat_gru import B5GatGru, B5ModelConfig


def cases():
    for name, cls, cfg in [('B4', B4GcnGru, B4ModelConfig), ('B5', B5GatGru, B5ModelConfig)]:
        for seed in (42, 43):
            _, overrides, _ = dense_configuration(name, 'near_precursor', seed, 'cpu')
            yield name, seed, cls, cfg(27, 0, 38, **overrides)


def test_only_gru_width_changes_against_near_and_old_variant_configs_stay_exact():
    path = 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/train_dense_baseline_control.py'
    source = subprocess.check_output(['git', 'show', 'ee838f59d2bdf2893a35ec00acae1308f9dd0e07:' + path])
    old = {'__name__': '_old_dense_control'}; exec(compile(source, 'old_dense_control', 'exec'), old)
    for model in ('B4', 'B5'):
        for seed in (42, 43):
            for variant in old['DENSE_VARIANTS']:
                if model == 'B4' and variant == 'vector_gat': continue
                before = old['dense_configuration'](model, variant, seed, 'cpu')
                assert before == dense_configuration(model, variant, seed, 'cpu')
            parent, arch, loss = dense_configuration(model, 'near_precursor', seed, 'cpu')
            candidate, new, new_loss = dense_configuration(model, 'gru_capacity32', seed, 'cpu')
            assert new == {**arch, 'gru_hidden': 32} and loss == new_loss
            assert replace(candidate, training_profile=parent.training_profile) == parent
            assert not candidate.evaluate_test and candidate.max_epochs == 60


def test_actual_models_keep_spatial_initialization_and_reduce_temporal_parameters():
    for name, seed, cls, config in cases():
        torch.manual_seed(seed); parent = cls(config)
        torch.manual_seed(seed); small = cls(replace(config, gru_hidden=32))
        assert small.gru.hidden_size == 32 and small.gru.input_size == 64
        assert small.gru.num_layers == 1 and small.gru.dropout == 0
        assert small.readout_dropout.p == 0 and small.history_attention is None and small.history_graph is None
        assert type(parent.heads) is type(small.heads) and not small.config.event_onset_aux
        spatial = [k for k in parent.state_dict() if k.startswith(('input_projection.', 'gcn', 'gat'))]
        assert spatial and all(torch.equal(parent.state_dict()[k], small.state_dict()[k]) for k in spatial)
        assert sum(p.numel() for p in small.parameters()) < sum(p.numel() for p in parent.parameters()) / 2
        assert sum(p.numel() for p in small.gru.parameters()) == 9408


def test_real_width_inputs_masks_and_upcoming_gradient_path():
    batch = dict(x=torch.randn(2, 30, 38, 27), adjacency=torch.ones(2, 38, 38),
        node_mask=torch.cat([torch.ones(2, 37), torch.zeros(2, 1)], 1),
        target_node_mask=torch.cat([torch.ones(2, 37), torch.zeros(2, 1)], 1),
        global_features=torch.empty(2, 30, 0), jobs_remaining=torch.ones(2), jobs_total=torch.ones(2) * 2,
        event_precursor=torch.randn(2, 38, 23))
    for name, seed, cls, config in cases():
        torch.manual_seed(seed); small = cls(replace(config, gru_hidden=32)).train()
        result = small(**batch)
        assert result['node_hidden'].shape == (2, 38, 32)
        assert torch.count_nonzero(result['node_hidden'][:, -1]) == 0
        assert result['event_will_logit'].shape == (2, 38) and all(torch.isfinite(v).all() for v in result.values())
        targets = torch.zeros(2, 37); targets[:, 0] = 1
        F.binary_cross_entropy_with_logits(result['event_will_logit'][:, :37], targets).backward()
        parameters = dict(small.named_parameters())
        for key in ('gru.weight_ih_l0', 'gru.weight_hh_l0', 'history_readout.0.weight', 'heads.event_will_head.0.weight',
                    'gcn1.linear.weight' if name == 'B4' else 'gat1.projection.weight'):
            g = parameters[key].grad; assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
