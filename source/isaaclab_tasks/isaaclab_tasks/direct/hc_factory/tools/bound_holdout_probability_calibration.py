#!/usr/bin/env python3
"""A necessary AP bound for single-score calibration at overall report P>=.8."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def minimum_ap_for_hits(upcoming, ongoing, hits):
    """Optimistically give all ongoing targets correct reports and no timing FP.

    P>=4/5 implies FP<=floor(TP/4)<=floor((ongoing+hits)/4)=F.
    A threshold with h upcoming hits includes at least h upcoming positives and
    at most F negatives in the upcoming-versus-negative ranking. Its j-th
    positive contributes at least j/(F+j)/U to AP, even with tied scores.
    Ignore every positive after the threshold for a deliberately loose bound.
    """
    if not all(isinstance(n, int) for n in (upcoming, ongoing, hits)) or upcoming <= 0 or ongoing < 0 or not 0 <= hits <= upcoming:
        raise ValueError('Invalid event counts')
    false_positives = (ongoing + hits) // 4
    return math.fsum(j / (false_positives + j) for j in range(1, hits + 1)) / upcoming


def calibration_bound(ap, upcoming, ongoing):
    if not isinstance(ap, (int, float)) or not math.isfinite(ap) or not 0 <= ap <= 1:
        raise ValueError('Invalid AP')
    minimum_ap_for_hits(upcoming, ongoing, 0)
    possible = [h for h in range(upcoming + 1) if minimum_ap_for_hits(upcoming, ongoing, h) <= ap + 1.e-12]
    maximum = max(possible)
    half = (upcoming + 1) // 2
    return dict(maximum_possible_upcoming_hits=maximum, upcoming_recall_upper_bound=maximum / upcoming,
        actual_upcoming_ap=ap, upcoming_count=upcoming, ongoing_count=ongoing,
        half_recall_hits_required=half, half_recall_ap_necessary_lower_bound=minimum_ap_for_hits(upcoming, ongoing, half),
        half_recall_ruled_out=half > maximum)


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--source_commit', required=True)
    parser.add_argument('--model', choices=('b4', 'b5'), required=True); args = parser.parse_args()
    repo = Path.cwd().resolve(); assert repo == Path('/home/sci/work/BSTAN_isaac_factory')
    tools = 'source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/tools/'
    d = (repo / tools).parent / 'output/bottleneck_dataset/experiments/factory_pdformer_134_v3'; tag = 'baseline_episodeholdout20260913'
    out = d / f'{tag}_{args.model}s42_calibration_bound.json'; assert not out.exists(), 'Reuse completed analytic bound'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == 'ee838f59d2bdf2893a35ec00acae1308f9dd0e07'
    own = subprocess.check_output(['git', 'show', args.source_commit + ':' + tools + Path(__file__).name]); assert sha(Path(__file__)) == hashlib.sha256(own).hexdigest()
    completed_path = d / f'{tag}_{args.model}s42_complete.json'; completed_sha = sha(completed_path)
    completed = json.loads(completed_path.read_text()); ident = completed['identity']
    assert ident['source_commit'] == 'ecc39e81c8f14483eaa50d90beea9743f9ac485c' and ident['model'] == args.model and ident['seed'] == 42
    assert completed['status'] == 'fixed_sixty_epoch_training_and_nine_views_complete' and len(completed['diagnostics']) == 9 and not completed['test_evaluated']
    plan_path = d / 'baseline_episode_holdout_plan20260913.json'; assert sha(plan_path) == ident['plan_sha256']; plan = json.loads(plan_path.read_text())
    rows = []; seen = set()
    for row in completed['diagnostics']:
        assert sha(d / row['file']) == row['sha256']; r = json.loads((d / row['file']).read_text()); assert r == row['result']
        p = r['provenance']; assert p['identity'] == ident and not p['test_evaluated']; e, v = p['epoch'], p['view']; assert (e, v) not in seen; seen.add((e, v))
        assert p['sample_indices'] == plan['plan']['sample_indices'][v]
        report = r['report']; u, o = (report['groups'][k]['count'] for k in ('upcoming', 'ongoing'))
        assert (u, o) == (plan['counts'][v]['upcoming'], plan['counts'][v]['ongoing'])
        result = calibration_bound(report['ranking']['upcoming_vs_negative']['tie_aware_average_precision'], u, o)
        rows.append(dict(epoch=e, view=v, report_file=row['file'], report_sha256=row['sha256'], **result))
        print('CALIBRATION_BOUND', args.model, e, v, result['maximum_possible_upcoming_hits'], '/', u, 'R', result['upcoming_recall_upper_bound'], 'AP50_NECESSARY', result['half_recall_ap_necessary_lower_bound'], flush=True)
    assert seen == {(e, v) for e in (10, 30, 60) for v in ('fit', 'heldout', 'original_validation')}
    assert sha(completed_path) == completed_sha
    record = dict(status='nine_fixed_view_analytic_single_score_calibration_bounds_complete', source_commit=args.source_commit,
        source_sha256=sha(Path(__file__)), model=args.model, completed_file=completed_path.name, completed_sha256=completed_sha,
        required_overall_report_precision=.8, results=rows,
        derivation='If h upcoming reports are correct, TP<=O+h and FP<=floor((O+h)/4). The jth included upcoming positive has at most F negative predecessors, so AP>=sum(j/(F+j),j=1..h)/U. Ignore all later positives and grant perfect ongoing/start prediction. Ties cannot invalidate this lower bound.',
        scope='Necessary, generally unattainable bound for a single global threshold or monotone score calibration preserving ranking. Not an exact threshold frontier, an adopted threshold, node-specific or multihead recalibration, an information bound, or an architecture ceiling.',
        model_training=False, model_forward=False, threshold_search=False, checkpoint_selection=False, test_evaluated=False, goal_met=False)
    with out.open('x') as f: json.dump(record, f, indent=2); f.write('\n')
    print('CALIBRATION_BOUND_COMPLETE', out.stat().st_size, sha(out), flush=True)


if __name__ == '__main__': main()
