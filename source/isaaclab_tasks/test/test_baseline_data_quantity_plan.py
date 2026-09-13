import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"isaaclab_tasks/direct/hc_factory/tools"))
from prepare_baseline_data_quantity import nested_episode_plan


def fixture():
    sizes=[2]*11+[5]+[6]*3+[9]+[10]*2+[11]*3
    groups={"fit":[],"heldout":[],"original_validation":[]}
    indices={key:[] for key in groups}
    rows=[]
    for run,n in enumerate(sizes):
        for view,count in (("fit",n),("heldout",1),("original_validation",1)):
            for episode in range(count):
                group=f"run_{run}:env_{view}_{episode}"
                groups[view].append(group)
                for _ in range(2+(episode%3)):
                    i=len(rows);indices[view].append(i)
                    rows.append(dict(group_id=group,sample_index=i,split="validation" if view=="original_validation" else "train",label=0))
    return dict(groups=groups,sample_indices=indices),rows


class DataQuantityPlanTests(unittest.TestCase):
    def test_exact_nested_half_all_runs_and_frozen_evaluation(self):
        parent,rows=fixture();plan=nested_episode_plan(parent,rows)
        self.assertEqual(len(plan["groups"]["fit"]),54)
        self.assertEqual(len(plan["groups"]["unused_parent_fit"]),53)
        self.assertEqual(len(plan["raw_run_strata"]),21)
        self.assertTrue(all(s["retained_episodes"]>0 and s["unused_episodes"]>0 for s in plan["raw_run_strata"]))
        for key in ("heldout","original_validation"):
            self.assertEqual(plan["sample_indices"][key],parent["sample_indices"][key])
        self.assertEqual(set(plan["sample_indices"]["fit"])|set(plan["sample_indices"]["unused_parent_fit"]),set(parent["sample_indices"]["fit"]))

    def test_label_and_iteration_order_do_not_select_episodes(self):
        parent,rows=fixture();expected=nested_episode_plan(parent,rows)
        for row in rows:row["label"]=1-row["label"]
        rows.reverse()
        for key in parent["groups"]:parent["groups"][key].reverse();parent["sample_indices"][key].reverse()
        self.assertEqual(nested_episode_plan(parent,rows),expected)

    def test_normalization_partition_includes_unused_only_as_heldout(self):
        parent,rows=fixture();plan=nested_episode_plan(parent,rows);views=plan["normalization_views"]
        self.assertFalse(set(views["fit"])&set(views["heldout"]))
        self.assertEqual(set(views["fit"])|set(views["heldout"]),set(parent["sample_indices"]["fit"]+parent["sample_indices"]["heldout"]))
        self.assertTrue(set(plan["sample_indices"]["unused_parent_fit"])<=set(views["heldout"]))
        self.assertEqual(plan["evaluation_views"]["heldout"],parent["sample_indices"]["heldout"])

    def test_reject_duplicate_missing_cross_split_and_test(self):
        parent,rows=fixture()
        bad=copy.deepcopy(rows);bad[0]["split"]="test"
        crossed=copy.deepcopy(rows);crossed[0]["group_id"]=parent["groups"]["heldout"][0]
        for candidate in (rows+[rows[0]],rows[1:],bad,crossed):
            with self.assertRaises(ValueError):nested_episode_plan(parent,candidate)


if __name__ == "__main__":unittest.main()
