from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_graphskillaa import DATASETS, FIXED_MODEL, SEEDS, build_command, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTests(unittest.TestCase):
    def test_public_surface_is_fixed(self):
        self.assertEqual(FIXED_MODEL, "gpt-5.6-sol")
        self.assertEqual(SEEDS, (42, 43, 44))
        self.assertEqual(set(DATASETS), {"searchqa", "docvqa", "livemathematicianbench"})

    def test_commands_lock_models_and_one_evaluation_per_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            for dataset in DATASETS:
                command = build_command(dataset, 42, output_root, resume=False)
                self.assertEqual(command.count(FIXED_MODEL), 2)
                self.assertIn("evaluation.final_test_repeats=1", command)
                self.assertIn("evaluation.run_epoch_test=false", command)
                self.assertIn("evaluation.run_component_ablation_tests=false", command)
                self.assertIn("gradient.update_protocol=case_complete", command)

    def test_manifests_contain_update_only_quadruples(self):
        for dataset in DATASETS:
            report = validate_manifest(dataset)
            self.assertEqual(len(report["update_ids"]), DATASETS[dataset]["train"])
            self.assertEqual(len(report["test_ids"]), DATASETS[dataset]["test"])
            self.assertFalse(set(report["update_ids"]).intersection(report["test_ids"]))
            for group in report["manifest"]["training_group_policy"]["groups"]:
                self.assertEqual(set(group), {"group_id", "train_ids"})
                self.assertEqual(len(group["train_ids"]), 4)

    def test_config_declares_fixed_roles_and_three_epochs(self):
        base = (ROOT / "configs/_base_/default.yaml").read_text(encoding="utf-8")
        hp = (ROOT / "configs/_base_/hyperparams.yaml").read_text(encoding="utf-8")
        self.assertIn("teacher_profile: teacher_gpt56", base)
        self.assertIn("optimizer: gpt-5.6-sol", base)
        self.assertIn("target: gpt-5.6-sol", base)
        self.assertIn("num_epochs: 3", hp)
        self.assertIn("final_test_repeats: 1", hp)
        self.assertIn("run_epoch_test: false", hp)

    def test_no_test_identifier_is_nested_in_update_manifest(self):
        for dataset in DATASETS:
            manifest = json.loads((ROOT / "data" / dataset / "split_manifest.json").read_text())
            for group in manifest["training_group_policy"]["groups"]:
                self.assertNotIn("test_id", group)
                self.assertNotIn("test_ids", group)
                self.assertNotIn("question_ids", group)


if __name__ == "__main__":
    unittest.main()
