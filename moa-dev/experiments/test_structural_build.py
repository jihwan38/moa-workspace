#!/usr/bin/env python3
"""Test recipe source selection and assembly without invoking Docker or a compiler."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("structural_build", Path(__file__).with_name("structural_build.py"))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
ROOT = builder.ROOT


class BeforeDocker(Exception):
    pass


class RecipeSelection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="recipe-test-", dir=ROOT / "moa-dev/experiments/candidates")
        self.here = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.variant = self.here / "variant"
        self.variant.mkdir()
        self.projection = b"// Source selection test; never compiled.\n"
        (self.variant / "projection.rs").write_bytes(self.projection)
        self.config = {
            "base_commit": json.loads((ROOT / "baseline/e55.json").read_text())["git"]["commit"],
            "comparison_recipe": "baseline/e104.json",
            "target_kernel": "sliding_attention_output",
            "changed_sources": {"projection.rs": "src/device/sliding/projection.rs"},
        }

    def resolve(self, config=None):
        (self.here / "candidate.json").write_text(json.dumps(config or self.config))
        return builder.resolve_plan(self.here.name, "variant")[2]

    def test_source_origin_is_independent_of_e104_comparison(self):
        for source in ("e55", "e79", "e104"):
            with self.subTest(source=source):
                self.config["base_commit"] = json.loads((ROOT / f"baseline/{source}.json").read_text())["git"]["commit"]
                plan = self.resolve()
                self.assertEqual(plan["source_base_commit"], self.config["base_commit"])
                self.assertEqual(plan["performance_comparison"]["id"], "E104")
                self.assertEqual(plan["expected_unchanged_images"], {})
                expected = subprocess.check_output(["git", "show", self.config["base_commit"] + ":furiosa-opt-gemma4-12B/src/device/shared/mlp.rs"], cwd=ROOT)
                self.assertEqual(plan["source_origin_inputs"]["src/device/shared/mlp.rs"], builder.digest(expected))

    def test_variant_recipe_selects_a_different_origin(self):
        variant = copy.deepcopy(self.config)
        variant["base_commit"] = json.loads((ROOT / "baseline/e79.json").read_text())["git"]["commit"]
        (self.variant / "recipe.json").write_text(json.dumps(variant))
        self.assertEqual(self.resolve()["source_base_commit"], variant["base_commit"])

    def test_duplicate_target_cannot_silently_discard_a_method(self):
        self.config["changed_sources"]["another.rs"] = "src/device/sliding/projection.rs"
        with self.assertRaisesRegex(ValueError, "duplicate targets"):
            self.resolve()

    def test_control_images_cannot_be_used_for_a_different_source(self):
        self.config["source_reference_build"] = "moa-dev/experiments/candidates/c07-native-scale-dpe/build-c07-e104-reviewed-v1-repro.json"
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.resolve()

    def test_existing_recipes_resolve_with_verified_source_reference(self):
        for candidate, variant in (("c09-output-pipeline", "two-stage-v1"), ("c11-ffn-partial-vrf", "two-row-v1"), ("c10-direct-weight-trf", "direct-v1")):
            with self.subTest(candidate=candidate):
                plan = builder.resolve_plan(candidate, variant)[2]
                self.assertEqual(plan["performance_comparison"]["id"], "E104")
                self.assertEqual(len(plan["expected_unchanged_images"]), 3)

    def test_real_builder_assembles_selected_origin_before_docker(self):
        # Exercise the actual git archive/copy path: mocking only the first
        # Docker invocation catches a hard-coded BASE in the builder itself.
        self.resolve()
        artifacts = ROOT / "moa-dev/experiments/artifacts" / self.here.name
        self.addCleanup(lambda: shutil.rmtree(artifacts) if artifacts.exists() else None)
        original_run = subprocess.run

        def stop_at_docker(args, *a, **kw):
            if args[0] == "docker":
                raise BeforeDocker()
            return original_run(args, *a, **kw)

        with patch.object(subprocess, "run", side_effect=stop_at_docker):
            with self.assertRaises(BeforeDocker):
                builder.build(self.here.name, "variant", "assembly-only")
        crate = artifacts / "assembly-only/crate"
        expected = subprocess.check_output(["git", "show", self.config["base_commit"] + ":furiosa-opt-gemma4-12B/src/device/shared/mlp.rs"], cwd=ROOT)
        self.assertEqual((crate / "src/device/shared/mlp.rs").read_bytes(), expected)
        self.assertEqual((crate / "src/device/sliding/projection.rs").read_bytes(), self.projection)


if __name__ == "__main__":
    unittest.main()
