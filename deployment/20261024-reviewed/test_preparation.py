"""Small stdlib-only preparation checks. No network, models, Docker or cloud."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

BASE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_source = load("prepare_source")
hydrate = load("hydrate_detector")
deploy = load("deploy_candidate")


class PreparationTests(unittest.TestCase):
    def test_context_allowlist(self):
        for path in ("services/cost_policy.py", "agents/verifier_agent/config/domain_intelligence.yaml",
                     "orchestration/auth.py", "artifacts/detector-best/config.json"):
            self.assertTrue(prepare_source.permitted(path), path)
        for path in (".env", "agents/verifier_agent/.env.production", "agents/a/tests/test_x.py",
                     "agents/memory_agent/data/entries.json", "runtime-logs/a.json", "README.md",
                     "frontend-v2/src/lib/api.js", "research/input.json", "../orchestration/api.py",
                     "artifacts/detector-best/model.safetensors", "artifacts/detector-best/test_metrics.json"):
            self.assertFalse(prepare_source.permitted(path), path)

    def test_exact_detector_identity(self):
        self.assertEqual(hydrate.EXPECTED_SIZE, 283348948)
        self.assertEqual(hydrate.EXPECTED_SHA256, "470261f81870f5c4f97fe85918d02aa08673ac08744b5cae4d4621ee9d631bf9")
        self.assertIn("82cdc6fb734f5fa31cd345a4060f60178bea2118", hydrate.URL)

    def test_verified_stream(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model"
            data = b"small fixture, not a model"
            hydrate.write_verified(io.BytesIO(data), target, len(data), hashlib.sha256(data).hexdigest())
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(list(Path(folder).iterdir()), [target])

    def test_bad_stream_rejected_and_temporary_removed(self):
        for data, size, digest in ((b"x", 0, "bad"), (b"x", 2, "bad"), (b"x", 1, "bad")):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    hydrate.write_verified(io.BytesIO(data), Path(folder) / "model", size, digest)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model"
            target.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                hydrate.write_verified(io.BytesIO(b"x"), target, 1, "bad")
            self.assertEqual(target.read_bytes(), b"preserve")

    def test_model_and_dependency_pins(self):
        models = json.loads((BASE / "models-pinned.json").read_text())
        self.assertEqual({m["repo_id"] for m in models}, {
            "BAAI/bge-m3", "BAAI/bge-reranker-large", "cross-encoder/nli-deberta-v3-base",
            "sentence-transformers/all-MiniLM-L6-v2"})
        self.assertTrue(all(re.fullmatch("[0-9a-f]{40}", m["revision"]) for m in models))
        lock = (BASE / "requirements-linux.lock").read_text()
        self.assertIn("torch==2.13.0+cpu", lock)
        self.assertIn("--hash=sha256:", lock)
        docker = (BASE / "Dockerfile").read_text()
        self.assertIn("--require-hashes", docker)
        self.assertIn("HALLUCIGUARD_FREE_ONLY=true", docker)
        self.assertNotIn("OPENROUTER_API_KEY=", docker)
        self.assertNotIn("JWT_SECRET=", docker)

    def test_build_is_bounded_and_does_not_deploy(self):
        build = (BASE / "cloudbuild.yaml").read_text()
        self.assertIn("timeout: 1200s", build)
        self.assertIn("timeout: 1100s", build)
        self.assertIn("machineType: E2_STANDARD_2", build)
        self.assertEqual(build.count("name: gcr.io/cloud-builders/docker"), 1)
        self.assertNotIn("gcloud", build)
        self.assertNotIn("triggers", build)
        config = json.loads((BASE / "candidate-config.json").read_text())
        self.assertEqual((config["cpu"], config["memory_gib"], config["concurrency"], config["max_instances"]), (2, 8, 1, 1))
        self.assertFalse(config["hard_zero_cost_cap_enforced"])
        self.assertFalse(config["memory_durable"])
        self.assertFalse(config["trial_credit_spending_approved"])
        self.assertEqual(config["build_worker"], "E2_STANDARD_2")
        self.assertFalse(config["billing_account_free_build_minutes_remaining_verified"])

    def test_all_hosted_model_selectors_use_the_tested_free_route(self):
        docker = (BASE / "Dockerfile").read_text()
        for variable in ("OPENROUTER_MODEL", "HALLUCIGUARD_LLM_MODEL", "HG_CORRECTOR_OPENROUTER_MODEL"):
            assignments = re.findall(r"(?:^|\s)" + variable + r"=([^\s\\]+)", docker)
            self.assertEqual(assignments, ["openrouter/free"], variable)
        self.assertNotIn("liquid/lfm-2.5-2.6b:free", docker)

    def test_candidate_plan_is_private_and_separate(self):
        result = deploy.plan("a" * 64, "runtime@" + deploy.PROJECT + ".iam.gserviceaccount.com", "provider-key:1", "jwt-key:2")
        command = result["command"]
        self.assertEqual(command[3], "halluciguard-staging-20261024")
        self.assertIn("--no-allow-unauthenticated", command)
        self.assertIn("--max-instances=1", command)
        self.assertIn("--no-cpu-boost", command)
        self.assertFalse(result["execute"])
        self.assertFalse(result["production_traffic_change"])
        self.assertFalse(any("update-traffic" in part for part in command))

    def test_candidate_plan_rejects_unpinned_or_wrong_project_inputs(self):
        valid = ["a" * 64, "runtime@" + deploy.PROJECT + ".iam.gserviceaccount.com", "key:1", "jwt:1"]
        for index, invalid in ((0, "latest"), (1, "runtime@other.iam.gserviceaccount.com"), (2, "key:latest"), (3, "raw secret value")):
            arguments = list(valid)
            arguments[index] = invalid
            with self.assertRaises(ValueError):
                deploy.plan(*arguments)

    def test_build_file_inventory_has_no_markdown_or_credentials(self):
        self.assertTrue(all((BASE / name).is_file() for name in prepare_source.BUILD_FILES))
        self.assertFalse(any(name.endswith(".md") or name.startswith(".env") for name in prepare_source.BUILD_FILES))
        self.assertEqual(prepare_source.REVIEW_COMMIT, "55ef3570cf9bb5d6556aeb404699d52e29085d2d")


if __name__ == "__main__":
    unittest.main()
