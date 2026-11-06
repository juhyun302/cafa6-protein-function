from pathlib import Path
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cafa_pipeline.cli import run_demo
from cafa_pipeline.download import download_one
from cafa_pipeline.ensemble import patch_scores, rank_ensemble, validate_submission
from cafa_pipeline.io import normalize_id, read_predictions, write_predictions
from cafa_pipeline.structure import extract_accessions, load_ancestors, make_submission, normalize_scores, propagate, transfer_go


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def text_file(self, name, content):
        path = self.directory / name
        path.write_text(content, encoding="utf-8")
        return path

    def frame(self, rows):
        return pd.DataFrame(rows, columns=["protein_id", "go_id", "aspect", "score"])

    def test_legacy_tsv_preserves_first_prediction(self):
        path = self.text_file("legacy.tsv", "Q1\tGO:9000001\t0.8\nQ2\tGO:9000002\t0.7\n")
        frame = read_predictions(path)
        self.assertEqual(frame["protein_id"].tolist(), ["Q1", "Q2"])
        self.assertEqual(frame["score"].tolist(), [0.8, 0.7])

    def test_normalized_score_takes_precedence(self):
        path = self.text_file("scores.tsv", "protein_id\tgo_id\tscore\tscore_norm\nQ1\tGO:9000001\t4\t0.8\n")
        self.assertEqual(read_predictions(path).iloc[0]["score"], 0.8)

    def test_round_trip_empty_namespace(self):
        path = self.directory / "empty.tsv"
        write_predictions(self.frame([]), path)
        self.assertTrue(read_predictions(path).empty)

    def test_malformed_inputs_are_rejected(self):
        for content in ["Q1\tGO:9000001\tNaN\n", "Q1\tGO:9000001\t-1\n", "Q1\tbad-term\t0.3\n"]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_predictions(self.text_file("invalid.tsv", content))

    def test_id_cleanup_prevents_self_hit(self):
        self.assertEqual(normalize_id("000001_AF-P12345-F1-model_v4.cif"), "P12345")
        frame = transfer_go(ROOT / "examples/foldseek-hits.tsv", ROOT / "examples/annotations.tsv")
        self.assertNotIn("GO:9999999", set(frame["go_id"]))
        self.assertEqual(set(frame["protein_id"]), {"DEMO_Q1", "DEMO_Q2"})

    def test_normalization_and_zero_scores(self):
        frame = self.frame([("Q", "GO:9000001", "BP", 2), ("Q", "GO:9000002", "BP", 1),
                            ("ZERO", "GO:9000001", "BP", 0)])
        result = normalize_scores(frame, top_k=5, min_score=0)
        self.assertEqual(result[result.protein_id == "Q"]["score"].tolist(), [1.0, 0.5])
        self.assertEqual(result[result.protein_id == "ZERO"].iloc[0]["score"], 0)
        self.assertTrue(np.isfinite(result["score"]).all())

    def test_transitive_max_propagation(self):
        path = self.text_file("parents.tsv", "child\tparent\nGO:9000002\tGO:9000001\nGO:9000001\tGO:9000000\n")
        result = propagate(self.frame([("Q", "GO:9000002", "BP", 0.9), ("Q", "GO:9000000", "BP", 0.2)]),
                           load_ancestors(parents=path))
        self.assertEqual(result.set_index("go_id")["score"].to_dict(),
                         {"GO:9000000": 0.9, "GO:9000001": 0.9, "GO:9000002": 0.9})

    def test_cycles_are_rejected(self):
        path = self.text_file("cycle.tsv", "child\tparent\nGO:9000000\tGO:9000001\nGO:9000001\tGO:9000000\n")
        with self.assertRaisesRegex(ValueError, "cycle"):
            load_ancestors(parents=path)

    def test_sparse_ancestor_artifacts(self):
        mappings = self.directory / "mapping.pkl"
        with mappings.open("wb") as stream:
            pickle.dump({"go_to_idx": {"GO:9000000": 0, "GO:9000001": 1},
                         "idx_to_go": {0: "GO:9000000", 1: "GO:9000001"}}, stream)
        matrix = self.directory / "ancestors.npz"
        save_npz(matrix, csr_matrix([[0, 0], [1, 0]]))
        self.assertEqual(load_ancestors(mapping_pkl=mappings, ancestor_npz=matrix)["GO:9000001"], ("GO:9000000",))

    def test_top_n_per_namespace(self):
        for aspect, term in [("bp", "GO:9000001"), ("mf", "GO:9100001"), ("cc", "GO:9200001")]:
            write_predictions(self.frame([("Q", term, aspect.upper(), 0.8)]),
                              self.directory / f"pred_{aspect}_propagated.tsv")
        self.assertEqual(len(make_submission(self.directory, top_n=1)), 3)

    def test_empty_high_score_patch(self):
        base = self.frame([("Q", "GO:9000001", "", 0.7)])
        low = self.frame([("Q", "GO:9000001", "", 0.98)])
        self.assertEqual(patch_scores(base, low).iloc[0]["score"], 0.7)

    def test_patch_overwrites_and_adds(self):
        base = self.frame([("Q", "GO:9000001", "", 0.2)])
        high = self.frame([("Q", "GO:9000001", "", 1.0), ("Q2", "GO:9000002", "", 0.99)])
        self.assertEqual(patch_scores(base, high).set_index("protein_id")["score"].to_dict(),
                         {"Q": 1.0, "Q2": 0.99})

    def test_global_rank_ties(self):
        base = self.frame([("Q", "GO:9000001", "", 0.8)])
        other = self.frame([("Q2", "GO:9000002", "", 0.6)])
        low = self.frame([("Q", "GO:9000001", "", 0.1)])
        self.assertEqual(rank_ensemble(base, other, low)["score"].tolist(), [0.75, 0.75])

    def test_invalid_submission(self):
        for frame in [self.frame([("Q", "GO:9000001", "", 1.2)]),
                      self.frame([("Q", "GO:9000001", "", 0.2), ("Q", "GO:9000001", "", 0.3)])]:
            with self.assertRaises(ValueError):
                validate_submission(frame)

    def test_download_failure_states(self):
        missing = urllib.error.HTTPError("https://example.invalid", 404, "missing", {}, None)
        with patch("urllib.request.urlopen", side_effect=missing):
            self.assertEqual(download_one("P12345", self.directory)[1], "NO_STRUCTURE")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            self.assertEqual(download_one("P12345", self.directory, retries=1)[1], "FAIL")

    def test_existing_structure_is_not_downloaded(self):
        self.text_file("P12345.cif", "data_P12345\n" + "#" * 1300)
        with patch("urllib.request.urlopen") as request:
            self.assertEqual(download_one("P12345", self.directory)[1], "SKIP")
            request.assert_not_called()

    def test_fasta_header_formats(self):
        fasta = self.text_file("query.fasta", ">sp|P12345|NAME\nMKTA\n>Q99999 plain\nGAVL\n")
        output = self.directory / "ids.txt"
        self.assertEqual(extract_accessions(fasta, output), 2)
        self.assertEqual(output.read_text().splitlines(), ["P12345", "Q99999"])

    def test_serial_and_parallel_full_demo(self):
        expected = run_demo(ROOT / "examples", self.directory / "serial")
        process = subprocess.run([sys.executable, str(ROOT / "scripts/cafa.py"), "demo",
                                  "--examples-dir", str(ROOT / "examples"), "--output-dir",
                                  str(self.directory / "parallel"), "--workers", "2"],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), expected)
        self.assertEqual((self.directory / "serial/submission.tsv").read_text(),
                         (self.directory / "parallel/submission.tsv").read_text())

    def test_azure_dry_run(self):
        process = subprocess.run([sys.executable, str(ROOT / "azureml/submit.py"),
                                  "--stage", "propagate", "--data", "azureml:demo:1",
                                  "--compute", "example-cpu", "--environment",
                                  "azureml:cafa-structure:1", "--dry-run"],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--ancestor-npz", json.loads(process.stdout)["command"])


if __name__ == "__main__":
    unittest.main()
