import importlib
import sys
import types
import unittest
from unittest.mock import patch


class BenchCliTest(unittest.TestCase):

    def import_bench_with_stubs(self):
        torch_stub = types.SimpleNamespace(manual_seed=lambda seed: None)
        nanovllm_stub = types.SimpleNamespace(
            LLM=object,
            SamplingParams=lambda **kwargs: types.SimpleNamespace(**kwargs),
        )
        with patch.dict(sys.modules, {"torch": torch_stub, "nanovllm": nanovllm_stub}):
            sys.modules.pop("bench", None)
            return importlib.import_module("bench")

    def test_matrix_repeat_and_jsonl_flags_parse(self):
        bench = self.import_bench_with_stubs()
        argv = [
            "bench.py",
            "--matrix",
            "--workload",
            "shared_prefix",
            "--num-seqs",
            "128",
            "--repeat",
            "5",
            "--output-jsonl",
            "logs/trick1_shared_matrix.jsonl",
        ]
        with patch.object(sys, "argv", argv):
            args = bench.parse_args()
        self.assertTrue(args.matrix)
        self.assertEqual(args.repeat, 5)
        self.assertEqual(args.output_jsonl, "logs/trick1_shared_matrix.jsonl")

    def test_single_run_json_flag_parse(self):
        bench = self.import_bench_with_stubs()
        argv = ["bench.py", "--single-run-json"]
        with patch.object(sys, "argv", argv):
            args = bench.parse_args()
        self.assertTrue(args.single_run_json)


if __name__ == "__main__":
    unittest.main()
