import os
import unittest
import shutil
import torch
from intel_extension_for_transformers.evaluation.lm_evaluation_harness.evaluator import evaluate
from transformers import (
    AutoModelForCausalLM
)

class TestLmEvaluationHarness(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.makedirs("./saved_results_clm", exist_ok=True)
        os.makedirs("./saved_results_seq2seq", exist_ok=True)
        os.makedirs("./saved_results_jit", exist_ok=True)
        self.clm_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True
        )
        tmp_model = torch.jit.trace(self.clm_model, self.clm_model.dummy_inputs["input_ids"])
        self.jit_model = torch.jit.freeze(tmp_model.eval())

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved_results_clm", ignore_errors=True)
        shutil.rmtree("./saved_results_seq2seq", ignore_errors=True)
        shutil.rmtree("./saved_results_jit", ignore_errors=True)
        shutil.rmtree("./lm_cache", ignore_errors=True)

    def test_evaluate_for_casualLM(self):
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="facebook/opt-125m",tokenizer="facebook/opt-125m",dtype=float32',
            tasks=["piqa"],
            limit=20,
            output_dir="./saved_results_clm"
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.70)

    def test_evaluate_for_Seq2SeqLM(self):
        results = evaluate(
            model="hf-seq2seq",
            model_args='pretrained="t5-small",tokenizer="t5-small",dtype=float32',
            tasks=["piqa"],
            limit=20,
            output_dir="./saved_results_seq2seq"
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.60)

    def test_evaluate_for_JitModel(self):
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="facebook/opt-125m",tokenizer="facebook/opt-125m",dtype=float32',
            user_model=self.jit_model,
            tasks=["piqa"],
            limit=20,
            output_dir="./saved_results_jit"
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.70)


if __name__ == "__main__":
    unittest.main()
