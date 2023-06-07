import os
import unittest
import shutil
import torch
from intel_extension_for_transformers.evaluation import evaluate
from transformers import (
    AutoModelForCausalLM
)

class TestLmEvaluationHarness(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.clm_model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-gptj",
            torchscript=True
        )
        tmp_model = torch.jit.trace(self.clm_model, self.clm_model.dummy_inputs["input_ids"])
        self.jit_model = torch.jit.freeze(tmp_model.eval())

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./lm_cache", ignore_errors=True)

    def test_evaluate_for_casualLM(self):
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="hf-internal-testing/tiny-random-gptj",tokenizer="hf-internal-testing/tiny-random-gptj",dtype=float32',
            tasks=["piqa"],
            limit=20,
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.45)

    def test_evaluate_for_Seq2SeqLM(self):
        results = evaluate(
            model="hf-seq2seq",
            model_args='pretrained="hf-internal-testing/tiny-random-t5",tokenizer="hf-internal-testing/tiny-random-t5",dtype=float32',
            tasks=["piqa"],
            limit=20,
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.60)

    def test_evaluate_for_JitModel(self):
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="hf-internal-testing/tiny-random-gptj",tokenizer="hf-internal-testing/tiny-random-gptj",dtype=float32',
            user_model=self.jit_model,
            tasks=["piqa"],
            limit=20,
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.45)

    def test_lambada_for_llama(self):
        results = evaluate(
                model="hf-causal",
                model_args='pretrained="decapoda-research/llama-7b-hf",tokenizer="decapoda-research/llama-7b-hf",dtype=float32',
                tasks=["lambada_openai", "lambada_standard"],
                limit=20,
                )
        self.assertEqual(results["results"]["lambada_standard"]["acc"], 0.75)
        self.assertEqual(results["results"]["lambada_openai"]["acc"], 0.70)
if __name__ == "__main__":
    unittest.main()
