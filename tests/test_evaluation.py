import os
import shutil
import unittest

import torch
from intel_extension_for_transformers.evaluation import evaluate
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM


class TestLmEvaluationHarness(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.clm_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True
        )
        self.seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        tmp_model = torch.jit.trace(
            self.clm_model, self.clm_model.dummy_inputs["input_ids"]
        )
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
        self.assertEqual(results["results"]["piqa"]["acc"], 0.65)

    def test_lambada_for_llama(self):
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="decapoda-research/llama-7b-hf",tokenizer="decapoda-research/llama-7b-hf",dtype=float32',
            tasks=["lambada_openai", "lambada_standard"],
            limit=20,
        )
        self.assertEqual(results["results"]["lambada_standard"]["acc"], 0.75)
        self.assertEqual(results["results"]["lambada_openai"]["acc"], 0.70)

    def test_cnn_daily(self):
        from intel_extension_for_transformers.evaluation import \
            summarization_evaluate

        results = summarization_evaluate(
           model=self.clm_model,
           tokenizer_name="facebook/opt-125m",
           batch_size=1,
           limit=5,
        )
        self.assertEqual(results["rouge2"], 10.6232)
        results = summarization_evaluate(
            model=self.seq2seq_model, tokenizer_name="t5-small", batch_size=1, limit=5
        )
        self.assertEqual(results["rouge2"], 13.4312)


if __name__ == "__main__":
    unittest.main()
