import os
import shutil
import unittest
import subprocess
import torch
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
        cmd = 'pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@83dbfbf6070324f3e5872f63e49d49ff7ef4c9b3'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate


    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./lm_cache", ignore_errors=True)
        shutil.rmtree("./t5", ignore_errors=True)
        shutil.rmtree("./t5-past", ignore_errors=True)
        shutil.rmtree("./gptj", ignore_errors=True)
        shutil.rmtree("./gptj-past", ignore_errors=True)
        shutil.rmtree("./evaluation_results.json", ignore_errors=True)
        shutil.rmtree("./llama", ignore_errors=True)
        cmd = 'pip uninstall lm_eval -y'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()


    def test_evaluate_for_casualLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="hf-internal-testing/tiny-random-gptj",tokenizer="hf-internal-testing/tiny-random-gptj",dtype=float32',
            tasks=["piqa"],
            limit=20,
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.45)

    def test_evaluate_for_Seq2SeqLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        results = evaluate(
            model="hf-seq2seq",
            model_args='pretrained="hf-internal-testing/tiny-random-t5",tokenizer="hf-internal-testing/tiny-random-t5",dtype=float32',
            tasks=["piqa"],
            limit=20,
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.60)

    def test_evaluate_for_JitModel(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="hf-internal-testing/tiny-random-gptj",tokenizer="hf-internal-testing/tiny-random-gptj",dtype=float32',
            user_model=self.jit_model,
            tasks=["piqa"],
            limit=20,
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.65)

    def test_lambada_for_llama(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="decapoda-research/llama-7b-hf",tokenizer="decapoda-research/llama-7b-hf",dtype=float32',
            tasks=["lambada_openai", "lambada_standard"],
            limit=20,
        )
        self.assertEqual(results["results"]["lambada_standard"]["acc"], 0.75)
        self.assertEqual(results["results"]["lambada_openai"]["acc"], 0.70)

    def test_cnn_daily(self):
        from intel_extension_for_transformers.llm.evaluation.hf_eval import summarization_evaluate
        results = summarization_evaluate(
           model=self.clm_model,
           tokenizer_name="facebook/opt-125m",
           batch_size=1,
           limit=5,
        )
        self.assertEqual(results["rouge2"], 18.0431)
        results = summarization_evaluate(
            model=self.seq2seq_model, tokenizer_name="t5-small", batch_size=1, limit=5
        )
        self.assertEqual(results["rouge2"], 14.2213)
    
    def test_evaluate_for_ort_Seq2SeqLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 --task text2text-generation-with-past t5-past/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
        results = evaluate(
            model="hf-seq2seq",
            model_args='pretrained="./t5-past",tokenizer="./t5-past",dtype=float32',
            tasks=["piqa"],
            limit=20,
            model_format="onnx"
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.60)

        cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 --task text2text-generation-with-past t5/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
        results = evaluate(
            model="hf-seq2seq",
            model_args='pretrained="./t5",tokenizer="./t5",dtype=float32',
            tasks=["piqa"],
            limit=20,
            model_format="onnx"
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.60)

    def test_evaluate_for_ort_casualLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation-with-past gptj-past/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="./gptj-past",tokenizer="./gptj-past",dtype=float32',
            tasks=["piqa"],
            limit=20,
            model_format="onnx"
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.45)

        cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation gptj/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="./gptj",tokenizer="./gptj",dtype=float32',
            tasks=["piqa"],
            limit=20,
            model_format="onnx"
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.45)


    def test_tokenizer_for_llama(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        cmd = 'optimum-cli export onnx --model decapoda-research/llama-7b-hf --task text-generation llama/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()

        results = evaluate(
            model="hf-causal",
            model_args='pretrained="./llama",tokenizer="decapoda-research/llama-7b-hf"',
            tasks=["lambada_openai"],
            limit=20,
            model_format="onnx"
        )
        self.assertEqual(results["results"]["lambada_openai"]["acc"], 0.70)

if __name__ == "__main__":
    unittest.main()
