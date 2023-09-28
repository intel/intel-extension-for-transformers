import os
import shutil
from sre_parse import Tokenizer
import unittest
import subprocess
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer
)
from intel_extension_for_transformers.transformers.utils import LazyImport
torch = LazyImport("torch")


class TestLmEvaluationHarness(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.clm_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True
        )
        self.clm_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        self.seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        self.seq2seq_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        tmp_model = torch.jit.trace(
            self.clm_model, self.clm_model.dummy_inputs["input_ids"]
        )
        self.jit_model = torch.jit.freeze(tmp_model.eval())
        cmd = 'pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@2c18e367c6ded428863cd1fd4cf9558ca49d68dc'
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


    def test_evaluate_for_CasualLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import HFCausalLM, evaluate
        clm_model = HFCausalLM(model=self.clm_model, tokenizer=self.clm_tokenizer)
        results = evaluate(
            model=clm_model,
            tasks=["piqa"],
            limit=20,
            no_cache=True
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.7)

    def test_evaluate_for_Seq2SeqLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import HFSeq2SeqLM, evaluate
        seq2seq_model = HFSeq2SeqLM(model=self.seq2seq_model, tokenizer=self.seq2seq_tokenizer)
        results = evaluate(
            model=seq2seq_model,
            tasks=["piqa"],
            limit=20,
            no_cache=True
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.60)

    def test_evaluate_for_JitModel(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import HFCausalLM, evaluate
        jit_clm_model = HFCausalLM(model=self.jit_model, tokenizer=self.clm_tokenizer)
        results = evaluate(
            model=jit_clm_model,
            tasks=["piqa"],
            limit=20,
            no_cache=True
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.7)

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
        self.assertEqual(results["rouge2"], 9.6795)
    
    def test_evaluate_for_ort_Seq2SeqLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import HFSeq2SeqLM, evaluate
        cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 --task text2text-generation-with-past t5-past/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()

        # test evaluate encoder_model + decoder_model_merged
        tokenizer = AutoTokenizer.from_pretrained("./t5-past")
        model = HFSeq2SeqLM(model_name_or_path="./t5-past", tokenizer=tokenizer, model_format="onnx")
        results = evaluate(
            model=model,
            tasks=["piqa"],
            limit=20,
            no_cache=True
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.60)

        # test evaluate encoder_model + decoder_model + decoder_with_past_model
        merged_model_path = "./t5-past/decoder_model_merged.onnx"
        model = HFSeq2SeqLM(model_name_or_path="./t5-past", tokenizer=tokenizer, model_format="onnx")
        if os.path.exists(merged_model_path):
            os.remove(merged_model_path)
            results = evaluate(
                model=model,
                tasks=["piqa"],
                limit=20,
                no_cache=True
            )
            self.assertEqual(results["results"]["piqa"]["acc"], 0.60)

        # test evaluate encoder_model + decoder_model
        cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 --task text2text-generation t5/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
        tokenizer = AutoTokenizer.from_pretrained("./t5")
        model = HFSeq2SeqLM(model_name_or_path="./t5", tokenizer=tokenizer, model_format="onnx")
        results = evaluate(
            model=model,
            tasks=["piqa"],
            limit=20,
            no_cache=True
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.60)

    def test_evaluate_for_ort_casualLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import HFCausalLM, evaluate
        cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation-with-past gptj-past/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()

        # test evaluate 
        tokenizer = AutoTokenizer.from_pretrained("./gptj-past")
        model = HFCausalLM(model_name_or_path="./gptj-past", tokenizer=tokenizer, model_format="onnx")      
        results = evaluate(
            model= model,
            tasks=["piqa"],
            limit=20,
            no_cache=True
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.45)

        # test evaluate decoder_model + decoder_with_past_model
        merged_model_path = "./gptj-past/decoder_model_merged.onnx"
        if os.path.exists(merged_model_path):
            os.remove(merged_model_path)
            model = HFCausalLM(model_name_or_path="./gptj-past", tokenizer=tokenizer, model_format="onnx")   
            results = evaluate(
                model= model,
                tasks=["piqa"],
                limit=20,
                no_cache=True
            )
            self.assertEqual(results["results"]["piqa"]["acc"], 0.45)

        # test evaluate decoder_model
        cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation gptj/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
        model = HFCausalLM(model_name_or_path="./gptj", tokenizer=tokenizer, model_format="onnx")   
        results = evaluate(
            model= model,
            tasks=["piqa"],
            limit=20,
            no_cache=True
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.45)


    def test_tokenizer_for_llama(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import HFCausalLM, evaluate
        cmd = 'optimum-cli export onnx --model decapoda-research/llama-7b-hf --task text-generation llama/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
        model = HFCausalLM(model_name_or_path="./llama", model_format="onnx")   
        results = evaluate(
            model= model,
            tasks=["piqa"],
            limit=20,
            no_cache=True
        )
        self.assertEqual(results["results"]["lambada_openai"]["acc"], 0.70)

if __name__ == "__main__":
    unittest.main()
