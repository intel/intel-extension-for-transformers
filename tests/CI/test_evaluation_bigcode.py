import os
import shutil
import unittest
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestLmEvaluationHarness(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.starcoder = AutoModelForCausalLM.from_pretrained("bigcode/tiny_starcoder_py") 
        cmd = 'pip install git+https://github.com/bigcode-project/bigcode-evaluation-harness.git@0d84db85f9ff971fa23a187a3347b7f59af288dc'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()


    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./evaluation_results.json", ignore_errors=True)
        cmd = 'pip uninstall lm_eval -y'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()

    def test_bigcode_lm_eval(self):
        from intel_extension_for_transformers.llm.evaluation.lm_code_eval import evaluate as bigcode_evaluate
        starcoder_tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py")
        class bigcode_args:
            limit=1
            n_samples=20
            allow_code_execution=True
            do_sample=True
            prefix=""
            generation_only=False
            postprocess=False
            save_references=False
            save_generations=False
            instruction_tokens=None
            save_generations_path=None
            load_generations_path=False
            metric_output_path="./evaluation_results.json"
            seed=0
            temperature=0.8
            max_length_generation=512
            top_p=0.95
            top_k=0
            do_sample=True
            batch_size=10

        args = bigcode_args()
        results = bigcode_evaluate(
                model=self.starcoder,
                tokenizer=starcoder_tokenizer,
                batch_size=1,
                tasks="humaneval",
                args=args
                )
        self.assertEqual(results["humaneval"]["pass@1"], 0.0)



if __name__ == "__main__":
    unittest.main()
