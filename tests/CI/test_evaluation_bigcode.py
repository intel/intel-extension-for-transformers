# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import unittest
import subprocess
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestLmEvaluationHarness(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.starcoder = AutoModelForCausalLM.from_pretrained("bigcode/tiny_starcoder_py")
        cmd = 'pip install git+https://github.com/bigcode-project/bigcode-evaluation-harness.git@094c7cc197d13a53c19303865e2056f1c7488ac1'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./evaluation_results.json", ignore_errors=True)
        cmd = 'pip uninstall bigcode_eval -y'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
    
    @unittest.skipIf(sys.version_info >= (3, 11),
            "bigcode-lmeval not support python version higher than 3.10")
    def test_bigcode_lm_eval(self):
        from intel_extension_for_transformers.transformers.llm.evaluation.bigcode_eval import evaluate as bigcode_evaluate
        starcoder_tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py")
        class bigcode_args:
            limit=5
            limit_start=0
            n_samples=5
            allow_code_execution=True
            do_sample=True
            prefix=""
            generation_only=False
            postprocess=False
            save_references=False
            save_generations=False
            instruction_tokens=None
            save_generations_path="generations.json"
            load_generations_path=False
            metric_output_path="./evaluation_results.json"
            seed=0
            temperature=0.2
            max_length_generation=512
            top_p=0.95
            top_k=0
            do_sample=True
            batch_size=1
            check_references=False
            save_every_k_tasks=-1
            load_data_path=None
            modeltype="causal"
            max_memory_per_gpu=None
            eos="<|endoftext|>"
            load_generations_intermediate_paths=None
            left_padding=False
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
