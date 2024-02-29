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
import torch
import optimum.version
import platform
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from packaging.version import Version
OPTIMUM114_VERSION = Version("1.14.0")
PYTHON_VERSION = Version(platform.python_version())

@unittest.skipIf(PYTHON_VERSION.release < Version("3.9.0").release,
    "Please use Python 3.9 or higher version for lm-eval")
class TestLmEvaluationHarness(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.clm_model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-gptj",
            torchscript=True
        )
        tmp_model = torch.jit.trace(
            self.clm_model, self.clm_model.dummy_inputs["input_ids"]
        )
        self.jit_model = torch.jit.freeze(tmp_model.eval())
        cmd = 'pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@cc9778fbe4fa1a709be2abed9deb6180fd40e7e2'
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
        cmd = 'pip uninstall lm_eval -y'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()


    def test_evaluate_for_CasualLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="hf-internal-testing/tiny-random-gptj",tokenizer="hf-internal-testing/tiny-random-gptj",dtype=float32',
            tasks=["piqa"],
            limit=5,
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.6)

    def test_evaluate_for_Seq2SeqLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        results = evaluate(
            model="hf-seq2seq",
            model_args='pretrained="hf-internal-testing/tiny-random-t5",tokenizer="hf-internal-testing/tiny-random-t5",dtype=float32',
            tasks=["piqa"],
            limit=5,
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 1.0)

    def test_evaluate_for_JitModel(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="hf-internal-testing/tiny-random-gptj",tokenizer="hf-internal-testing/tiny-random-gptj",dtype=float32',
            user_model=self.jit_model,
            tasks=["piqa"],
            limit=5,
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.6)

    def test_cnn_daily(self):
        from intel_extension_for_transformers.llm.evaluation.hf_eval import summarization_evaluate
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        results = summarization_evaluate(
           model=model,
           tokenizer_name="facebook/opt-125m",
           batch_size=1,
           limit=5,
        )
        self.assertEqual(results["rouge2"], 18.0431)
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        results = summarization_evaluate(
            model=model, tokenizer_name="t5-small", batch_size=1, limit=5
        )
        self.assertEqual(results["rouge2"], 9.5858)

    def test_evaluate_for_ort_Seq2SeqLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 --task text2text-generation-with-past t5-past/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()

        # test evaluate encoder_model + decoder_model_merged
        results = evaluate(
            model="hf-seq2seq",
            model_args='pretrained="./t5-past",tokenizer="./t5-past",dtype=float32',
            tasks=["piqa"],
            limit=5,
            model_format="onnx"
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 1.0)

        # test evaluate encoder_model + decoder_model + decoder_with_past_model
        merged_model_path = "./t5-past/decoder_model_merged.onnx"
        if os.path.exists(merged_model_path):
            os.remove(merged_model_path)
            results = evaluate(
                model="hf-seq2seq",
                model_args='pretrained="./t5-past",tokenizer="./t5-past",dtype=float32',
                tasks=["piqa"],
                limit=5,
                model_format="onnx"
            )
            self.assertEqual(results["results"]["piqa"]["acc"], 1.0)

        # test evaluate encoder_model + decoder_model
        cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 --task text2text-generation t5/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
        results = evaluate(
            model="hf-seq2seq",
            model_args='pretrained="./t5",tokenizer="./t5",dtype=float32',
            tasks=["piqa"],
            limit=5,
            model_format="onnx"
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 1.0)

    def test_evaluate_for_ort_CasualLM(self):
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
        if Version(optimum.version.__version__) >= OPTIMUM114_VERSION:
            cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation-with-past --legacy gptj-past/'
        else:
            cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation-with-past gptj-past/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()

        # test evaluate decoder_model_merged
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="./gptj-past",tokenizer="./gptj-past",dtype=float32',
            tasks=["piqa"],
            limit=5,
            model_format="onnx"
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.6)

        # test evaluate decoder_model + decoder_with_past_model
        merged_model_path = "./gptj-past/decoder_model_merged.onnx"
        if os.path.exists(merged_model_path):
            os.remove(merged_model_path)
            results = evaluate(
                model="hf-causal",
                model_args='pretrained="./gptj-past",tokenizer="./gptj-past",dtype=float32',
                tasks=["piqa"],
                limit=5,
                model_format="onnx"
            )
            self.assertEqual(results["results"]["piqa"]["acc"], 0.6)

        # test evaluate decoder_model
        if Version(optimum.version.__version__) >= OPTIMUM114_VERSION:
            cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation --legacy gptj/'
        else:
            cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation gptj/'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
        results = evaluate(
            model="hf-causal",
            model_args='pretrained="./gptj",tokenizer="./gptj",dtype=float32',
            tasks=["piqa"],
            limit=5,
            model_format="onnx"
        )
        self.assertEqual(results["results"]["piqa"]["acc"], 0.6)

        # test evaluate model exported with optimum >= 1.14.0
        if Version(optimum.version.__version__) >= OPTIMUM114_VERSION:
            cmd = 'optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation-with-past gptj-past/'
            p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, shell=True) # nosec
            p.communicate()

            results = evaluate(
                model="hf-causal",
                model_args='pretrained="./gptj-past",tokenizer="./gptj-past",dtype=float32',
                tasks=["piqa"],
                limit=5,
                model_format="onnx"
            )
            self.assertEqual(results["results"]["piqa"]["acc"], 0.6)

if __name__ == "__main__":
    unittest.main()
