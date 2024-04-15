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
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from packaging.version import Version

OPTIMUM114_VERSION = Version("1.14.0")
PYTHON_VERSION = Version(platform.python_version())


@unittest.skipIf(
    PYTHON_VERSION.release < Version("3.9.0").release,
    "Please use Python 3.9 or higher version for lm-eval",
)

class TestLmEvaluationHarness(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        cmd = 'pip install protobuf==4.24.4 && pip install wandb'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        p.communicate()
        output,error = p.communicate()
        print(error.decode())
        print(output.decode())
        if os.path.exists("./evaluation_results.json"):
            os.remove("./evaluation_results.json")
        if os.path.exists("./include_path.json"):
            os.remove("./include_path.json")
    @classmethod
    def tearDownClass(self):
        cmd = 'pip uninstall wandb -y && pip install protobuf==4.25.3'
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, shell=True) # nosec
        output,error = p.communicate()
        print(error.decode())
        print(output.decode())
        shutil.rmtree("./lm_cache", ignore_errors=True)
        shutil.rmtree("./t5", ignore_errors=True)
        shutil.rmtree("./t5-past", ignore_errors=True)
        shutil.rmtree("./gptj", ignore_errors=True)
        shutil.rmtree("./gptj-past", ignore_errors=True)
        shutil.rmtree("./wandb", ignore_errors=True)
        if os.path.exists("./evaluation_results.json"):
            os.remove("./evaluation_results.json")
        if os.path.exists("./include_path.json"):
            os.remove("./include_path.json")

    def test_evaluate_for_CasualLM(self):
        from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
        args = LMEvalParser(model="hf",
                            model_args="pretrained=hf-internal-testing/tiny-random-gptj,dtype=float32",
                            tasks="piqa",
                            device="cpu",
                            batch_size=1,
                            limit=5,
                            trust_remote_code=True,
                            verbosity="DEBUG",
                            include_path="./include_path.json"
                            )
        results = evaluate(args)
        self.assertEqual(results["results"]["piqa"]["acc,none"], 0.4)

    def test_evaluate_for_CasualLM_with_wandb_args(self):
        from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
        args = LMEvalParser(model="hf",
                            model_args="pretrained=hf-internal-testing/tiny-random-gptj,dtype=float32",
                            tasks="piqa",
                            device="cpu",
                            batch_size=1,
                            limit=5,
                            wandb_args="project=test-project,name=test-run,mode=offline"
                            )
        results = evaluate(args)
        self.assertEqual(results["results"]["piqa"]["acc,none"], 0.4)

    def test_evaluate_for_CasualLM_Predict_Only(self):
        from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
        args = LMEvalParser(model="hf",
                            model_args="pretrained=hf-internal-testing/tiny-random-gptj,dtype=float32",
                            tasks="piqa",
                            device="cpu",
                            batch_size=1,
                            limit=5,
                            predict_only=True
                            )

    def test_evaluate_for_NS(self):
        from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
        args = LMEvalParser(model="hf",
                            model_args="pretrained=facebook/opt-125m,dtype=float32,model_format=neural_speed",
                            tasks="piqa",
                            device="cpu",
                            batch_size=1,
                            limit=5)
        results = evaluate(args)
        self.assertEqual(results["results"]["piqa"]["acc,none"], 0.6)

    def test_evaluate_for_user_model(self):
        from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
        user_model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gptj")
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gptj")
        args = LMEvalParser(model="hf",
                            user_model=user_model,
                            tokenizer=tokenizer,
                            tasks="piqa",
                            device="cpu",
                            batch_size=1,
                            limit=5,
                            output_path = "./evaluation_results.json",
                            log_samples=True
                            )
        results = evaluate(args)
        self.assertEqual(results["results"]["piqa"]["acc,none"], 0.4)

    def test_evaluate_for_Seq2SeqLM(self):
        from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
        args = LMEvalParser(model="hf",
                            model_args="pretrained=hf-internal-testing/tiny-random-t5,dtype=float32",
                            tasks="piqa",
                            device="cpu",
                            batch_size=1,
                            limit=5)
        results = evaluate(args)
        self.assertEqual(results["results"]["piqa"]["acc,none"], 0.6)

    def test_cnn_daily(self):
        from intel_extension_for_transformers.transformers.llm.evaluation.hf_eval import (
            summarization_evaluate,
        )

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
        cmd = "optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 --task text2text-generation-with-past t5-past/"
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)  # nosec
        p.communicate()
        from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
        args = LMEvalParser(model="hf",
                            model_args="pretrained=./t5-past,dtype=float32,model_format=onnx",
                            tasks="piqa",
                            device="cpu",
                            batch_size=1,
                            limit=5)
        results = evaluate(args)
        self.assertEqual(results["results"]["piqa"]["acc,none"], 0.6)

        # test evaluate encoder_model + decoder_model + decoder_with_past_model
        merged_model_path = "./t5-past/decoder_model_merged.onnx"
        if os.path.exists(merged_model_path):
            os.remove(merged_model_path)
            args = LMEvalParser(model="hf",
                                model_args="pretrained=./t5-past,dtype=float32,model_format=onnx",
                                tasks="piqa",
                                device="cpu",
                                batch_size=1,
                                limit=5)
            results = evaluate(args)
            self.assertEqual(results["results"]["piqa"]["acc,none"], 0.6)

        # test evaluate encoder_model + decoder_model
        cmd = "optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 --task text2text-generation t5/"
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)  # nosec
        p.communicate()
        args = LMEvalParser(model="hf",
                            model_args="pretrained=./t5,dtype=float32,model_format=onnx",
                            tasks="piqa",
                            device="cpu",
                            batch_size=1,
                            limit=5)
        results = evaluate(args)
        self.assertEqual(results["results"]["piqa"]["acc,none"], 0.6)

    def test_evaluate_for_ort_CasualLM(self):
        if Version(optimum.version.__version__) >= OPTIMUM114_VERSION:
            cmd = "optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation-with-past --legacy gptj-past/"
        else:
            cmd = "optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation-with-past gptj-past/"
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)  # nosec
        p.communicate()

        # test evaluate decoder_model_merged
        from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
        args = LMEvalParser(model="hf",
                            model_args="pretrained=./gptj-past,dtype=float32,model_format=onnx",
                            tasks="piqa",
                            device="cpu",
                            batch_size=1,
                            limit=5)
        results = evaluate(args)
        self.assertEqual(results["results"]["piqa"]["acc,none"], 0.4)

        # test evaluate decoder_model + decoder_with_past_model
        merged_model_path = "./gptj-past/decoder_model_merged.onnx"
        if os.path.exists(merged_model_path):
            os.remove(merged_model_path)
            args = LMEvalParser(model="hf",
                                model_args="pretrained=./gptj-past,dtype=float32,model_format=onnx",
                                tasks="piqa",
                                device="cpu",
                                batch_size=1,
                                limit=5)
            results = evaluate(args)
            self.assertEqual(results["results"]["piqa"]["acc,none"], 0.4)

        # test evaluate decoder_model
        if Version(optimum.version.__version__) >= OPTIMUM114_VERSION:
            cmd = "optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation --legacy gptj/"
        else:
            cmd = "optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation gptj/"
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)  # nosec
        p.communicate()
        args = LMEvalParser(model="hf",
                            model_args="pretrained=./gptj,dtype=float32,model_format=onnx",
                            tasks="piqa",
                            device="cpu",
                            batch_size=1,
                            limit=5)
        results = evaluate(args)
        self.assertEqual(results["results"]["piqa"]["acc,none"], 0.4)

        # test evaluate model exported with optimum >= 1.14.0
        if Version(optimum.version.__version__) >= OPTIMUM114_VERSION:
            cmd = "optimum-cli export onnx --model hf-internal-testing/tiny-random-gptj --task text-generation-with-past gptj-past/"
            p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)  # nosec
            p.communicate()
            args = LMEvalParser(model="hf",
                                model_args="pretrained=./gptj-past,dtype=float32,model_format=onnx",
                                tasks="piqa",
                                device="cpu",
                                batch_size=1,
                                limit=5)
            results = evaluate(args)
            self.assertEqual(results["results"]["piqa"]["acc,none"], 0.4)


if __name__ == "__main__":
    unittest.main()
