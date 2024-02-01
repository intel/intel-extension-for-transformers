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
import sys
import torch
import unittest
from unittest.mock import patch
from intel_extension_for_transformers.transformers.modeling import OptimizedModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# example test for question-answering quantization with IPEX only for now
EXAMPLE_PATH="../../examples/huggingface/pytorch/"
if not os.path.exists(EXAMPLE_PATH):
    EXAMPLE_PATH="../examples/huggingface/pytorch/"
SRC_DIRS = [
    os.path.join(EXAMPLE_PATH, dirname)
    for dirname in [
        "question-answering/quantization/",
    ]
]
sys.path.extend(SRC_DIRS)

if SRC_DIRS is not None:
    import run_qa

class TestExamples(unittest.TestCase):
    def test_run_qa_ipex(self):
        test_args = f"""
            run_qa.py
            --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad
            --dataset_name squad
            --tune
            --quantization_approach PostTrainingStatic
            --do_train
            --do_eval
            --max_eval_samples 100
            --max_train_samples 50
            --output_dir ./tmp/squad_output
            --overwrite_output_dir
            --framework ipex
            """.split()

        with patch.object(sys, "argv", test_args):
            run_qa.main()
            int8_model = OptimizedModel.from_pretrained("./tmp/squad_output")
            self.assertTrue(isinstance(int8_model, torch.jit.ScriptModule))

        test_args = f"""
            run_qa.py
            --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad
            --dataset_name squad
            --quantization_approach PostTrainingStatic
            --do_train
            --do_eval
            --max_eval_samples 100
            --max_train_samples 50
            --output_dir ./tmp/squad_output
            --overwrite_output_dir
            --framework ipex
            --benchmark_only
            --cores_per_instance 8
            --num_of_instance 1
            """.split()

        with patch.object(sys, "argv", test_args):
            run_qa.main()


if __name__ == "__main__":
    unittest.main()
