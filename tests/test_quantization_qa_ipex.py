import os
import sys
import torch
import unittest
from unittest.mock import patch
from intel_extension_for_transformers.optimization.model import OptimizedModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# example test for question-answering quantization with IPEX only for now
SRC_DIRS = [
    os.path.join("../examples/huggingface/pytorch/", dirname)
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
            --cores_per_instance 16
            --num_of_instance 1
            """.split()

        with patch.object(sys, "argv", test_args):
            run_qa.main()


if __name__ == "__main__":
    unittest.main()
