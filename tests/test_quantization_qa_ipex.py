import json
import os
import sys
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# example test for question-answering quantization with IPEX only for now
SRC_DIRS = [
    os.path.join("../examples/optimization/pytorch/huggingface/", dirname)
    for dirname in [
        "question-answering/quantization/inc/",
    ]
]
sys.path.extend(SRC_DIRS)

if SRC_DIRS is not None:
    import run_qa

    def get_results(output_dir):
        results = {}
        path = os.path.join(output_dir, "best_configure.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                results = json.load(f)
        else:
            raise ValueError(f"Can't find {path}.")
        return results


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
            --max_eval_samples 200
            --max_train_samples 50
            --output_dir ./tmp/squad_output
            --overwrite_output_dir
            --framework ipex
            """.split()

        with patch.object(sys, "argv", test_args):
            run_qa.main()
            results = get_results("./tmp/squad_output")


if __name__ == "__main__":
    unittest.main()