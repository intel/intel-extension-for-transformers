import os
import unittest

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from intel_extension_for_transformers.llm.finetuning.data_utils import preprocess_dataset

os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"
MODEL_NAME = "hf-internal-testing/tiny-random-GPTJForCausalLM"

class TestChatDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        class TestArgs:
            train_on_inputs = False
            task = "chat"
            max_seq_length = 512
            max_source_length = 256
            dataset_name = "HuggingFaceH4/ultrachat_200k"

        self.test_args = TestArgs()

        self.sample_datasets = DatasetDict()

        raw_datasets = load_dataset("HuggingFaceH4/ultrachat_200k")
        self.sample_datasets["train"] = raw_datasets["train_sft"].select(range(100))

    def test_process(self):

        raw_datasets, preprocess_fn  = preprocess_dataset(self.sample_datasets, self.tokenizer,
                self.test_args, self.test_args)

        column_names = list(raw_datasets["train"].features)

        tokenized_datasets = raw_datasets.map(
                preprocess_fn,
                batched=True,
                remove_columns=column_names)

        self.assertTrue(isinstance(tokenized_datasets, DatasetDict))


class TestCompletionDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        class TestArgs:
            train_on_inputs = False
            task = "completion"
            max_seq_length = 512
            max_source_length = 256
            dataset_name = "tatsu-lab/alpaca"

        self.test_args = TestArgs()

        self.sample_datasets = DatasetDict()

        raw_datasets = load_dataset("tatsu-lab/alpaca")
        self.sample_datasets["train"] = raw_datasets["train"].select(range(100))

    def test_process(self):

        raw_datasets, preprocess_fn  = preprocess_dataset(self.sample_datasets, self.tokenizer,
                self.test_args, self.test_args)

        column_names = list(raw_datasets["train"].features)

        tokenized_datasets = raw_datasets.map(
                preprocess_fn,
                batched=True,
                remove_columns=column_names)

        self.assertTrue(isinstance(tokenized_datasets, DatasetDict))


class TestSlimOrcaDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        class TestArgs:
            train_on_inputs = False
            task = "SlimOrca"
            max_seq_length = 512
            max_source_length = 256
            dataset_name = "Open-Orca/SlimOrca"

        self.test_args = TestArgs()

        self.sample_datasets = DatasetDict()

        raw_datasets = load_dataset("Open-Orca/SlimOrca")
        self.sample_datasets["train"] = raw_datasets["train"].select(range(100))

    def test_process(self):

        raw_datasets, preprocess_fn  = preprocess_dataset(self.sample_datasets, self.tokenizer,
                self.test_args, self.test_args)

        column_names = list(raw_datasets["train"].features)

        tokenized_datasets = raw_datasets.map(
                preprocess_fn,
                batched=True,
                remove_columns=column_names)

        self.assertTrue(isinstance(tokenized_datasets, DatasetDict))


class TestSummarizationDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        class TestArgs:
            train_on_inputs = False
            task = "summarization"
            max_seq_length = 512
            max_source_length = 256
            dataset_name = "cnn_dailymail"

        self.test_args = TestArgs()

        self.sample_datasets = DatasetDict()

        raw_datasets = load_dataset("cnn_dailymail", "3.0.0")
        self.sample_datasets["train"] = raw_datasets["train"].select(range(100))

    def test_process(self):

        raw_datasets, preprocess_fn  = preprocess_dataset(self.sample_datasets, self.tokenizer,
                self.test_args, self.test_args)

        column_names = list(raw_datasets["train"].features)

        tokenized_datasets = raw_datasets.map(
                preprocess_fn,
                batched=True,
                remove_columns=column_names)

        self.assertTrue(isinstance(tokenized_datasets, DatasetDict))


class TestDPODataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        class TestArgs:
            train_on_inputs = False
            task = "chat"
            max_seq_length = 512
            max_source_length = 256
            dataset_name = "Intel/orca_dpo_pairs"

        self.test_args = TestArgs()

        self.sample_datasets = DatasetDict()

        raw_datasets = load_dataset("Intel/orca_dpo_pairs")
        self.sample_datasets["train"] = raw_datasets["train"].select(range(100))

    def test_process(self):

        raw_datasets, preprocess_fn  = preprocess_dataset(self.sample_datasets, self.tokenizer,
                self.test_args, self.test_args)

        column_names = list(raw_datasets["train"].features)

        tokenized_datasets = raw_datasets.map(
                preprocess_fn,
                batched=True,
                remove_columns=column_names)

        self.assertTrue(isinstance(tokenized_datasets, DatasetDict))


if __name__ == "__main__":
    unittest.main()
