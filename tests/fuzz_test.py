from intel_extension_for_transformers.transformers.trainer import NLPTrainer
from intel_extension_for_transformers.llm.runtime.compile import compile
from intel_extension_for_transformers.transformers import (
    metrics,
    objectives,
    QuantizationConfig,
)
import atheris
import sys
import torch.utils.data as torch_data
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
class DummyDataset(torch_data.Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b)
        self.encoded_dict['labels'] = 1

    def __len__(self):
        return 10

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        if index < 10:
            return self.encoded_dict


def TestOneInput(data):
    print("test nlp trainer fuzz eval dataset start")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
        dummy_dataset = DummyDataset()
        trainer = NLPTrainer(
            model=model,
            train_dataset=dummy_dataset,
            eval_dataset=data
        )
    except AssertionError:
        pass
    print("test nlp trainer fuzz eval dataset end")

    print("test nlp trainer fuzz model start")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
        dummy_dataset = DummyDataset()
        trainer = NLPTrainer(
            model=data,
            train_dataset=dummy_dataset,
            eval_dataset=data
        )
    except AttributeError:
        pass
    print("test nlp trainer fuzz model end")

    print("test nlp trainer fuzz train data start")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
        dummy_dataset = DummyDataset()
        trainer = NLPTrainer(
            model=model,
            train_dataset=data,
            eval_dataset=dummy_dataset
        )
    except AssertionError:
        pass
    print("test nlp trainer fuzz train data end")

    print("test nlp trainer fuzz config approach start")
    try:
        tune_metric = metrics.Metric(
            name="eval_loss", greater_is_better=False, is_relative=False, criterion=0.5
        )
        quantization_config = QuantizationConfig(
                approach=data,
                metrics=[tune_metric],
                objectives=[objectives.performance]
            )
    except AssertionError:
        pass
    print("test nlp trainer fuzz config approach end")

    print("test nlp trainer fuzz config metrics start")
    try:
        quantization_config = QuantizationConfig(
                approach="post_training_static_quant",
                metrics=[data],
                objectives=[objectives.performance]
            )
    except AssertionError:
        pass
    print("test nlp trainer fuzz config metrics end")

    print("test nlp trainer fuzz config objectives start")
    try:
        tune_metric = metrics.Metric(
            name="eval_loss", greater_is_better=False, is_relative=False, criterion=0.5
        )
        quantization_config = QuantizationConfig(
                approach="post_training_static_quant",
                metrics=[tune_metric],
                objectives=[data]
            )
    except AssertionError:
        pass
    print("test nlp trainer fuzz config objectives end")

    print("test engine fuzz compile start")
    try:
        model = compile(data)
    except AssertionError:
        pass
    print("test engine fuzz compile end")


atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()