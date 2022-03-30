import shutil
import unittest

import torch
import torch.utils.data as data

from transformers import DistilBertForSequenceClassification
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from nlp_toolkit import NLPTrainer
from nlp_toolkit import OptimizedModel
from nlp_toolkit import QuantizationMode



class DummyDataset(data.Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.sequence_a = "NLP-toolkit is based in SH"
        self.sequence_b = "Where is NLP-toolkit based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b)
        self.encoded_dict['label'] = 1

    def __len__(self):
        return 1

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.encoded_dict


class TestQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased'
        )
        self.dummy_dataset = DummyDataset()
        self.trainer = NLPTrainer(
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./quantized_model', ignore_errors=True)

    def test_fx_model_quant(self):
        fp32_output = self.trainer.predict(self.dummy_dataset).predictions
        for mode in QuantizationMode:
            print("Quantization approach:", mode.value)
            self.trainer = NLPTrainer(
                model=self.model,
                train_dataset=self.dummy_dataset,
                eval_dataset=self.dummy_dataset,
            )
            self.trainer.provider_arguments = {
                "quantization":{
                    "approach": mode.name,
                    "performance_only": True,
                    "metrics": {"metrics":["eval_samples_per_second"]},
                }
            }
            quantized_model = self.trainer.quantize()
            # By default, model will be saved in tmp_trainer dir.
            self.trainer.save_model('./quantized_model')
            output_1 = self.trainer.predict(self.dummy_dataset).predictions
            loaded_model = OptimizedModel.from_pretrained(
                './quantized_model',
            )
            self.trainer.model = loaded_model
            output_2 = self.trainer.predict(self.dummy_dataset).predictions
            # check quantized model
            self.assertTrue((fp32_output != output_1).any())
            # check loaded model
            self.assertTrue((output_1 == output_2).all())


if __name__ == "__main__":
    unittest.main()
