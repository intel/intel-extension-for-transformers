import copy
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
from nlp_toolkit import DistillationMode



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
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased'
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
        shutil.rmtree('./distilled_model', ignore_errors=True)

    def test_fx_model_distil(self):
        origin_weight = copy.deepcopy(self.model.classifier.weight)
        for mode in DistillationMode:
            print("Distillation approach:", mode.value)
            self.trainer = NLPTrainer(
                model=self.model,
                train_dataset=self.dummy_dataset,
                eval_dataset=self.dummy_dataset,
            )
            self.trainer.provider_arguments = {
                "distillation":{
                    "metrics": {"metrics":["eval_samples_per_second"]},
                }
            }
            distilled_model = self.trainer.distill(self.teacher_model)
            # By default, model will be saved in tmp_trainer dir.
            self.trainer.save_model('./distilled_model')
            loaded_model = OptimizedModel.from_pretrained(
                './distilled_model',
            )
            distilled_weight = copy.deepcopy(distilled_model.model.classifier.weight)
            loaded_weight = copy.deepcopy(loaded_model.classifier.weight)
            # check distilled model
            self.assertTrue((distilled_weight != origin_weight).any())
            # check loaded model
            self.assertTrue((distilled_weight == loaded_weight).all())


if __name__ == "__main__":
    unittest.main()
