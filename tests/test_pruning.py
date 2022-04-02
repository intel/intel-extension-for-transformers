import copy
import shutil
import torch.utils.data as data
import unittest
from nlp_toolkit import (
    Metric,
    NLPTrainer,
    OptimizedModel,
    PruningConfig,
    PruningMode,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)



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


class TestPruning(unittest.TestCase):
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
        shutil.rmtree('./pruned_model', ignore_errors=True)

    def test_fx_model_prune(self):
        origin_weight = copy.deepcopy(self.model.classifier.weight)
        for mode in PruningMode:
            # not supported yet
            if mode.value != "basic_magnitude":
                continue
            print("Pruning mode:", mode.value)
            self.trainer = NLPTrainer(
                model=self.model,
                train_dataset=self.dummy_dataset,
                eval_dataset=self.dummy_dataset,
            )
            metric = Metric(name="eval_loss")
            pruning_conf = PruningConfig(
                approach=mode.name, target_sparsity_ratio=0.9, metrics=[metric]
            )
            self.trainer.provider_config.pruning = pruning_conf
            pruned_model = self.trainer.prune()
            pruned_model.report_sparsity()
            # By default, model will be saved in tmp_trainer dir.
            self.trainer.save_model('./pruned_model')
            loaded_model = OptimizedModel.from_pretrained(
                './pruned_model',
            )
            pruned_weight = copy.deepcopy(pruned_model.model.classifier.weight)
            loaded_weight = copy.deepcopy(loaded_model.classifier.weight)
            # check pruned model
            self.assertTrue((pruned_weight != origin_weight).any())
            # check loaded model
            self.assertTrue((pruned_weight == loaded_weight).all())


if __name__ == "__main__":
    unittest.main()
