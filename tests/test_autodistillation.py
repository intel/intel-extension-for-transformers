import os
import shutil
import torch.utils.data as data
import unittest
from nlp_toolkit import (
    AutoDistillationConfig,
    FlashDistillationConfig,
    metrics,
    NLPTrainer,
)
from transformers import (
    AutoModelForPreTraining,
    AutoTokenizer
)

os.environ["WANDB_DISABLED"] = "true"


class DummyDataset(data.Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.sequence_a = "NLP-toolkit is based in SH"
        self.sequence_b = "Where is NLP-toolkit based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b)
        self.encoded_dict['labels'] = [-100] * len(self.encoded_dict['input_ids'])
        self.encoded_dict['labels'][1] = 17953
        self.encoded_dict['next_sentence_label'] = 0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.encoded_dict


class TestAutoDistillation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForPreTraining.from_pretrained(
            'google/mobilebert-uncased'
        )
        self.teacher_model = AutoModelForPreTraining.from_pretrained(
            'bert-large-uncased'
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

    def test_fx_model_distil(self):
        autodistillation_config =\
          AutoDistillationConfig(
            search_space={'hidden_size': [128, 256]},
            metrics=[metrics.Metric(name="eval_loss", greater_is_better=False)],
            knowledge_transfer=FlashDistillationConfig(
              block_names=['mobilebert.encoder.layer.1'],
              layer_mappings_for_knowledge_transfer=[
                [('mobilebert.encoder.layer.1.output',
                  'bert.encoder.layer.1.output')]
                ],
              train_steps=[3]),
            regular_distillation=FlashDistillationConfig(
              layer_mappings_for_knowledge_transfer=[
                [('cls', '0', 'cls', '0')]
                ],
              loss_types=[['KL']],
              add_origin_loss=[True],
              train_steps=[5]
          ),
        )
        best_model_archs = self.trainer.autodistillation(
            autodistillation_config,
            self.teacher_model,
            model_cls=AutoModelForPreTraining
        )
        # check best model architectures
        self.assertTrue(len(best_model_archs) > 0)

if __name__ == "__main__":
    unittest.main()
