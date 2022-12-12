import os
import shutil
import torch
import torch.utils.data as data
import unittest
from intel_extension_for_transformers.optimization import (
    AutoDistillationConfig,
    FlashDistillationConfig,
    metrics,
)
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
from intel_extension_for_transformers.optimization.utils.utility import distributed_init
from transformers import (
    AutoModelForPreTraining,
    AutoTokenizer,
    TrainingArguments,
)

os.environ["WANDB_DISABLED"] = "true"


def main_worker(rank, world_size, model, teacher_model, dataset):
    try:
        distributed_init("gloo",
                         world_size=world_size,
                         rank=rank,
                         init_method='tcp://127.0.0.1:23456')
    except:
        distributed_init("gloo",
                         world_size=world_size,
                         rank=rank,
                         init_method='tcp://127.0.0.1:12345')

    training_args = TrainingArguments(
        output_dir='tmp_trainer',
        overwrite_output_dir=True,
        no_cuda=True,
        local_rank=rank
    )
    trainer = NLPTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            args=training_args,
    )
    autodistillation_config = AutoDistillationConfig(
        search_space={'hidden_size': [128, 256], 'intermediate_size':[256, 512]},
        search_algorithm='Grid',
        metrics=[metrics.Metric(name="eval_loss", greater_is_better=False)],
        knowledge_transfer=FlashDistillationConfig(
            block_names=['bert.encoder.layer.0'],
            layer_mappings_for_knowledge_transfer=[
                [
                    [('bert.encoder.layer.0.output',)]
                ],
            ],
            train_steps=[3],
            loss_types=[['MSE']],),
        regular_distillation=FlashDistillationConfig(
            layer_mappings_for_knowledge_transfer=[
                [[('cls', '0')]]
            ],
            loss_types=[['KL']],
            add_origin_loss=[True],
            train_steps=[5]
        ),
    )
    best_model_archs = trainer.autodistillation(
        autodistillation_config,
        teacher_model,
        model_cls=AutoModelForPreTraining
    )
    assert len(best_model_archs) > 0, "Expected at least one best model archs."


class DummyDataset(data.Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
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
        self.model = AutoModelForPreTraining.from_pretrained('prajjwal1/bert-tiny')
        self.teacher_model = AutoModelForPreTraining.from_pretrained('bert-base-uncased')
        self.dummy_dataset = DummyDataset()
        self.trainer = NLPTrainer(
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp_trainer', ignore_errors=True)

    def test_autodistillation(self):
        for search_algorithm in ['BO', 'Grid', 'Random']:
            max_trials = 6 if search_algorithm == 'Random' else 3
            autodistillation_config = \
              AutoDistillationConfig(
                search_space={'hidden_size': [128, 256], 'intermediate_size':[256, 512]},
                search_algorithm=search_algorithm,
                max_trials=max_trials,
                metrics=[metrics.Metric(name="eval_loss", greater_is_better=False)],
                knowledge_transfer=FlashDistillationConfig(
                    block_names=['bert.encoder.layer.0'],
                    layer_mappings_for_knowledge_transfer=[
                        [
                            [('bert.encoder.layer.0.output',)]
                        ],
                    ],
                    train_steps=[3],
                    loss_types=[['MSE']],),
                regular_distillation=FlashDistillationConfig(
                    layer_mappings_for_knowledge_transfer=[
                        [[('cls', '0')]]
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

    def test_autodistillation_distributed(self):
        try:
            distributed_init() # for coverage purpose only
        except:
            pass
        ngpus_per_node = 2
        torch.multiprocessing.spawn(
            main_worker, nprocs=ngpus_per_node,
            args=(ngpus_per_node, self.model, self.teacher_model, self.dummy_dataset)
        )

if __name__ == "__main__":
    unittest.main()
