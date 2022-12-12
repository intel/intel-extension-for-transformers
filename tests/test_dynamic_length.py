import os
import onnx
import shutil
import torch
import torch.utils.data as data
import unittest
import transformers
import logging
import numpy as np

from intel_extension_for_transformers.optimization.trainer import NLPTrainer
from intel_extension_for_transformers.optimization.utils.models.modeling_roberta_dynamic import RobertaForQuestionAnswering
from intel_extension_for_transformers.optimization.utils.models.modeling_bert_dynamic import BertForQuestionAnswering
from intel_extension_for_transformers.optimization.dynamic.drop_and_restore_utils import (
    sample_length_configuration,
    sample_layer_configuration
)

from intel_extension_for_transformers.optimization.dynamic.evolution import (
    approx_ratio, inverse, store2str, Evolution
)

transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering = RobertaForQuestionAnswering
transformers.models.bert.modeling_bert.BertForQuestionAnswering = BertForQuestionAnswering

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
)

from intel_extension_for_transformers.optimization import (
    DynamicLengthConfig,
)


os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"
BERT_MODEL = "bert-base-uncased"
MINILM_MODEL = "sguskin/minilmv2-L6-H384-squad1.1"
MAX_LENGTH = 27
SANDWICHES = 2
LENGTHDROP_RATIO = 0.2
LAYER_DROPOUT_PROB = 0.2
LAYER_DROPOUT_BOUND = 0.2
EVO_ITER = 2
POPULATION_SIZE = 2
MUTATION_SIZE = 2
CROSSOVER_SIZE = 2
NUM_LAYERS = 6


class DummyDataset(data.Dataset):
    def __init__(self, labels=False, type=None):
        MODEL_NAME = BERT_MODEL if type=='bert' else MINILM_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based?"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b, padding="max_length", max_length=MAX_LENGTH)
        if labels:
            self.encoded_dict['start_positions'] = [21]
            self.encoded_dict['end_positions'] = [25]
        
    def __len__(self):
        return 1

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        return self.encoded_dict


def check_onnx(model_path, dataloader):
    import onnxruntime as ort
    import numpy as np
    # Check onnxruntime
    ort_session = ort.InferenceSession(model_path)
    # Preprocess input for onnxruntime
    it = iter(dataloader)
    input = next(it)
    input_names = list(input.keys())
    for k in input_names:
        if 'label' in k:
            input.pop(k)
        else:
            input[k] = np.array(input[k])
    # Run onnxruntime inference session
    ort_session.run(None, input,)
    return True

class TestDynamicLengthInferenceRoberta(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            MINILM_MODEL
        )
        self.dummy_dataset = DummyDataset(type='roberta')
        self.dynamic_trainer = NLPTrainer(
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./mlruns', ignore_errors=True)
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./dynamic-model.onnx', ignore_errors=True)

    def test_dynamic_inference(self):
        full_output = self.dynamic_trainer.predict(self.dummy_dataset).predictions
        
        dynamic_length_config = DynamicLengthConfig(
            const_rate=0.2,
            max_length=MAX_LENGTH
        )
        self.dynamic_trainer.set_dynamic_config(dynamic_length_config)
        
        dynamic_output = self.dynamic_trainer.predict(self.dummy_dataset).predictions
          
        self.assertTrue((full_output[0] != dynamic_output[0]).any())
  
        #check onnx
        self.dynamic_trainer.export_to_onnx('dynamic-model.onnx')
        self.assertTrue(check_onnx('dynamic-model.onnx', self.dynamic_trainer.get_eval_dataloader()))

class TestDynamicLengthInferenceBert(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            BERT_MODEL
        )
        self.dummy_dataset = DummyDataset(type='bert')
        self.dynamic_trainer = NLPTrainer(
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./mlruns', ignore_errors=True)
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./dynamic-model.onnx', ignore_errors=True)

    def test_dynamic_inference(self):
        full_output = self.dynamic_trainer.predict(self.dummy_dataset).predictions
        
        dynamic_length_config = DynamicLengthConfig(
            const_rate=0.2,
            max_length=MAX_LENGTH
        )
        self.dynamic_trainer.set_dynamic_config(dynamic_length_config)
        
        dynamic_output = self.dynamic_trainer.predict(self.dummy_dataset).predictions
          
        self.assertTrue((full_output[0] != dynamic_output[0]).any())
  
        #check onnx
        self.dynamic_trainer.export_to_onnx('dynamic-model.onnx')
        self.assertTrue(check_onnx('dynamic-model.onnx', self.dynamic_trainer.get_eval_dataloader()))



class TestDynamicLengthTrainingRoberta(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            MINILM_MODEL
        )
        self.dummy_dataset = DummyDataset(labels=True,type='roberta')
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
        )
        self.dynamic_trainer = NLPTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./mlruns', ignore_errors=True)
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./results', ignore_errors=True)


    def test_dynamic_training(self):
        dynamic_length_config = DynamicLengthConfig(
            dynamic_training=True,
            num_sandwich=SANDWICHES,
            length_drop_ratio_bound=LENGTHDROP_RATIO,
            layer_dropout_prob=LAYER_DROPOUT_PROB,
            layer_dropout_bound=LAYER_DROPOUT_BOUND,
            max_length=MAX_LENGTH
        )

        self.dynamic_trainer.set_dynamic_config(dynamic_config=dynamic_length_config)
        self.dynamic_trainer.train()

class TestDynamicLengthTrainingBert(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            BERT_MODEL
        )
        self.dummy_dataset = DummyDataset(labels=True,type='bert')
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
        )
        self.dynamic_trainer = NLPTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./mlruns', ignore_errors=True)
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./results', ignore_errors=True)


    def test_dynamic_training(self):
        dynamic_length_config = DynamicLengthConfig(
            dynamic_training=True,
            num_sandwich=SANDWICHES,
            length_drop_ratio_bound=LENGTHDROP_RATIO,
            layer_dropout_prob=LAYER_DROPOUT_PROB,
            layer_dropout_bound=LAYER_DROPOUT_BOUND,
            max_length=MAX_LENGTH
        )

        self.dynamic_trainer.set_dynamic_config(dynamic_config=dynamic_length_config)
        self.dynamic_trainer.train()


class TestEvolutionarySearchRoberta(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            MINILM_MODEL
        )
        self.dummy_dataset = DummyDataset(labels=True,type='roberta')
        training_args = TrainingArguments(
            output_dir='./results',
        )

        self.dynamic_trainer = NLPTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )



    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./mlruns', ignore_errors=True)
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./results', ignore_errors=True)


    def test_dynamic_training(self):
        
        dynamic_length_config = DynamicLengthConfig(
            evo_iter=EVO_ITER,
            population_size=POPULATION_SIZE,
            mutation_size=MUTATION_SIZE,
            crossover_size=CROSSOVER_SIZE,
            max_length=MAX_LENGTH,
            evo_eval_metric='eval_loss' # dummy metric just for testing
        )

        self.dynamic_trainer.set_dynamic_config(dynamic_config=dynamic_length_config)
        self.dynamic_trainer.run_evolutionary_search() 

class TestEvolutionarySearchBert(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            BERT_MODEL
        )
        self.dummy_dataset = DummyDataset(labels=True,type='bert')
        training_args = TrainingArguments(
            output_dir='./results',
        )

        self.dynamic_trainer = NLPTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )



    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./mlruns', ignore_errors=True)
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./results', ignore_errors=True)


    def test_search(self):
        
        dynamic_length_config = DynamicLengthConfig(
            evo_iter=EVO_ITER,
            population_size=POPULATION_SIZE,
            mutation_size=MUTATION_SIZE,
            crossover_size=CROSSOVER_SIZE,
            max_length=MAX_LENGTH,
            evo_eval_metric='eval_loss'
        )

        self.dynamic_trainer.set_dynamic_config(dynamic_config=dynamic_length_config)
        # self.dynamic_trainer.run_evolutionary_search() # dummy metric just for testing

    def test_search_functions(self):
        dummy_ratios = [inverse(r) for r in np.linspace(approx_ratio(0.2,n=NUM_LAYERS,l=MAX_LENGTH), 1, POPULATION_SIZE + 2)[1:-1]]
        self.assertTrue( len(dummy_ratios) == POPULATION_SIZE, msg='{0},{1}'.format(len(dummy_ratios), POPULATION_SIZE))
        self.assertTrue( all(i < 1 for i in dummy_ratios), msg='{0}'.format(dummy_ratios))
        res = store2str(gene=(MAX_LENGTH,), macs=1234, score=88, method='dummy')
        subs = ('MACs','score','method')
        self.assertTrue( all(i in res for i in subs), msg='{0}'.format(res))
        evo = Evolution(self.model,MAX_LENGTH, 'cpu', None, eval_metric='eval_loss')
        


class TestSampleConfiguration(unittest.TestCase):
   
    def test_sample_length_config(self):
 
        no_drop_lc = tuple( MAX_LENGTH for _ in range(NUM_LAYERS))
        lc = sample_length_configuration(MAX_LENGTH, NUM_LAYERS)

        self.assertTrue( lc == no_drop_lc, msg='{0}, {1}'.format(lc, no_drop_lc))

        rate = 0.1
        next_length = MAX_LENGTH
        lc_const_drop = ()
        for _ in range(NUM_LAYERS):
            next_length = int(np.ceil(next_length * (1 - rate)))
            lc_const_drop += (next_length,)

        lc = sample_length_configuration(MAX_LENGTH, NUM_LAYERS, length_drop_ratio=rate)
        self.assertTrue( lc == lc_const_drop, msg='{0}, {1}'.format(lc, lc_const_drop))
        lc = sample_length_configuration(MAX_LENGTH, NUM_LAYERS, length_drop_prob=0.2)
        self.assertTrue( lc[-1] < MAX_LENGTH , msg='{0}, {1}'.format(lc[-1], MAX_LENGTH))
        lc = sample_length_configuration(MAX_LENGTH, NUM_LAYERS, length_drop_ratio_bound=0.2)
        self.assertTrue( lc[-1] < MAX_LENGTH , msg='{0}, {1}'.format(lc[-1], MAX_LENGTH))

        no_drop_lc = tuple( l for l in range(NUM_LAYERS))
        lc = sample_layer_configuration(NUM_LAYERS, layer_dropout=0)
        self.assertTrue( lc == no_drop_lc, msg='{0}, {1}'.format(lc, no_drop_lc))
        drop_rate = NUM_LAYERS-3
        lc = sample_layer_configuration(NUM_LAYERS, layer_dropout=drop_rate)
        expected = tuple(i for i in range(drop_rate))
        self.assertTrue( lc == expected, msg='{0}, {1}'.format(lc, expected))
        layer_conf = sample_layer_configuration(NUM_LAYERS, layer_dropout_prob=0.2)
        length_conf = sample_length_configuration(MAX_LENGTH, NUM_LAYERS, layer_config=layer_conf, length_drop_ratio=0.1)
        self.assertTrue( len(length_conf) == NUM_LAYERS , msg='{0}, {1}'.format(length_conf, layer_conf))


       

        




if __name__ == "__main__":
    unittest.main()
   