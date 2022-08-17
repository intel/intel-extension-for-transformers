import numpy as np
import os
import onnx
import shutil
import torch
import torch.utils.data as data
import unittest
from nlp_toolkit import (
    metrics,
    objectives,
    OptimizedModel,
    QuantizationConfig,
    QuantizationMode,
    NoTrainerOptimizer,
)
from nlp_toolkit.optimization.trainer import NLPTrainer
from nlp_toolkit.optimization.trainer import NLPSeq2SeqTrainer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

class DummyDataset(data.Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.sequence_a = "NLP-toolkit is based in SH"
        self.sequence_b = "Where is NLP-toolkit based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b)
        self.encoded_dict['labels'] = 1

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

class TestQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
        self.dummy_dataset = DummyDataset()
        self.trainer = NLPTrainer(
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )
        self.optimizer = NoTrainerOptimizer(self.model)


    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./mlruns', ignore_errors=True)
        shutil.rmtree('./tmp_trainer', ignore_errors=True)
        shutil.rmtree('./quantized_model', ignore_errors=True)
        shutil.rmtree('fp32-model.onnx', ignore_errors=True)
        shutil.rmtree('int8-model.onnx', ignore_errors=True)

    def test_fx_model_quant(self):
        fp32_output = self.trainer.predict(self.dummy_dataset).predictions
        for mode in QuantizationMode:
            print("Quantization approach:", mode.value)
            self.trainer = NLPTrainer(
                model=self.model,
                train_dataset=self.dummy_dataset,
                eval_dataset=self.dummy_dataset,
            )

            # Check fp32 jit and onnx model, only once.
            if mode == QuantizationMode.POSTTRAININGSTATIC:
                jit_model = self.trainer.export_to_jit()
                self.trainer.export_to_onnx('fp32-model.onnx')
                self.assertTrue(check_onnx('fp32-model.onnx', self.trainer.get_eval_dataloader()))

            tune_metric = metrics.Metric(
                name="eval_loss", greater_is_better=False, is_relative=False, criterion=0.5
            )
            quantization_config = QuantizationConfig(
                approach=mode.name,
                metrics=[tune_metric],
                objectives=[objectives.performance]
            )
            quantized_model = self.trainer.quantize(quant_config=quantization_config, provider="inc")
            # By default, model will be saved into tmp_trainer dir.
            self.trainer.save_model('./quantized_model')

            # Check int8 onnx model
            if mode == QuantizationMode.POSTTRAININGSTATIC:
                # test different configure to improve UT coverage
                self.trainer.export_to_onnx(
                    save_path=None,
                    quant_format='Qlinear',
                    dtype='S8S8',
                    opset_version=13,
                )
                self.assertTrue(check_onnx('./tmp_trainer/int8-model.onnx', self.trainer.get_eval_dataloader()))
            else:
                self.trainer.export_to_onnx('int8-model.onnx')
                self.assertTrue(check_onnx('int8-model.onnx', self.trainer.get_eval_dataloader()))

            if mode == QuantizationMode.QUANTIZATIONAWARETRAINING:
                model = onnx.load('int8-model.onnx')
                tensor_list = {tensor.name:tensor for tensor in model.graph.initializer}
                torch_data = quantized_model.model.classifier.state_dict()\
                                ['module._packed_params._packed_params'][0].\
                                dequantize().detach().cpu().numpy().T
                from onnx.numpy_helper import to_array
                onnx_data = to_array(tensor_list['classifier.weight_quantized'])
                onnx_scale = to_array(tensor_list['classifier.weight_scale'])
                self.assertTrue(np.allclose(torch_data, onnx_data * onnx_scale, atol=0.001))
            # Check quantized model
            output_1 = self.trainer.predict(self.dummy_dataset).predictions
            loaded_model = OptimizedModel.from_pretrained(
                './quantized_model',
            )
            self.trainer.model = loaded_model
            output_2 = self.trainer.predict(self.dummy_dataset).predictions
            self.assertTrue((fp32_output != output_1).any())

            # check loaded model
            self.assertTrue((output_1 == output_2).all())

    def test_functional_quant(self):
        def eval_func(model):
            return 1

        def train_func(model):
            return model

        self.trainer = NLPTrainer(self.model, train_dataset=self.dummy_dataset)
        quantization_config = QuantizationConfig(
            approach='PostTrainingStatic',
            objectives=[objectives.performance]
        )
        self.trainer.quantize(quant_config=quantization_config, 
                              provider="inc",
                              train_func = train_func,
                              eval_func = eval_func,)

    def test_no_trainer_quant(self):
        def eval_func(model):
            return 1

        def train_func(model):
            return model

        tune_metric = metrics.Metric(
            name="eval_loss", greater_is_better=False, is_relative=False, criterion=0.5
        )
        quantization_config = QuantizationConfig(
            approach='PostTrainingStatic',
            metrics=[tune_metric],
            objectives=[objectives.performance]
        )
        self.optimizer.eval_func = eval_func
        self.optimizer.train_func = train_func
        self.optimizer.provider = "INC"
        self.optimizer.calib_dataloader = self.trainer.get_eval_dataloader()

        opt_model = self.optimizer.quantize(quant_config=quantization_config, 
                              provider="inc",
                              train_func = train_func,
                              eval_func = eval_func)

    def test_online_models(self):
        model = OptimizedModel.from_pretrained(
            'Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static'
        )

    def test_seq2seq_models(self):
        model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
        tokenizer = AutoTokenizer.from_pretrained('t5-small')
        trainer = NLPSeq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=self.dummy_dataset,
                eval_dataset=self.dummy_dataset,
            )
        trainer.max_length = 128
        trainer.num_beams = 3
        trainer.metrics = metrics.Metric(
                name="eval_loss"
            )
        trainer.args.predict_with_generate = True
        trainer.builtin_eval_func(self.model)

    def test_bf16_onnx(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            'google/bert_uncased_L-2_H-128_A-2'
        )
        tokenizer = AutoTokenizer.from_pretrained(
            'google/bert_uncased_L-2_H-128_A-2'
        )
        trainer = NLPSeq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=self.dummy_dataset,
                eval_dataset=self.dummy_dataset,
            )
        trainer.enable_bf16 = True
        trainer.export_to_onnx()
        from onnx import TensorProto
        model = onnx.load('./tmp_trainer/bf16-model.onnx')
        for tensor in model.graph.initializer:
            if 'MatMul' in tensor.name:
                self.assertEqual(tensor.data_type, TensorProto.BFLOAT16)
                break


if __name__ == "__main__":
    unittest.main()
