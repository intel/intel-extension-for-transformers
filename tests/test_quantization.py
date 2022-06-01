import os
import shutil
import torch
import torch.utils.data as data
import unittest
from nlp_toolkit import (
    metrics,
    NLPTrainer,
    objectives,
    OptimizedModel,
    QuantizationConfig,
    QuantizationMode
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"
MODEL_NAME = "distilbert-base-uncased"

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

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./mlruns', ignore_errors=True)
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

            self.trainer.enable_engine = True
            self.trainer.export_to_onnx('int8-model.onnx')
            self.assertTrue(check_onnx('int8-model.onnx', self.trainer.get_eval_dataloader()))

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


if __name__ == "__main__":
    unittest.main()
