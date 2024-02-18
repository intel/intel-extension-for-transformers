# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ast import LShift
from math import isclose
import numpy as np
import os
import onnx
import shutil
import torch
import torch.utils.data as data
import unittest
from intel_extension_for_transformers.transformers import (
    metrics,
    objectives,
    OptimizedModel,
    QuantizationConfig,
    QuantizationMode,
    NoTrainerOptimizer,
)
from intel_extension_for_transformers.transformers.trainer import NLPTrainer
from intel_extension_for_transformers.transformers.trainer import NLPSeq2SeqTrainer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
)
import neural_compressor.adaptor.pytorch as nc_torch
from packaging.version import Version

os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
PT_VERSION = nc_torch.get_torch_version()

class DummyDataset(data.Dataset):
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


def check_onnx(model_path, dataloader):
    import onnxruntime as ort
    import numpy as np
    # Check onnxruntime
    ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
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
        shutil.rmtree('export.onnx', ignore_errors=True)
        shutil.rmtree('./local_model_dir', ignore_errors=True)

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

            self.trainer.benchmark(num_of_instance=1)
            tune_metric = metrics.Metric(
                name="eval_loss", greater_is_better=False, is_relative=False, criterion=0.5
            )
            quantization_config = QuantizationConfig(
                approach=mode.name,
                metrics=[tune_metric],
                objectives=[objectives.performance]
            )
            quantized_model = self.trainer.quantize(quant_config=quantization_config, provider="inc")
            self.trainer.benchmark(self.trainer.args.output_dir, num_of_instance=1)
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
                torch_data = quantized_model.classifier.state_dict()\
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

    def test_fx_model_with_smooth_quant(self):
        def eval_func(model):
            return 1

        def train_func(model):
            return model

        trainer = NLPTrainer(
            model=self.model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )

        tune_metric = metrics.Metric(
            name="eval_loss", greater_is_better=False, is_relative=False, criterion=0.5
        )
        quantization_config = QuantizationConfig(
            approach="PostTrainingStatic",
            metrics=[tune_metric],
            objectives=[objectives.performance],
            recipes={"smooth_quant": True,
                     "smooth_quant_args": {"alpha": 0.6},
                     }
        )
        recipes = quantization_config.recipes
        self.assertTrue(recipes["smooth_quant"])
        quantized_model = trainer.quantize(quant_config=quantization_config)
        self.assertTrue("quantize" in str(type(quantized_model.classifier.module)))
        quantization_config = QuantizationConfig(
            approach="PostTrainingStatic",
            metrics=[tune_metric],
            objectives=[objectives.performance],
            recipes={}
        )
        quantized_model = trainer.quantize(quant_config=quantization_config,
                                           train_func=train_func,
                                           eval_func=eval_func)
        self.assertTrue("quantize" in str(type(quantized_model.classifier.module)))

        with self.assertRaises(ValueError):
            quantization_config = QuantizationConfig(
                approach="PostTrainingStatic",
                metrics=[tune_metric],
                objectives=[objectives.performance],
                recipes=[]
            )

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
        args=Seq2SeqTrainingArguments(output_dir='./tmp_trainer')
        trainer = NLPSeq2SeqTrainer(
                model=model,
                args=args,
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
        args=Seq2SeqTrainingArguments(output_dir='./tmp_trainer')
        trainer = NLPSeq2SeqTrainer(
                model=model,
                args=args,
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

    @unittest.skipIf(PT_VERSION.release < Version("2.1.0").release,
            "Please use PyTroch 2.1.0 or higher version for executor backend")
    def test_quantization_for_llm(self):
        model_name_or_path = "hf-internal-testing/tiny-random-gptj"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        from intel_extension_for_transformers.transformers import (
            MixedPrecisionConfig,
            SmoothQuantConfig,
            WeightOnlyQuantConfig,
            BitsAndBytesConfig
        )
        from intel_extension_for_transformers.transformers import AutoModelForCausalLM
        fp32_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_llm_runtime=False)
        dummy_input = fp32_model.dummy_inputs["input_ids"]
        #smooth-quant
        sq_config = SmoothQuantConfig(
                                    tokenizer=tokenizer,  # either two of one, tokenizer or calib_func
                                    calib_iters=2,
                                    ipex_opt_llm=False
                                    )
        q_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    quantization_config=sq_config,
                                                    use_llm_runtime=False
                                                )
        self.assertTrue(isinstance(q_model.model, torch.jit.ScriptModule))
        #SQ auto
        recipes = {
            "smooth_quant": True,
            "smooth_quant_args": { "alpha": "auto", "auto_alpha_args":{"alpha_max": 0.6,
                "alpha_min":0.5, "alpha_step":0.1, "shared_criterion": "mean", "do_blockwise": False}},
            }
        sq_config = SmoothQuantConfig(
                            tokenizer=tokenizer,  # either two of one, tokenizer or calib_func
                            calib_iters=2,
                            recipes=recipes,
                            ipex_opt_llm=False
                            )
        q_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    quantization_config=sq_config,
                                                    use_llm_runtime=False
                                                )
        self.assertTrue(isinstance(q_model.model, torch.jit.ScriptModule))
        # weight-only
        #RTN
        woq_config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange")
        woq_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    quantization_config=woq_config,
                                                    use_llm_runtime=False
                                                )
        woq_model.eval()
        output = woq_model(dummy_input)
        print("output:", float(output[0][0][0][0]))
        self.assertTrue(isclose(float(output[0][0][0][0]), 0.16387596726417542, rel_tol=1e-04))
        #AWQ
        woq_config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange",
                                           calib_iters=5,
                                           tokenizer=tokenizer,
                                           algorithm="AWQ")
        woq_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    quantization_config=woq_config,
                                                    use_llm_runtime=False
                                                )
        woq_model.eval()
        output = woq_model(dummy_input)
        print("output:", float(output[0][0][0][0]))
        self.assertTrue(isclose(float(output[0][0][0][0]), 0.17239853739738464, rel_tol=1e-04))
        #TEQ
        woq_config = WeightOnlyQuantConfig(weight_dtype="int4_fullrange",
                                           calib_iters=5,
                                           tokenizer=tokenizer,
                                           algorithm="TEQ")
        woq_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    quantization_config=woq_config,
                                                    use_llm_runtime=False
                                                )
        woq_model.eval()
        output = woq_model(dummy_input)
        #fp8
        woq_config = WeightOnlyQuantConfig(weight_dtype="fp8_e5m2", scale_dtype="fp8_e8m0")
        woq_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, quantization_config=woq_config, use_llm_runtime=False
        )
        woq_model.eval()
        output = woq_model(dummy_input)
        self.assertTrue(
           isclose(float(output[0][0][0][0]), 0.16162332892417908, rel_tol=1e-04)
        )
        # amp
        amp_config = MixedPrecisionConfig()
        amp_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    quantization_config=amp_config,
                                                    use_llm_runtime=False
                                                )
        amp_model.eval()
        output = amp_model(dummy_input)
        self.assertTrue(isclose(float(output[0][0][0][0]), 0.1689453125, rel_tol=1e-04))
        # bitsandbytes, for cpu is fp32 model
        bab_config = BitsAndBytesConfig()
        bab_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    quantization_config=bab_config,
                                                    use_llm_runtime=False
                                                )
        # load_in_4bit
        bit4_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     load_in_4bit=True,
                                                     use_llm_runtime=False
                                                )
        bit4_model.eval()
        output = bit4_model(dummy_input)
        print("output:", float(output[0][0][0][0]))
        self.assertTrue(isclose(float(output[0][0][0][0]), 0.18726778030395508, rel_tol=1e-04))

        # load_in_8bit
        bit8_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     load_in_8bit=True,
                                                     use_llm_runtime=False,
                                                     device_map="cpu"
                                                )
        bit8_model.eval()
        output = bit8_model(dummy_input)
        print("output:", float(output[0][0][0][0]))
        self.assertTrue(isclose(float(output[0][0][0][0]), 0.1675747185945511, rel_tol=1e-04))

        #GPTQ
        algorithm_args = {
            "act_order": False,
            "percdamp": 0.01,
            "block_size": 32 ,
            "nsamples": 3,
            "use_max_length": True,
            "pad_max_length": 256,
        }
        woq_config = WeightOnlyQuantConfig(weight_dtype="int4_clip",
                                        algorithm_args=algorithm_args,
                                        tokenizer=tokenizer,
                                        algorithm="GPTQ")
        woq_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    quantization_config=woq_config,
                                                    use_llm_runtime=False
                                                )
        woq_model.eval()
        output = woq_model(dummy_input)
        print("output:", float(output[0][0][0][0]))
        self.assertTrue(isclose(float(output[0][0][0][0]), 0.17126554250717163, rel_tol=1e-04))

    def test_export(self):
        # test model with model_id
        self.trainer.export_to_onnx("export.onnx")
        self.assertTrue(check_onnx("export.onnx", self.trainer.get_eval_dataloader()))

        # test local model path
        from huggingface_hub import snapshot_download
        snapshot_download("distilbert-base-uncased-finetuned-sst-2-english", local_dir ="./local_model_dir")

        local_model = AutoModelForSequenceClassification.from_pretrained("local_model_dir")
        local_model_trainer = NLPTrainer(
            model=local_model,
            train_dataset=self.dummy_dataset,
            eval_dataset=self.dummy_dataset,
        )
        local_model_trainer.export_to_onnx("export.onnx")
        self.assertTrue(check_onnx("export.onnx", self.trainer.get_eval_dataloader()))


if __name__ == "__main__":
    unittest.main()
