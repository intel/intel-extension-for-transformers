#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFOptimization: provides the optimization class for Tensorflow."""
import logging
import pstats
import numpy as np
import os
import time
from neural_compressor import __version__
from neural_compressor.experimental import common
try:
    from neural_compressor.model.model import saved_model_session, get_model_type
except ImportError:
    from neural_compressor.model.tensorflow_model import saved_model_session, get_model_type
from intel_extension_for_transformers.optimization import (DistillationConfig,
                                                           QuantizationConfig,
                                                           PruningConfig,
                                                           AutoDistillation)
from intel_extension_for_transformers.optimization.quantization import QuantizationMode
from intel_extension_for_transformers.optimization.utils.metrics import Metric
from intel_extension_for_transformers.optimization.utils.utility import LazyImport
from packaging import version
from transformers import PreTrainedModel
from transformers.training_args_tf import TFTrainingArguments
from typing import Callable, Optional, List
from .utils.utility_tf import TFDataloader, TMPPATH, TEACHERPATH, get_filepath

from functools import partial

tf = LazyImport("tensorflow")
logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class TFOptimization:
    """TFOptimization is the entry class for Tensorflow to use the optimization techniques in neural compressor."""
    def __init__(self,
                 model: PreTrainedModel,
                 args: TFTrainingArguments,
                 train_dataset=None,
                 eval_dataset=None,
                 compute_metrics: Optional[Callable] = None,
                 criterion=None,
                 optimizer=None,
                 task_type=None,
                 task_id=None,
                 strategy=None):
        """Init a TFOptimziation object.

        Args:
            model: FP32 model specified for low precision tuning
            args: Training arguments for TF
            train_dataset: Training datas
            eval_dataset: Evaluation data
            compute_metrics: Metrics computing function during the evaluation process
            criterion: Tuning criterion
            optimizer: The optimizer you used
            task_type: Used for distributed multi-node settings. Default set as "worker"
            task_id: Used for distributed multi-node settings.
                Set as 0 on the leader node and 1, 2, 3... on the followers
            strategy: Equals to MultiWorkerMirroredStrategy if use distributed distributed multi-node settings,
                otherwise None
        """
        self.model = model
        self.teacher_model = None
        self.component = None
        self.eval_dataset = eval_dataset
        self.train_dataset = train_dataset
        self._eval_func = None
        self._train_func = None
        self.quant_config = None
        self.pruning_config = None
        self.distillation_config = None
        self.pruner = None
        self.quantizer = None
        self.distiller = None
        self.in_training = False
        self._input_names = None
        self._output_names = None
        self._inputs = None
        self.compute_metrics = compute_metrics
        self.args = args
        self.optimizer = optimizer
        self.task_type = task_type
        self.task_id = task_id
        self.criterion = criterion if criterion is not None else \
            self.model.loss if hasattr(self.model, "loss") else None
        self.model.save_pretrained(get_filepath(TMPPATH, self.task_type, self.task_id), saved_model=True)
        _, self.input_names, self.output_names = saved_model_session(
            os.path.join(get_filepath(TMPPATH, self.task_type, self.task_id), "saved_model/1"), input_tensor_names=[],
             output_tensor_names=[])
        self.eval_distributed = False
        self.strategy = strategy

    @property
    def inputs(self):
        """Get the inputs."""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: dict):
        """Set the inputs."""
        self._inputs = inputs

    @property
    def input_names(self):
        """Get the input names."""
        return self._input_names

    @input_names.setter
    def input_names(self, input_names: List):
        """Set the input names.
        
        Args:
            input_names: the names of inputs.
        """
        self._input_names = input_names

    @property
    def output_names(self):
        """Get the output names."""
        return self._output_names

    @output_names.setter
    def output_names(self, output_names: List):
        """Set the output names.
        
        Args:
            output_names: the names of outputs.
        """
        self._output_names = output_names

    @property
    def eval_func(self):
        """Get the evaluation function."""
        return self._eval_func

    @eval_func.setter
    def eval_func(self, func: Callable):
        """Set the evaluation function.
        
        Args:
            func: evaluation function.
        """
        self._eval_func = func

    @property
    def train_func(self):
        """Get the training function."""
        return self._train_func

    @train_func.setter
    def train_func(self, func: Callable):
        """Set the training function.
        
        Args:
            func: train function.
        """
        self._train_func = func

    @property
    def train_dataset(self):
        """Get the training dataset."""
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, train_dataset):
        """Set the training dataset.
        
        Args:
            train_dataset: train dataset.
        """
        assert isinstance(train_dataset, tf.data.Dataset) or train_dataset is None, \
            "train_dataset should be obj of tf.data.Dataset"
        self._train_dataset = train_dataset

    @property
    def eval_dataset(self):
        """Get the evaluation dataset."""
        return self._eval_dataset

    @eval_dataset.setter
    def eval_dataset(self, eval_dataset):
        """Set the evaluation dataset.
        
        Args:
            eval_dataset: evaluation dataset.
        """
        assert isinstance(eval_dataset, tf.data.Dataset) or eval_dataset is None, \
            "eval_dataset should be obj of tf.data.Dataset"
        self._eval_dataset = eval_dataset

    def builtin_eval_func(self, model):
        """Customize Evaluate function to inference the model for specified metric on the validation dataset.

        Args:
            model ([tf.saved_model.load]): The model will be the class of tf.saved_model.load(quantized_model_path).

        Returns:
            [float]: evaluation result, the larger is better.
        """
        model_type = None
        label_ids: np.ndarray = None
        try:
            model_type = get_model_type(model)
        except ValueError:
            logger.info("use keras savedModel")

        num_examples = sum(1 for _ in (
            self._eval_dataset.unbatch() if hasattr(self._eval_dataset, "unbatch") else self._eval_dataset))
        logger.info(f"***** Running Evaluation *****")
        logger.info(f"  Num examples in dataset = {num_examples}")
        logger.info(f"  Batch size = {self.args.per_device_eval_batch_size}")

        if model_type is None:
            preds: np.ndarray = None
            infer = model.signatures["serving_default"]

            for idx, (inputs, labels) in enumerate(self._eval_dataset):
                for name in inputs:
                    inputs[name] = tf.constant(inputs[name].numpy(), dtype=infer.inputs[0].dtype)

                results = infer(**inputs)
                for val in results:
                    if preds is None:
                        preds = results[val].numpy()
                    else:
                        preds = np.append(preds, results[val].numpy(), axis=0)

                if label_ids is None:
                    label_ids = labels[0].numpy() if isinstance(
                        labels, list) else labels.numpy()
                else:
                    label_ids = np.append(
                        label_ids,
                        labels[0].numpy()
                        if isinstance(labels, list) else labels.numpy(),
                        axis=0)
            test_predictions = {"logits": preds}
            eval_metrics = self.compute_metrics(test_predictions, label_ids)
            acc = eval_metrics["accuracy"]
            return acc
        else:  # pragma: no cover
            from neural_compressor.adaptor.tf_utils.util import get_tensor_by_name
            input_tensor = [get_tensor_by_name(\
                model, x) for x in self.input_names]
            output_tensor = [get_tensor_by_name(\
                model, x) for x in self.output_names]

            logger.info("Start to evaluate the TensorFlow model.")

            total_time = 0
            config = tf.compat.v1.ConfigProto()
            config.use_per_session_threads = 1
            config.inter_op_parallelism_threads = 1
            sess = tf.compat.v1.Session(graph=model, config=config)
            feed_dict = {}
            label_ids: np.ndarray = None
            preds: np.ndarray = None
            for idx, (inputs, labels) in enumerate(self._eval_dataset):
                assert len(input_tensor) == len(inputs), \
                    'inputs len must equal with input_tensor'
                feed_dict = {}
                for name in inputs:
                    for tensor in input_tensor:
                        pos = tensor.name.rfind(":")
                        t_name = tensor.name if pos < 0 else tensor.name[:pos]
                        if name == t_name:
                            feed_dict[tensor] = inputs[name].numpy()
                            break

                start = time.time()
                logits = sess.run(output_tensor, feed_dict)
                total_time += time.time() - start
                if not self.args.prediction_loss_only:
                    if isinstance(logits, tuple):
                        logits = logits[0]

                    if isinstance(labels, tuple):
                        labels = labels[0].numpy()

                    if isinstance(logits,
                                list) and len(logits) > 1:  # pragma: no cover
                        for val in logits:
                            if preds is None:
                                preds = val
                            else:
                                preds = np.append(preds, val, axis=0)

                        for val in labels:
                            if label_ids is None:
                                label_ids = val.numpy()
                            else:
                                label_ids = np.append(label_ids,
                                                    val.numpy(),
                                                    axis=0)
                    else:
                        if preds is None:
                            preds = logits[0] if isinstance(logits,
                                                            list) else logits
                        else:
                            preds = np.append(
                                preds,
                                logits[0] if isinstance(logits, list) else logits,
                                axis=0)

                        if label_ids is None:
                            label_ids = labels[0].numpy() if isinstance(
                                labels, list) else labels.numpy()
                        else:
                            label_ids = np.append(
                                label_ids,
                                labels[0].numpy()
                                if isinstance(labels, list) else labels.numpy(),
                                axis=0)

            if self.compute_metrics is not None and preds is not None and label_ids is not None:
                try:
                    loss = self.criterion(
                        label_ids, preds) if self.criterion is not None else None
                except Exception as e:  # pragma: no cover
                    logger.info(e)
                    logger.info("There is no loss function or loss compute error, \
                                    Please compute loss in compute_metrics function"
                                )
                    loss = None
                results = self.compute_metrics({"logits": preds}, label_ids)
                if loss is not None:
                    results["loss"] = loss.numpy()

                if isinstance(self.metrics, list):
                    nums = len(self.metrics)
                    for metric in self.metrics:
                        assert metric.name in results.keys(), \
                            "Please set metric from {}".format(results.keys())
                    if nums == 1:
                        result = results.get(self.metrics[0].name)
                    else:  # pragma: no cover
                        result = 0
                        for metric in self.metrics:
                            assert metric.weight_ratio is not None, \
                                "Please set weights for metric if you want to use more than one metric"
                            result += results[metric.name] * metric.weighted
                    logger.info("metric Accuracy: {}".format(result))
                elif isinstance(self.metrics, Metric):
                    assert self.metrics.name in results.keys(), \
                            "Please set metric from {}".format(results.keys())
                    result = results.get(self.metrics.name)
                    logger.info("metric Accuracy: {}".format(result))
                else:  # pragma: no cover
                    assert False, "Please set the correct metrics format from the README"
            else:
                result = 0
            logger.info("Throughput: {} samples/sec".format(num_examples / total_time))
            return result

    def init_quantizer(
        self,
        quant_config,
    ):
        """Init a Quantization object with config.
        
        Args:
            quant_config: quantization config.
        """
        from neural_compressor.experimental import Quantization

        self.quant_config = QuantizationConfig() if quant_config is None else quant_config
        self.quant_config.framework = "tensorflow"
        self.metrics = self.quant_config.metrics

        quantizer = Quantization(self.quant_config.inc_config)
        quantizer.model = common.Model(
            os.path.join(get_filepath(TMPPATH, self.task_type, self.task_id),"saved_model/1"), modelType="saved_model")

        self.quantizer = quantizer
        return quantizer

    def _inc_quantize(
        self,
        quant_config,
    ):
        """Do the quantization.
        
        Args:
            quant_config: quantization config.
        """
        if self.quantizer is None:
            self.init_quantizer(quant_config=quant_config)
        if self._eval_func is not None:
            self.quantizer.eval_func = self._eval_func
        else:
            assert self.metrics is not None, \
                "Please pass the metrics to QuantizationConfig.metrics!"
            self.quantizer.eval_func = self.builtin_eval_func

        if self.quant_config.approach == QuantizationMode.POSTTRAININGSTATIC.value:
            if self._train_dataset is not None:
                self.quantizer.calib_dataloader = TFDataloader(
                    self._train_dataset,
                    batch_size=self.args.per_device_train_batch_size)
            elif self._eval_dataset is not None:
                self.quantizer.calib_dataloader = TFDataloader(
                    self._eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size)
            else:  # pragma: no cover
                assert False, "Please pass calibration dataset to TFNoTrainerOptimizer.calib_dataloader"
        elif self.quant_config.approach == QuantizationMode.QUANTIZATIONAWARETRAINING.value:   # pragma: no cover
            assert False, \
                "Unsupport quantization aware training for tensorflow framework"

        opt_model = self.quantizer.fit()
        opt_model.save(self.args.output_dir)
        logger.info(
            "quantized model have saved to {}".format(self.args.output_dir)
        )
        return opt_model.model

    def quantize(
        self,
        quant_config: QuantizationConfig = None,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
        train_dataset=None,
        eval_dataset=None,
    ):
        """Prepare for invoking INC quantize function.
        
        Args:
            quant_config: quantization config.
            eval_func: evaluation function.
            train_func: train function.
            train_dataset: train dataset.
            eval_dataset: evaluation dataset.
        """
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func
        if train_dataset is not None:
            self.train_dataset = train_dataset

        if eval_dataset is not None:
            self.eval_dataset = eval_dataset

        return self._inc_quantize(quant_config=quant_config)

    def init_pruner(
        self,
        pruning_config=None,
    ):
        """Init a Pruning object with config.
        
        Args:
            pruning_config: pruning config.
        """
        from neural_compressor.experimental import Pruning
        if pruning_config.framework != 'tensorflow':
            logger.warning('pruning_config.framework is {}, should be tensorflow'.format(pruning_config.framework))
            pruning_config.framework = 'tensorflow'
        self.pruning_config = pruning_config
        self.metrics = self.pruning_config.metrics

        assert isinstance(self.pruning_config, PruningConfig), \
            "please pass a instance of PruningConfig to trainer.prune!"

        pruner = Pruning(self.pruning_config.inc_config)
        pruner.model = os.path.join(get_filepath(TMPPATH, self.task_type, self.task_id), "saved_model/1")
        pruner.model.model_type = "saved_model"

        self.pruner = pruner
        self.component = pruner
        return pruner

    def prune(
        self,
        pruning_config=None,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
        train_dataset=None,
        eval_dataset=None,
    ):
        """Do the pruning.
        
        Args:
            pruning_config: pruning config.
            eval_func: evaluation function.
            train_func: train function.
            train_dataset: train dataset.
            eval_dataset: evaluation dataset.
        """
        if self.pruner is None:
            self.init_pruner(pruning_config=pruning_config)
        if eval_func is not None:
            self.eval_func = eval_func
        if train_func is not None:
            self.train_func = train_func

        if train_dataset is not None:
            self.train_dataset = train_dataset

        if eval_dataset is not None:
            self.eval_dataset = eval_dataset

        if self._eval_func is not None:
            self.pruner.eval_func = self._eval_func
        else:
            assert self.metrics is not None, \
                "Please pass the metrics to PruningConfig.metrics!"
            self.pruner.eval_func = self.builtin_eval_func

        if self.train_func is not None:
            if version.parse(__version__) <= version.parse("1.12"):
                self.pruner.pruning_func = self._train_func
            else:
                self.pruner.train_func = self._train_func
        else:
            if version.parse(__version__) <= version.parse("1.12"):
                self.pruner.pruning_func = self.build_train_func
            else:
                self.pruner.train_func = self.build_train_func

        opt_model = self.pruner.fit()
        stats, sparsity = opt_model.report_sparsity()
        logger.info(stats)
        logger.info(sparsity)

        opt_model.save(self.args.output_dir)
        logger.info(
            "pruned model have saved to {}".format(self.args.output_dir)
        )
        return opt_model.model

    def init_distiller(
        self,
        distillation_config,
        teacher_model: PreTrainedModel,
    ):
        """Init a Distillation object with config and the teacher model.
        
        Args:
            distillation_config: distillation config.
            teacher_model: set the teacher model.
        """
        from neural_compressor.experimental import Distillation
        assert isinstance(distillation_config, DistillationConfig), \
            "please pass a instance of DistillationConfig to trainer.distill!"

        def train_step(data):
            if len(data) == 3:
                x, y, sample_weight = data  # pragma: no cover
            else:
                sample_weight = None
                x, y = data
            with tf.GradientTape() as tape:
                y_pred = self.model(x)
                teacher_outputs = self.distiller.criterion.teacher_model_forward(
                    input=x, teacher_model=teacher_model)

                loss = self.model.compute_loss(x, y, y_pred, sample_weight)
                # _on_after_compute_loss(self, input, student_output, student_loss, teacher_output=None)
                # TODO: check, combile
                loss = self.distiller.on_after_compute_loss(
                    x, y_pred.logits, loss, teacher_outputs.logits)
            self.model._validate_target_and_loss(y, loss)
            # Run backwards pass.
            self.model.optimizer.minimize(loss,
                                          self.model.trainable_variables,
                                          tape=tape)
            return self.model.compute_metrics(x, y, y_pred, sample_weight)

        self.model.train_step = train_step
        # re-compile
        self.model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            metrics=self.model.compiled_metrics._user_metrics
            )

        if distillation_config.framework != 'tensorflow':
            logger.warning(
                'distillation_config.framework is {}, should be tensorflow'.
                format(distillation_config.framework))
            distillation_config.framework = 'tensorflow'
        self.distillation_config = distillation_config
        self.metrics = self.distillation_config.metrics
        self.teacher_model = teacher_model

        distiller = Distillation(self.distillation_config.inc_config)
        distiller.model = os.path.join(TMPPATH, "saved_model/1")
        distiller.model.model_type = "saved_model"
        self.teacher_model.save_pretrained(TEACHERPATH, saved_model=True)
        distiller.teacher_model = os.path.join(TEACHERPATH, "saved_model/1")
        distiller.teacher_model.model_type = "saved_model"

        self.distiller = distiller
        self.component = distiller
        return distiller

    def distill(
        self,
        distillation_config,
        teacher_model: PreTrainedModel,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        """Do the distillation.
        
        Args:
            distillation_config: distillation config.
            teacher_model: set the teacher model.
            eval_func: evaluation function.
            train_func: train function.
        """
        if self.distiller is None:
            self.init_distiller(
                distillation_config=distillation_config,
                teacher_model=teacher_model,
            )
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func
        else:
            self._train_func = self.build_train_func

        self.distiller.eval_func = self._eval_func
        self.distiller.train_func = self._train_func
        self.distiller.create_criterion()

        opt_model = self.distiller.fit()
        opt_model.save(self.args.output_dir)
        logger.info(
            "distilled model have saved to {}".format(self.args.output_dir)
        )

        return opt_model.model

    def model_builder_builtin(self, arch_paras=None, model_cls=None):
        """Specify model_cls to use the built-in model builder.
        
        Args:
            arch_paras: architecture parameters.
            model_cls: model information.
        """
        config = self.model.config
        if arch_paras is not None:
            assert isinstance(arch_paras, dict), "Expect arch_paras to be a dict."
            for k in arch_paras:
                if hasattr(config, k):
                    config.__setattr__(k, arch_paras[k])
                    # for MobileBERT, 'intra_bottleneck_size' is associated with
                    # 'true_hidden_size', and must have the same values.
                    if k == 'intra_bottleneck_size':
                        config.__setattr__('true_hidden_size', arch_paras[k])
        return model_cls.from_config(config)


    def autodistill(
        self,
        autodistillation_config,
        teacher_model: PreTrainedModel,
        model_builder: Optional[Callable] = None,
        model_cls: Optional[Callable] = None,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None
        ):
        """Do the auto distillation.
        
        Args:
            autodistillation_config: autodistillation config.
            teacher_model: set the teacher model.
            model_builder: the configuration of build in model.
            model_cls: the model information.
            eval_func: evaluation function.
            train_func: train function.
        """
        self.autodistillation_config = autodistillation_config
        if model_builder is None:
            assert model_cls is not None, "Must specify model_cls to use the built-in " + \
                "model_builder, e.g. model_cls=AutoModelForPreTraining, or you can use " + \
                "the customized model_builder."
            model_builder = partial(self.model_builder_builtin, model_cls=model_cls)
        agent = AutoDistillation(model_builder, self.autodistillation_config, framework='tensorflow')

        def train_func_builtin(model):
            """Get the build in train function.

            Args:
                model (object): the input model
            """
            def run_distillers(
                model,
                distillers,
                train_steps,
                block_names,
                presentation='flash distillation'
            ):
                """Get the distiller.

                Args:
                    model (object): the input model.
                    distillers: distillers.
                    train_steps: number of train steps.
                    block_names: the name of the block.
                    presentation: presentation format.
                """
                for i, elements in enumerate(zip(distillers, train_steps, block_names)):
                    distiller, ts, bln = elements
                    logger.info(' '.join(
                        ['=' * 30, 'Step {} of'.format(i + 1), presentation, '=' * 30]))

                    def train_step(data):
                        if len(data) == 3:
                            x, y, sample_weight = data  # pragma: no cover
                        else:
                            sample_weight = None
                            x, y = data
                        with tf.GradientTape() as tape:
                            y_pred = model(x)
                            teacher_outputs = distiller.criterion.teacher_model_forward(
                                input=x, teacher_model=teacher_model)

                            loss = model.compute_loss(x, y, y_pred, sample_weight)
                            # _on_after_compute_loss(self, input, student_output, student_loss, teacher_output=None)
                            # TODO: check, combile
                            loss = distiller.on_after_compute_loss(
                                x, y_pred.logits, loss, teacher_outputs.logits)
                        model._validate_target_and_loss(y, loss)
                        # Run backwards pass.
                        optimizer = self.model.optimizer
                        optimizer.minimize(
                            loss,
                            model.trainable_variables,
                            tape=tape)
                        return model.compute_metrics(x, y, y_pred, sample_weight)

                    model.save_pretrained(get_filepath(TMPPATH, self.task_type, self.task_id), saved_model=True)

                    # re-build optimizer
                    opt_kwargs = {}
                    for k, v in self.model.optimizer.__dict__.items():
                        if not k.startswith('_'):
                            opt_kwargs[k] = v
                    optimizer = self.model.optimizer.__class__(**opt_kwargs)
                    if self.strategy:  # pragma: no cover
                        with self.strategy.scope():
                            model = model_cls.from_pretrained(get_filepath(TMPPATH, self.task_type, self.task_id))
                            model.compile(
                                    optimizer=optimizer,
                                    loss=self.model.loss,
                                    metrics=self.model.compiled_metrics._user_metrics
                                    )
                            model.train_step = train_step
                    else:
                        model.train_step = train_step
                        model.compile(
                            optimizer=optimizer,
                            loss=self.model.loss,
                            metrics=self.model.compiled_metrics._user_metrics)
                    self.model = model

                    distiller.model = os.path.join(TMPPATH, "saved_model/1")
                    distiller.model.model_type = "saved_model"
                    teacher_model.save_pretrained(TEACHERPATH, saved_model=True)
                    distiller.teacher_model = os.path.join(TEACHERPATH, "saved_model/1")
                    distiller.teacher_model.model_type = "saved_model"

                    if eval_func is not None:
                        self._eval_func = eval_func
                    else:
                        self._eval_func = self.builtin_eval_func
                    if train_func is not None:
                        self._train_func = train_func
                    else:
                        self._train_func = self.build_train_func

                    distiller.eval_func = self._eval_func
                    distiller.train_func = self._train_func
                    distiller.create_criterion()

                    self.component = self.distiller = distiller

                    opt_model = distiller.fit()
                    opt_model.save(self.args.output_dir)
                    return opt_model

            agent.create_distillers()
            # run flash_distillers
            ori_model = model
            if agent.flash_distillers:
                model = run_distillers(ori_model, agent.flash_distillers,
                                       agent.flash_train_steps,
                                       agent.flash_block_names)
            # run regular_distillers
            if agent.regular_distillers:
                model = run_distillers(ori_model,
                                       agent.regular_distillers,
                                       agent.regular_train_steps,
                                       agent.regular_block_names,
                                       presentation='regular distillation')
            return model.model

        def eval_func_builtin(model):
            """Get the build in evaluation function.

            Args:
                model (object): the input model
            """
            if self._eval_func:
                result = self._eval_func(model)
            else:
                result = self.builtin_eval_func(model)  # pragma: no cover
            return {'metric': result}

        agent.framework = 'tensorflow'
        agent.train_func = train_func \
            if train_func else train_func_builtin
        agent.eval_func = eval_func \
            if eval_func else eval_func_builtin
        # pylint: disable=E1101
        os.makedirs(self.args.output_dir, exist_ok=True)
        return agent.search(self.args.output_dir, model_cls)

    def build_train_func(self, model):
        """Build the training function for pruning or distillation.
        
        Args:
            model (object): the input model
        """
        tf.random.set_seed(1)
        epochs = 1

        component = self.component
        prune_model = self.model
        model_path = get_filepath(TMPPATH, self.task_type, self.task_id)

        if 'distillation' in self.component.cfg:
            epochs = max(epochs, self.component.cfg.distillation.train.get("epoch", 1))
            hooks = self.component.hooks
        if 'pruning' in self.component.cfg:
            epochs = max(epochs, self.component.cfg.pruning.train.get("epoch", 1))
            callbacks = self.pruner.callbacks
            hooks = callbacks['tf_pruning'](self.pruner.model, self.model,
                                            self.pruner.hooks)

        class callback(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                if version.parse(__version__) <= version.parse("1.12"):
                    hooks['pre_epoch_begin']()  # pragma: no cover
                else:
                    hooks['on_train_begin']()

            def on_train_end(self, logs=None):
                if version.parse(__version__) <= version.parse("1.12"):
                    hooks['post_epoch_end']()  # pragma: no cover
                else:
                    hooks['on_train_end']()

            def on_epoch_begin(self, epoch, logs=None):
                # pylint: disable=E1121
                hooks['on_epoch_begin'](epoch)

            def on_epoch_end(self, epoch, logs=None):
                component.model._session = None
                prune_model.save_pretrained(model_path, saved_model=True)
                component.model = os.path.join(model_path, "saved_model/1")
                component.model.model_type = "saved_model"
                component.model.sess
                hooks['on_epoch_end']()

            # pylint: disable=E1121
            def on_train_batch_begin(self, batch, logs=None):
                if version.parse(__version__) <= version.parse("1.12"):
                    hooks['on_batch_begin'](batch)  # pragma: no cover
                else:
                    hooks['on_step_begin'](batch)

            def on_train_batch_end(self, batch, logs=None):
                if version.parse(__version__) <= version.parse("1.12"):
                    hooks['on_batch_end']()  # pragma: no cover
                else:
                    hooks['on_step_end']()

        self.model.fit(self.train_dataset,
                       validation_data=self.eval_dataset,
                       epochs=epochs,
                       callbacks=[callback()])
        self.component.model._session = None
        self.model.save_pretrained(get_filepath(TMPPATH, self.task_type, self.task_id), saved_model=True)