#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from intel_extension_for_transformers.optimization import metrics, objectives, OptimizedModel, QuantizationConfig
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

os.environ["WANDB_DISABLED"] = "true"

""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.30.2")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

@dataclass
class OptimizationArguments:
    """
    Arguments pertaining to what type of optimization we are going to apply on the model.
    """

    tune: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply quantization."},
    )
    quantization_approach: Optional[str] = field(
        default="PostTrainingStatic",
        metadata={"help": "Quantization approach. Supported approach are PostTrainingStatic, "
                  "PostTrainingDynamic and QuantizationAwareTraining."},
    )
    metric_name: Optional[str] = field(
        default="eval_accuracy",
        metadata={"help": "Metric used for the tuning strategy."},
    )
    is_relative: Optional[bool] = field(
        default=True,
        metadata={"help": "Metric tolerance model, expected to be relative or absolute."},
    )
    perf_tol: Optional[float] = field(
        default=0.01,
        metadata={"help": "Performance tolerance when optimizing the model."},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "run benchmark."})
    int8: bool = field(
        default=False,
        metadata={"help":"run benchmark."})
    accuracy_only: bool = field(
        default=False,
        metadata={"help":"Whether to only test accuracy for model tuned by Neural Compressor."})
    load_dataset_from_file: Optional[str] = field(
        default=None,
        metadata={"help":"Whether to load dataset from local file"}
    )
    enable_bf16: bool = field(
        default=False,
        metadata={"help":"Whether or not to apply bf16."})

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, OptimizationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, optim_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, optim_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_image_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        if optim_args.load_dataset_from_file is not None:
            dataset = datasets.load_from_disk(optim_args.load_dataset_from_file)
        else:    
            dataset = load_dataset(
                data_args.dataset_name,
                cache_dir=model_args.cache_dir,
                task="image-classification",
                ignore_verifications=True,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            task="image-classification",
        )

    # If we don't have a validation split, split off a percentage of train as validation.
    # data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    data_args.train_val_split = None if "train" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["validation"].train_test_split(data_args.train_val_split, shuffle=False)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["validation"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if optim_args.int8:
        # Load the model obtained after Intel Neural Compressor (INC) quantization
        model = OptimizedModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        model = AutoModelForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        return example_batch

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(val_transforms)

    # Initalize our trainer
    trainer = NLPTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )

    metric_name = optim_args.metric_name

    if optim_args.tune:

        if not training_args.do_eval:
            raise ValueError("do_eval must be set to True for quantization.")

        trainer.save_model(training_args.output_dir)
        if optim_args.quantization_approach != "PostTrainingDynamic":
            if not training_args.do_train:
                raise ValueError(
                    "do_train must be set to True for static and aware training quantization."
                )
        if optim_args.quantization_approach == "QuantizationAwareTraining":
            early_stopping_patience = 6
            early_stopping_threshold = 0.001 # optional
            trainer.add_callback(transformers.EarlyStoppingCallback(early_stopping_patience,
                                                                    early_stopping_threshold))

        tune_metric = metrics.Metric(
            name=metric_name, is_relative=optim_args.is_relative, criterion=optim_args.perf_tol
        )
        objective = objectives.performance
        quantization_config = QuantizationConfig(
            approach=optim_args.quantization_approach,
            max_trials=600,
            metrics=[tune_metric],
            objectives=[objective],
            config_file='vit.yaml'
        )

        model = trainer.quantize(quant_config=quantization_config)


    if optim_args.benchmark or optim_args.accuracy_only:

        results = trainer.evaluate()
        print(results)
        logger.info("metrics keys: {}".format(results.keys()))
        bert_task_acc_keys = ['eval_f1', 'eval_accuracy', 'eval_matthews_correlation',
                              'eval_pearson', 'eval_mcc', 'eval_spearmanr']
        ret = False
        for key in bert_task_acc_keys:
            if key in results.keys():
                ret = True
                throughput = results.get("eval_samples_per_second")
                print('Batch size = {}'.format(training_args.per_device_eval_batch_size))
                print("Finally Eval {} Accuracy: {:.5f}".format(key, results[key]))
                print("Latency: {:.5f} ms".format(1000 / throughput))
                print("Throughput: {:.5f} samples/sec".format(throughput))
                break
        assert ret, "No metric returned, Please check inference metric!"
    if optim_args.enable_bf16:
        trainer.enable_bf16 = True
        
    trainer.enable_executor = True
    trainer.export_to_onnx()

if __name__ == "__main__":
    main()