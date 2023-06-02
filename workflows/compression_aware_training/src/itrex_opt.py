# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#


import functools
import logging
import os
import random
import sys

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric

# Need to use itrex domain toolkit
from intel_extension_for_transformers.optimization import (
    DistillationConfig,
    PrunerConfig,
    PruningConfig,
    OptimizedModel,
    QuantizationConfig,
    metrics,
    objectives,
)
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Need to use Huggingface transformers package
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

from utils import (
    DataTrainingArguments,
    ModelArguments,
    OptimizationArguments,
    task_to_keys,
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoModelForSequenceClassification

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.0")

logger = logging.getLogger(__name__)


class ItrexOpt(object):
    def __init__(self, args):
        if int(os.environ.get("LOCAL_RANK", -1)) != -1 and "--no_cuda" in args:
            from intel_extension_for_transformers.optimization.utils.utility import (
                distributed_init,
            )

            distributed_init()

        parser = HfArgumentParser(
            (
                ModelArguments,
                DataTrainingArguments,
                TrainingArguments,
                OptimizationArguments,
            )
        )
        
        if "--local-rank=0" not in args:
            if len(args) == 2 and args[1].endswith(".yaml"):
                model_args, data_args, training_args, optim_args = parser.parse_yaml_file(
                    yaml_file=os.path.abspath(args[1])
                )
            elif len(args) == 2 and args[1].endswith(".json"):
                model_args, data_args, training_args, optim_args = parser.parse_json_file(
                    json_file=os.path.abspath(args[1])
                )
            else:
                (
                    model_args,
                    data_args,
                    training_args,
                    optim_args,
                ) = parser.parse_args_into_dataclasses()
        else:
            filename = None
            for arg in args:
                if arg.endswith(".yaml") or arg.endswith(".json"):
                    filename = arg
                    break
            if filename is None:
                (
                    model_args,
                    data_args,
                    training_args,
                    optim_args,
                ) = parser.parse_args_into_dataclasses()
            elif filename.endswith(".yaml"):
                model_args, data_args, training_args, optim_args = parser.parse_yaml_file(
                    yaml_file=os.path.abspath(filename)
                )
            else:
                model_args, data_args, training_args, optim_args = parser.parse_json_file(
                    json_file=os.path.abspath(filename)
                )


        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.optim_args = optim_args

    def e2e(self):
        self._prepare_env()
        self._load_data()
        self._load_model()
        self._preprocess()
        if self.optim_args.distillation:
            self._do_distillation()
        if self.optim_args.quantization:
            self._do_quantization_aware_training()
        if self.optim_args.sat:
            self._do_sparsity_aware_training()

    def _prepare_env(self):
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {self.training_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if (
            os.path.isdir(self.training_args.output_dir)
            and self.training_args.do_train
            and not self.training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if (
                last_checkpoint is None
                and len(os.listdir(self.training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({self.training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and self.training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Set seed before initializing model.
        set_seed(self.training_args.seed)

    def _load_data(self):
        if self.data_args.task_name is not None:

            if self.data_args.task_name == "emotion":
                raw_datasets = load_dataset(f"SetFit/{self.data_args.task_name}")
            else:
                # Downloading and loading a dataset from the hub.
                raw_datasets = load_dataset(
                    "glue",
                    self.data_args.task_name,
                    cache_dir=self.model_args.cache_dir,
                )
        elif self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.model_args.cache_dir,
            )
        else:
            # Loading a dataset from your local files.
            # CSV/JSON training and evaluation files are needed.
            data_files = {
                "train": self.data_args.train_file,
                "validation": self.data_args.validation_file,
            }

            for key in data_files.keys():
                logger.info(f"load a local file for {key}: {data_files[key]}")

            if self.data_args.train_file.endswith(".csv"):
                # Loading a dataset from local csv files
                raw_datasets = load_dataset(
                    "csv", data_files=data_files, cache_dir=self.model_args.cache_dir
                )
            else:
                # Loading a dataset from local json files
                raw_datasets = load_dataset(
                    "json", data_files=data_files, cache_dir=self.model_args.cache_dir
                )
        # See more about loading any type of standard or custom dataset at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        print("Step 1: Load the dataset")
        print("#######################")
        # Load the dataset Labels
        if self.data_args.task_name is not None:
            is_regression = self.data_args.task_name == "stsb"
            if not is_regression:
                if self.data_args.task_name == "emotion":
                    label_list = list(set(raw_datasets["train"]["label_text"]))
                else:
                    label_list = raw_datasets["train"].features["label"].names
                num_labels = len(label_list)
            else:
                num_labels = 1
        else:
            # Trying to have good defaults here, don't hesitate to tweak to your needs.
            is_regression = raw_datasets["train"].features["label"].dtype in [
                "float32",
                "float64",
            ]
            if is_regression:
                num_labels = 1
            else:
                # A useful fast method:
                # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
                label_list = raw_datasets["train"].unique("label")
                label_list.sort()  # Let's sort it for determinism
                num_labels = len(label_list)

        self.raw_datasets = raw_datasets
        self.is_regression = is_regression
        self.num_labels = num_labels
        self.label_list = label_list

    def _load_model(self):
        config = AutoConfig.from_pretrained(
            self.model_args.config_name
            if self.model_args.config_name
            else self.model_args.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=self.data_args.task_name,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        self.config = config

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name
            if self.model_args.tokenizer_name
            else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        self.tokenizer = tokenizer

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        if self.optim_args.distillation or self.optim_args.sat:
            teacher_config = AutoConfig.from_pretrained(
                self.optim_args.teacher_model_name_or_path,
                num_labels=self.num_labels,
                finetuning_task=self.data_args.task_name,
            )
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                self.optim_args.teacher_model_name_or_path,
                use_fast=self.model_args.use_fast_tokenizer,
            )
            assert (
                teacher_tokenizer.vocab == self.tokenizer.vocab
            ), "teacher model and student model should have same tokenizer."
            teacher_model = AutoModelForSequenceClassification.from_pretrained(
                self.optim_args.teacher_model_name_or_path,
                from_tf=bool(".ckpt" in self.optim_args.teacher_model_name_or_path),
                config=teacher_config,
            )
            teacher_model.to(self.training_args.device)

            self.teacher_tokenizer = teacher_tokenizer
            self.teacher_model = teacher_model

        if self.optim_args.int8:
            # Load the model obtained after Intel Neural Compressor (INC) quantization
            model = OptimizedModel.from_pretrained(
                self.model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                config=config,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

        self.model = model

    def _preprocess(self):
        # Preprocessing the raw_datasets
        if self.data_args.task_name is not None:
            sentence1_key, sentence2_key = task_to_keys[self.data_args.task_name]
        else:
            # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
            non_label_column_names = [
                name
                for name in self.raw_datasets["train"].column_names
                if name != "label"
            ]
            if (
                "sentence1" in non_label_column_names
                and "sentence2" in non_label_column_names
            ):
                sentence1_key, sentence2_key = "sentence1", "sentence2"
            else:
                if len(non_label_column_names) >= 2:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
                else:
                    sentence1_key, sentence2_key = non_label_column_names[0], None

        # Padding strategy
        if self.data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        if (
            self.model.config.label2id
            != PretrainedConfig(num_labels=self.num_labels).label2id
            and self.data_args.task_name is not None
            and not self.is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {
                k.lower(): v for k, v in self.model.config.label2id.items()
            }
            if list(sorted(label_name_to_id.keys())) == list(sorted(self.label_list)):
                label_to_id = {
                    i: int(label_name_to_id[self.label_list[i]])
                    for i in range(self.num_labels)
                }
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(self.label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif self.data_args.task_name is None and not self.is_regression:
            label_to_id = {v: i for i, v in enumerate(self.label_list)}

        if label_to_id is not None:
            self.model.config.label2id = label_to_id
            self.model.config.id2label = {
                id: label for label, id in self.config.label2id.items()
            }
        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        max_seq_length = min(
            self.data_args.max_seq_length, self.tokenizer.model_max_length
        )

        def preprocess_function(examples, tokenizer=self.tokenizer):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(
                *args,
                padding=padding,
                max_length=max_seq_length,
                truncation=True,
            )

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [
                    (label_to_id[l] if l != -1 else -1) for l in examples["label"]
                ]
            return result

        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = self.raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )
        if self.training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]

            if self.data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(
                    range(self.data_args.max_train_samples)
                )

        if self.training_args.do_eval:
            if (
                "validation" not in raw_datasets
                and "validation_matched" not in raw_datasets
            ):
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets[
                "validation_matched"
                if self.data_args.task_name == "mnli"
                else "validation"
            ]
            if self.data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(
                    range(self.data_args.max_eval_samples)
                )

        # Log a few random samples from the training set:
        if self.training_args.do_train:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(
                    f"Sample {index} of the training set: {train_dataset[index]}."
                )

        # Get the metric function
        if (
            self.data_args.task_name is not None
            and self.data_args.task_name != "emotion"
        ):
            metric = load_metric("glue", self.data_args.task_name)
        else:
            metric = load_metric("accuracy")

        self.metric = metric

        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.

        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        if self.data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif self.training_args.fp16:
            data_collator = DataCollatorWithPadding(
                self.tokenizer, pad_to_multiple_of=8
            )
        else:
            data_collator = None

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        if self.optim_args.distillation or self.optim_args.sat:
            # prepare datasets for teacher model
            teacher_processed_datasets = self.raw_datasets.map(
                functools.partial(
                    preprocess_function, tokenizer=self.teacher_tokenizer
                ),
                batched=True,
                remove_columns=self.raw_datasets["train"].column_names,
            )
            teacher_train_dataset = teacher_processed_datasets["train"]
            if self.data_args.max_train_samples is not None:
                teacher_train_dataset = teacher_train_dataset.select(
                    range(self.data_args.max_train_samples)
                )
            teacher_eval_dataset = teacher_processed_datasets[
                "validation_matched"
                if self.data_args.task_name == "mnli"
                else "validation"
            ]
            if self.data_args.max_eval_samples is not None:
                teacher_eval_dataset = teacher_eval_dataset.select(
                    range(self.data_args.max_eval_samples)
                )
            assert (
                self.train_dataset.num_rows == teacher_train_dataset.num_rows
                and self.eval_dataset.num_rows == teacher_eval_dataset.num_rows
            ), "Length of train or evaluation dataset of teacher doesnot match that of student."

            self.teacher_train_dataset = teacher_train_dataset
            self.teacher_eval_dataset = teacher_eval_dataset

        def compute_metrics(p: EvalPrediction):
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            preds = (
                np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
            )
            if self.data_args.task_name is not None:
                result = self.metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif self.is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {
                    "accuracy": (preds == p.label_ids).astype(np.float32).mean().item()
                }

        # Initialize and setup our itrexTrainer
        from neural_compressor.adaptor.torch_utils.symbolic_trace import symbolic_trace
        self.model = symbolic_trace(self.model, self.optim_args.quantization_approach=="QuantizationAwareTraining")

        self.trainer = NLPTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

    def _do_distillation(self):
        class BertModelforLogitsOutputOnly(torch.nn.Module):
            def __init__(self, model):
                super(BertModelforLogitsOutputOnly, self).__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                output = self.model(*args, **kwargs)
                return output["logits"]

        # #############################################################################################
        print(
            "Step 4: Inference teacher model: get logits for usage in distilling child model. (bert-mini)"
        )
        print(
            "###############################################################################################"
        )

        # get logits of teacher model
        def dict_tensor_to_model_device(batch, model):
            device = next(model.parameters()).device
            for k in batch:
                batch[k] = batch[k].to(device)

        def get_logits(teacher_model, train_dataset, teacher_train_dataset):
            logger.info(
                "***** Inferencing teacher model to Get logits of teacher *****"
            )
            logger.info(f"  Num examples = {len(train_dataset) }")
            teacher_model.eval()
            npy_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "{}.{}.npy".format(
                    self.data_args.task_name,
                    self.optim_args.teacher_model_name_or_path.replace("/", "."),
                ),
            )
            if os.path.exists(npy_file):
                teacher_logits = [x for x in np.load(npy_file)]
            else:
                sampler = None
                if self.training_args.world_size > 1:
                    from transformers.trainer_pt_utils import ShardSampler

                    sampler = ShardSampler(
                        teacher_train_dataset,
                        batch_size=self.training_args.per_device_eval_batch_size,
                        num_processes=self.training_args.world_size,
                        process_index=self.training_args.process_index,
                    )
                    teacher_model = torch.nn.parallel.DistributedDataParallel(
                        teacher_model,
                        device_ids=[self.training_args.local_rank]
                        if self.training_args._n_gpu != 0
                        else None,
                        output_device=self.training_args.local_rank
                        if self.training_args._n_gpu != 0
                        else None,
                    )
                train_dataloader = DataLoader(
                    teacher_train_dataset,
                    collate_fn=self.data_collator,
                    sampler=sampler,
                    batch_size=self.training_args.per_device_eval_batch_size,
                )
                train_dataloader = tqdm(train_dataloader, desc="Evaluating")
                teacher_logits = []
                for step, batch in enumerate(train_dataloader):
                    dict_tensor_to_model_device(batch, teacher_model)
                    outputs = teacher_model(**batch)
                    if self.training_args.world_size > 1:
                        outputs_list = [
                            None for i in range(self.training_args.world_size)
                        ]
                        torch.distributed.all_gather_object(outputs_list, outputs)
                        outputs = torch.concat(outputs_list, dim=0)
                    teacher_logits += [x for x in outputs.cpu().numpy()]
                if self.training_args.world_size > 1:
                    teacher_logits = teacher_logits[: len(teacher_train_dataset)]
                if self.training_args.local_rank in [-1, 0]:
                    np.save(npy_file, np.array(teacher_logits))
            return train_dataset.add_column("teacher_logits", teacher_logits)

        with torch.no_grad():
            self.train_dataset = get_logits(
                BertModelforLogitsOutputOnly(self.teacher_model),
                self.train_dataset,
                self.teacher_train_dataset,
            )

        para_counter = lambda model: sum(p.numel() for p in model.parameters())
        logger.info(
            "***** Number of teacher model parameters: {:.2f}M *****".format(
                para_counter(self.teacher_model) / 10**6
            )
        )
        logger.info(
            "***** Number of student model parameters: {:.2f}M *****".format(
                para_counter(self.model) / 10**6
            )
        )

        # #############################################################################################
        print(
            "Step 6: Distill teacher model to student Model "
        )
        print(
            "###############################################"
        )

        metric_name = (
            self.optim_args.metric_name
            if self.optim_args.metric_name is not None
            else "eval_"
            + (
                "pearson"
                if self.data_args.task_name == "stsb"
                else "matthews_correlation"
                if self.data_args.task_name == "cola"
                else "accuracy"
            )
        )
        # #############################################################################################
        print(
            "Step 7: Do the actual Distillation using itrex to get the distilled student model (Bert Mini)"
        )
        print(
            "# #############################################################################################"
        )

        # Initialize and setup our itrexTrainer
        if self.optim_args.distillation:

            if not self.training_args.do_eval:
                raise ValueError("do_eval must be set to True for distillation.")

            tune_metric = metrics.Metric(name=metric_name)
            distillation_conf = DistillationConfig(metrics=tune_metric)
            model = self.trainer.distill(
                distillation_config=distillation_conf, teacher_model=self.teacher_model
            )
            self.trainer.save_model(self.training_args.output_dir)

        # ############################################################
        print(
            "Step 8: run inference on distilled student Model for accuracy (Bert Mini)"
        )
        print(
            "#########################################################################"
        )

        # Check Accuracy
        if (
            self.optim_args.benchmark
            or self.optim_args.accuracy_only
            or self.optim_args.distillation
        ):
            # Load the model obtained after distillation
            # Can we do QAT also? Intel Neural Compressor (INC) quantization
            model = OptimizedModel.from_pretrained(
                self.training_args.output_dir,
            )
            model.eval()
            self.trainer.model = model
            results = self.trainer.evaluate()  # Actual Inference for accuracy
            logger.info("metrics keys: {}".format(results.keys()))
            bert_task_acc_keys = [
                "eval_f1",
                "eval_accuracy",
                "eval_matthews_correlation",
                "eval_pearson",
                "eval_mcc",
                "eval_spearmanr",
            ]
            ret = False
            for key in bert_task_acc_keys:
                if key in results.keys():
                    ret = True
                    throughput = results.get("eval_samples_per_second")
                    print(
                        "Batch size = ", self.training_args.per_device_eval_batch_size
                    )
                    print("Final Eval {} Accuracy: {}".format(key, results[key]))
                    print("Latency: {:.5f} ms".format(1000 / throughput))
                    print("Throughput: {:.5f} samples/sec".format(throughput))
            assert ret, "No metric returned, Please check inference metric!"

    def _do_quantization_aware_training(self):
        metric_name = (
            self.optim_args.metric_name
            if self.optim_args.metric_name is not None
            else "eval_"
            + (
                "pearson"
                if self.data_args.task_name == "stsb"
                else "matthews_correlation"
                if self.data_args.task_name == "cola"
                else "accuracy"
            )
        )

        if not self.training_args.do_eval:
            raise ValueError("do_eval must be set to True for quantization.")

        self.trainer.save_model(self.training_args.output_dir)
        if self.optim_args.quantization_approach != "PostTrainingDynamic":
            if not self.training_args.do_train:
                raise ValueError(
                    "do_train must be set to True for static and aware training quantization."
                )
            elif self.optim_args.quantization_approach == "QuantizationAwareTraining":
                early_stopping_patience = 6
                early_stopping_threshold = 0.001  # optional
                # trainer.add_callback(transformers.EarlyStoppingCallback(early_stopping_patience,
                #                                                    early_stopping_threshold))

        tune_metric = metrics.Metric(
            name=metric_name,
            is_relative=self.optim_args.is_relative,
            criterion=self.optim_args.perf_tol,
        )
        objective = objectives.performance
        quantization_config = QuantizationConfig(
            approach=self.optim_args.quantization_approach,
            max_trials=600,
            metrics=[tune_metric],
            objectives=[objective],
            sampling_size=len(self.train_dataset) // 20,
        )
        model = self.trainer.quantize(quant_config=quantization_config)

        if self.optim_args.benchmark or self.optim_args.accuracy_only:

            results = self.trainer.evaluate()
            logger.info("metrics keys: {}".format(results.keys()))
            bert_task_acc_keys = [
                "eval_f1",
                "eval_accuracy",
                "eval_matthews_correlation",
                "eval_pearson",
                "eval_mcc",
                "eval_spearmanr",
            ]
            ret = False
            for key in bert_task_acc_keys:
                if key in results.keys():
                    ret = True
                    throughput = results.get("eval_samples_per_second")
                    print(
                        "Batch size = {}".format(
                            self.training_args.per_device_eval_batch_size
                        )
                    )
                    print("Finally Eval {} Accuracy: {:.5f}".format(key, results[key]))
                    print("Latency: {:.5f} ms".format(1000 / throughput))
                    print("Throughput: {:.5f} samples/sec".format(throughput))
                    break
            assert ret, "No metric returned, Please check inference metric!"

    def _do_sparsity_aware_training(self):
        class BertModelforLogitsOutputOnly(torch.nn.Module):
            def __init__(self, model):
                super(BertModelforLogitsOutputOnly, self).__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                output = self.model(*args, **kwargs)
                return output["logits"]

        # #############################################################################################
        print(
            "Step 4: Inference teacher model: get logits for usage in pruning child model."
        )
        print(
            "###############################################################################################"
        )

        # get logits of teacher model
        def dict_tensor_to_model_device(batch, model):
            device = next(model.parameters()).device
            for k in batch:
                batch[k] = batch[k].to(device)

        def get_logits(teacher_model, train_dataset, teacher_train_dataset):
            logger.info(
                "***** Inferencing teacher model to Get logits of teacher *****"
            )
            logger.info(f"  Num examples = {len(train_dataset) }")
            teacher_model.eval()
            npy_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "{}.{}.npy".format(
                    self.data_args.task_name,
                    self.optim_args.teacher_model_name_or_path.replace("/", "."),
                ),
            )
            if os.path.exists(npy_file):
                teacher_logits = [x for x in np.load(npy_file)]
            else:
                sampler = None
                if self.training_args.world_size > 1:
                    from transformers.trainer_pt_utils import ShardSampler

                    sampler = ShardSampler(
                        teacher_train_dataset,
                        batch_size=self.training_args.per_device_eval_batch_size,
                        num_processes=self.training_args.world_size,
                        process_index=self.training_args.process_index,
                    )
                    teacher_model = torch.nn.parallel.DistributedDataParallel(
                        teacher_model,
                        device_ids=[self.training_args.local_rank]
                        if self.training_args._n_gpu != 0
                        else None,
                        output_device=self.training_args.local_rank
                        if self.training_args._n_gpu != 0
                        else None,
                    )
                train_dataloader = DataLoader(
                    teacher_train_dataset,
                    collate_fn=self.data_collator,
                    sampler=sampler,
                    batch_size=self.training_args.per_device_eval_batch_size,
                )
                train_dataloader = tqdm(train_dataloader, desc="Evaluating")
                teacher_logits = []
                for step, batch in enumerate(train_dataloader):
                    dict_tensor_to_model_device(batch, teacher_model)
                    outputs = teacher_model(**batch)
                    if self.training_args.world_size > 1:
                        outputs_list = [
                            None for i in range(self.training_args.world_size)
                        ]
                        torch.distributed.all_gather_object(outputs_list, outputs)
                        outputs = torch.concat(outputs_list, dim=0)
                    teacher_logits += [x for x in outputs.cpu().numpy()]
                if self.training_args.world_size > 1:
                    teacher_logits = teacher_logits[: len(teacher_train_dataset)]
                if self.training_args.local_rank in [-1, 0]:
                    np.save(npy_file, np.array(teacher_logits))
            return train_dataset.add_column("teacher_logits", teacher_logits)

        with torch.no_grad():
            self.train_dataset = get_logits(
                BertModelforLogitsOutputOnly(self.teacher_model),
                self.train_dataset,
                self.teacher_train_dataset,
            )

        para_counter = lambda model: sum(p.numel() for p in model.parameters())
        logger.info(
            "***** Number of teacher model parameters: {:.2f}M *****".format(
                para_counter(self.teacher_model) / 10**6
            )
        )
        logger.info(
            "***** Number of student model parameters: {:.2f}M *****".format(
                para_counter(self.model) / 10**6
            )
        )

        # #############################################################################################
        print(
            "Step 6: Prune teacher model to student Model"
        )
        print(
            "#####################################################################################"
        )

        metric_name = (
            self.optim_args.metric_name
            if self.optim_args.metric_name is not None
            else "eval_"
            + (
                "pearson"
                if self.data_args.task_name == "stsb"
                else "matthews_correlation"
                if self.data_args.task_name == "cola"
                else "accuracy"
            )
        )
        # #############################################################################################
        print(
            "Step 7: Do the actual Pruning using itrex to get the pruned student model"
        )
        print(
            "# #############################################################################################"
        )

        # Initialize and setup our itrexTrainer
        if self.optim_args.sat and self.optim_args.orchestrate_optimizations:

            if not self.training_args.do_train:
                raise ValueError("do_train must be set to True for pruning.")

            tune_metric = metrics.Metric(
                name=metric_name, is_relative=self.optim_args.is_relative, criterion=self.optim_args.perf_tol
            )
            prune_type = 'PatternLock' \
                if self.optim_args.pruning_approach else self.optim_args.pruning_approach
            target_sparsity_ratio = self.optim_args.target_sparsity_ratio \
                if self.optim_args.target_sparsity_ratio else None
            pruner_config = PrunerConfig(prune_type=prune_type, target_sparsity_ratio=target_sparsity_ratio)
            pruning_conf = PruningConfig(framework="pytorch_fx",pruner_config=[pruner_config], metrics=tune_metric)
            distillation_conf = DistillationConfig(framework="pytorch_fx", metrics=tune_metric)
        
            objective = objectives.performance
            quantization_conf = QuantizationConfig(
                approach=self.optim_args.quantization_approach,
                max_trials=600,
                metrics=[tune_metric],
                objectives=[objective]
            )
            conf_list = [pruning_conf, distillation_conf, quantization_conf]
            model = self.trainer.orchestrate_optimizations(config_list=conf_list, teacher_model=self.teacher_model)

        # ############################################################
        print(
            "Step 8: run inference on pruned student Model for accuracy"
        )
        print(
            "#########################################################################"
        )

        # Check Accuracy
        if (
            self.optim_args.benchmark
            or self.optim_args.accuracy_only
            or self.optim_args.sat
        ):
            # Load the model obtained after distillation
            # Can we do QAT also? Intel Neural Compressor (INC) quantization
            model = OptimizedModel.from_pretrained(
                self.training_args.output_dir,
            )
            model.eval()
            self.trainer.model = model
            results = self.trainer.evaluate()  # Actual Inference for accuracy
            logger.info("metrics keys: {}".format(results.keys()))
            bert_task_acc_keys = [
                "eval_f1",
                "eval_accuracy",
                "eval_matthews_correlation",
                "eval_pearson",
                "eval_mcc",
                "eval_spearmanr",
            ]
            ret = False
            for key in bert_task_acc_keys:
                if key in results.keys():
                    ret = True
                    throughput = results.get("eval_samples_per_second")
                    print(
                        "Batch size = ", self.training_args.per_device_eval_batch_size
                    )
                    print("Final Eval {} Accuracy: {}".format(key, results[key]))
                    print("Latency: {:.5f} ms".format(1000 / throughput))
                    print("Throughput: {:.5f} samples/sec".format(throughput))
            assert ret, "No metric returned, Please check inference metric!"
