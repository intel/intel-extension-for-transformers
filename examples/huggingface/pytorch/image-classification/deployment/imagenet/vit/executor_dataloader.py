#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

import math
import numpy as np
from sympy import false
import transformers
import datasets
from datasets import load_dataset
from dataclasses import dataclass, field
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    TrainingArguments,
)
from torch.utils.data import DataLoader
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

class dataloader_wrapper(object):
    def __init__(self, batch_size, image_processor_name, data_dir = './cached-2k-imagenet-1k-datasets'):
        self.batch_size = batch_size       

        dataset = datasets.load_from_disk(data_dir)
        
        train_val_split = 0.15
        # If we don't have a validation split, split off a percentage of train as validation.
        train_val_split = None if "train" in dataset.keys() else train_val_split
        if isinstance(train_val_split, float) and train_val_split > 0.0:
            print("************dataset split***************")
            split = dataset["validation"].train_test_split(train_val_split, shuffle=False)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]

        labels = dataset["validation"].features["labels"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        image_processor = AutoImageProcessor.from_pretrained(
            image_processor_name,
            cache_dir=None,
            revision="main",
            use_auth_token=False,
        )
        # Define torchvision transforms to be applied to each image.
        if "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        else:
            size = (image_processor.size["height"], image_processor.size["width"])
        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

        _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

        def val_transforms(example_batch):
            """Apply _val_transforms across a batch."""
            example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
            return example_batch

        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        
        dataset["validation"].set_transform(val_transforms)

        self.eval_data = DataLoader(
            dataset["validation"], collate_fn=collate_fn, batch_size=self.batch_size)

    def get_eval_data(self):
        return self.eval_data
