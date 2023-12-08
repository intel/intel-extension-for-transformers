#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import argparse
from datasets import load_dataset
from intel_extension_for_transformers.setfit import SetFitModel, SetFitTrainer, sample_dataset
from intel_extension_for_transformers.setfit.utils import LOSS_NAME_TO_CLASS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sentence-transformers/paraphrase-mpnet-base-v2")
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--sample_sizes", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument(
        "--classifier",
        default="logistic_regression",
        choices=[
            "logistic_regression",
            "svc-rbf",
            "svc-rbf-norm",
            "knn",
            "pytorch",
            "pytorch_complex",
        ],
    )
    parser.add_argument("--loss", default="CosineSimilarityLoss")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--keep_body_frozen", default=False, action="store_true")
    parser.add_argument("--output_dir", type=str, default="setfit_model")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Configure loss function
    loss_class = LOSS_NAME_TO_CLASS[args.loss]

    # Load dataset
    dataset = load_dataset(args.dataset)
    train_dataset = sample_dataset(dataset["train"])
    eval_dataset = dataset["validation"] 

    # Load model
    if args.classifier == "pytorch":
        model = SetFitModel.from_pretrained(
            args.model,
            use_differentiable_head=True,
            head_params={"out_features": len(set(train_dataset["label"]))},
        )
    else:
        model = SetFitModel.from_pretrained(args.model)
    model.model_body.max_seq_length = args.max_seq_length

    # Train on current split
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_class=loss_class,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_iterations=args.num_iterations,
        column_mapping={"sentence": "text", "label": "label"},
    )
    if args.classifier == "pytorch":
        trainer.freeze()
        trainer.train()
        trainer.unfreeze(keep_body_frozen=args.keep_body_frozen)
        trainer.train(
            num_epochs=25,
            body_learning_rate=1e-5,
            learning_rate=args.lr,  # recommend: 1e-2
            l2_weight=0.0,
            batch_size=args.batch_size,
        )
    else:
        trainer.train()

    # Evaluate the model on the test data
    metrics = trainer.evaluate()
    print(f"Metrics: {metrics}")

    # Save the trained model
    model.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}.")

if __name__ == "__main__":
    main()