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


from datasets import load_dataset, Features, Value, ClassLabel
from transformers import AutoTokenizer

from utils import Benchmark


class DlsaFinetune(object):
    def __init__(self, **kwargs):
        self.args = kwargs["args"]
        self.training_args = kwargs["training_args"]

        self.max_train, self.max_test = (
            self.args.max_train_samples,
            self.args.max_test_samples,
        )
        if self.args.smoke_test:
            self.max_train, self.max_test = 100, 100

        self.bench = Benchmark()
        self.track = self.bench.track

    def e2e_finetune(self):
        with self.track("Total Run"):
            self._load_data()
            self._preprocess()
            self._load_model()
            self._do_finetune()
            self._do_infer()
            # self.bench.summary()

    def _load_data(self):
        with self.track("Load Data"):
            self.remove_columns = []
            self.text_column = []
            self.num_labels = 2
            if self.args.dataset == "local":
                dataset_cfg = self.args.local_dataset
                class_names = dataset_cfg["label_list"]
                self.num_labels = len(class_names)
                customer_features = {v: k for k, v in dataset_cfg["features"].items()}

                for key, value in customer_features.items():
                    if value == "class_label":
                        customer_features[key] = ClassLabel(names=class_names)
                    elif value == "data_column":
                        customer_features[key] = Value("string")
                        self.text_column = key
                        self.remove_columns.append(key)
                    else:
                        customer_features[key] = Value("string")
                        self.remove_columns.append(key)

                features = Features(customer_features)

                train_all = load_dataset(
                    "csv",
                    data_files=dataset_cfg["finetune_input"],
                    delimiter=dataset_cfg["delimiter"],
                    features=features,
                    split="train",
                )
                test_all = load_dataset(
                    "csv",
                    data_files=dataset_cfg["inference_input"],
                    delimiter=dataset_cfg["delimiter"],
                    features=features,
                    split="train",
                )
                train_data = (
                    train_all.select(range(self.max_train))
                    if self.max_train
                    else train_all
                )
                test_data = (
                    test_all.select(range(self.max_test)) if self.max_test else test_all
                )

                label2id = {class_names[i]: i for i in range(len(class_names))}
                self.train_data = train_data.align_labels_with_mapping(
                    label2id, "label"
                )
                self.test_data = test_data.align_labels_with_mapping(label2id, "label")

            else:
                data = load_dataset(self.args.dataset)
                train_all = data["train"]
                test_split = "validation" if self.args.dataset == "sst2" else "test"
                len_train = len(train_all)
                self.train_data = (
                    train_all.select(range(len_train - self.max_train, len_train))
                    if self.max_train
                    else train_all
                )
                self.test_data = (
                    data[test_split].select(range(self.max_test))
                    if self.max_test
                    else data[test_split]
                )
                self.text_column = [
                    c
                    for c in self.test_data.column_names
                    if type(self.test_data[c][0]) != int
                ][0]
                self.remove_columns = [self.text_column] + (
                    ["idx"] if self.args.dataset == "sst2" else []
                )

    def _preprocess(self):
        with self.track("Pre-process"):
            with self.track("----Init tokenizer"):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args.tokenizer_name
                    if self.args.tokenizer_name
                    else self.args.model_name_or_path
                )

            max_seq_len = min(self.args.max_seq_len, self.tokenizer.model_max_length)

            with self.track("----Tokenize + Extract Features"):

                def preprocess(examples):
                    return self.tokenizer(
                        examples[self.text_column],
                        padding="max_length",
                        truncation=True,
                        max_length=max_seq_len,
                    )

                kwargs = dict(
                    function=preprocess,
                    batched=True,
                    num_proc=self.args.preprocessing_num_workers,
                    remove_columns=self.remove_columns,
                    load_from_cache_file=not self.args.overwrite_cache,
                )

                self.train_data = (
                    self.train_data.map(**kwargs)
                    if self.training_args.do_train
                    else None
                )
                self.test_data = (
                    self.test_data.map(**kwargs)
                    if self.training_args.do_predict
                    else None
                )

    def _load_model(self):
        raise NotImplementedError

    def _do_finetune(self):
        raise NotImplementedError

    def _do_infer(self):
        raise NotImplementedError
