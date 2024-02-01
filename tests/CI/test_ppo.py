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

import os
import unittest

from transformers import (
    AutoTokenizer,
    pipeline,
)
import copy
import torch
from peft import LoraConfig
import torch.utils.data as data
from torch.utils.data import DataLoader

from intel_extension_for_transformers.transformers.ppo_core import (
    LengthSampler,
    set_seed,
)
from intel_extension_for_transformers.transformers.ppo_config import PPOConfig
from intel_extension_for_transformers.transformers.ppo_trainer import PPOTrainer
from intel_extension_for_transformers.transformers.modeling.trl_models import (
    AutoModelForCausalLMWithValueHead,
)
from huggingface_hub import PyTorchModelHubMixin
from tqdm import tqdm

MODEL_NAME = "hf-internal-testing/tiny-random-GPTJForCausalLM"
REWARD_NAME = "hf-internal-testing/tiny-random-GPTJForSequenceClassification"
os.environ["ACCELERATE_USE_IPEX"] = "false"

class DummyDataset(data.Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_prompt_length = 128
        self.max_length = 256
        question = "you are a AI assistant that created by Intel."
        chosen = "intel-extension-for-transformers is based in SH"

        self.encoded_dict = {}

        query = "Question: " + question + "\n\nAnswer: "
        tokenized_question = self.tokenizer(query, truncation=True)

        self.encoded_dict["query"] = query
        self.encoded_dict["input_ids"] = torch.tensor(tokenized_question["input_ids"])

    def __len__(self):
        return 10

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        if index < 10:
            return self.encoded_dict


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


class TestPPO(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            MODEL_NAME,
            peft_config=lora_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(torch.bfloat16)
        self.config = PPOConfig(
            steps=10,
            model_name=MODEL_NAME,
            learning_rate=1.41e-5,
            log_with=None,
            batch_size=8,
            mini_batch_size=1,
            gradient_accumulation_steps=8,
            optimize_device_cache=True,
            early_stopping=True,
            target_kl=0.1,
            ppo_epochs=4,
            seed=100,
            init_kl_coef=0.2,
            adap_kl_ctrl=True,
            use_habana=False,
            pad_for_acceleration=False,
            pad_max_len=512,
            pad_max_input_len=128,
        )
        self.dataset = DummyDataset()
        self.trainer = PPOTrainer(
            self.config,
            self.model,
            ref_model=None,
            tokenizer=self.dataset.tokenizer,
            dataset=self.dataset,
            data_collator=collator,
            optimizer=None,
        )
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=REWARD_NAME,
            tokenizer=self.dataset.tokenizer,
            return_token_type_ids=False,
            device="cpu",
            model_kwargs={
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.bfloat16,
            },
        )

    def test_init(self):
        self.trainer = PPOTrainer(
            self.config,
            self.model,
            ref_model=None,
            tokenizer=self.dataset.tokenizer,
            dataset=self.dataset,
            data_collator=collator,
            optimizer=None,
        )

        self.assertTrue(isinstance(self.trainer, PyTorchModelHubMixin))

    def test_generation_batched(self):
        generation_kwargs = {
            # "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.dataset.tokenizer.pad_token_id,
            "eos_token_id": 100_000,
        }
        epochs = tqdm(
            enumerate(self.trainer.dataloader),
            total=len(self.trainer.dataloader),
            desc="rl progress",
        )
        for epoch, batch in epochs:
            question_tensors = batch["input_ids"]
            response_tensors, ref_response= self.trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=LengthSampler(100, 120),
                generate_ref_response= True,
                **generation_kwargs,
            )
            batch["response"] = self.dataset.tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )

    def test_generation(self):
        generation_kwargs = {
            # "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.dataset.tokenizer.pad_token_id,
            "eos_token_id": 100_000,
        }
        epochs = tqdm(
            enumerate(self.trainer.dataloader),
            total=len(self.trainer.dataloader),
            desc="rl progress",
        )
        for epoch, batch in epochs:
            question_tensors = batch["input_ids"]
            for question_tensor in question_tensors:
                response_tensor, ref_response= self.trainer.generate(
                    question_tensor,
                    return_prompt=False,
                    length_sampler=LengthSampler(100, 120),
                    generate_ref_response= True,
                    **generation_kwargs,
                )
                response = self.dataset.tokenizer.batch_decode(
                    response_tensor, skip_special_tokens=True
                )

    def test_train(self):
        generation_kwargs = {
            # "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.dataset.tokenizer.pad_token_id,
            "eos_token_id": 100_000,
        }
        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "truncation": True,
        }
        epochs = tqdm(
            enumerate(self.trainer.dataloader),
            total=len(self.trainer.dataloader),
            desc="rl progress",
        )
        for epoch, batch in epochs:
            question_tensors = batch["input_ids"]
            response_tensors = self.trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=LengthSampler(100, 120),
                **generation_kwargs,
            )
            batch["response"] = self.dataset.tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )
            # Compute reward score (using the sentiment analysis pipeline)
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = self.sentiment_pipe(texts, **sent_kwargs)
            rewards = [
                torch.tensor(output[0]["score"] - 0.0) for output in pipe_outputs
            ]
            # Run PPO step
            stats = self.trainer.step(question_tensors, response_tensors, rewards)
            self.trainer.log_stats(stats, batch, rewards)

        self.trainer.save_pretrained("/tmp/output")

    def test_train_with_pad_and_custom_config(self):
        self.config.pad_for_acceleration = True
        self.config.adap_kl_ctrl = False
        self.config.use_score_scaling = True
        self.config.whiten_rewards = True
        self.config.kl_penalty = "full"
        self.config.max_grad_norm = 1.0
        self.config.early_stopping = True
        self.config.use_score_norm = True
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
        self.trainer = PPOTrainer(
            self.config,
            self.model,
            ref_model=self.model,
            tokenizer=self.dataset.tokenizer,
            dataset=self.dataset,
            data_collator=collator,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
        )
        generation_kwargs = {
            # "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.dataset.tokenizer.pad_token_id,
            "eos_token_id": 100_000,
        }
        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "truncation": True,
        }
        epochs = tqdm(
            enumerate(self.trainer.dataloader),
            total=len(self.trainer.dataloader),
            desc="rl progress",
        )
        for epoch, batch in epochs:
            question_tensors = batch["input_ids"]
            response_tensors = self.trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=LengthSampler(100, 120),
                **generation_kwargs,
            )
            batch["response"] = self.dataset.tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )
            # Compute reward score (using the sentiment analysis pipeline)
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = self.sentiment_pipe(texts, **sent_kwargs)
            rewards = [
                torch.tensor(output[0]["score"] - 0.0) for output in pipe_outputs
            ]
            # Run PPO step
            stats = self.trainer.step(question_tensors, response_tensors, rewards)
            self.trainer.log_stats(stats, batch, rewards)

        self.trainer.save_pretrained("/tmp/ppo_output")

    def test_train_no_peft(self):
        generation_kwargs = {
            # "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.dataset.tokenizer.pad_token_id,
            "eos_token_id": 100_000,
        }
        sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "truncation": True,
        }
        epochs = tqdm(
            enumerate(self.trainer.dataloader),
            total=len(self.trainer.dataloader),
            desc="rl progress",
        )
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.trainer = PPOTrainer(
            self.config,
            self.model,
            tokenizer=self.dataset.tokenizer,
            dataset=self.dataset,
            data_collator=collator,
        )
        for epoch, batch in epochs:
            question_tensors = batch["input_ids"]
            response_tensors = self.trainer.generate(
                question_tensors,
                return_prompt=False,
                **generation_kwargs,
            )
            batch["response"] = self.dataset.tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )
            # Compute reward score (using the sentiment analysis pipeline)
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = self.sentiment_pipe(texts, **sent_kwargs)
            rewards = [
                torch.tensor(output[0]["score"] - 0.0) for output in pipe_outputs
            ]
            # Run PPO step
            stats = self.trainer.step(question_tensors, response_tensors, rewards)
            self.trainer.log_stats(stats, batch, rewards)

if __name__ == "__main__":
    unittest.main()
