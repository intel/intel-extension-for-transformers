import os
import unittest
from intel_extension_for_transformers.transformers.dpo_trainer import DPOTrainer

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
import copy
import torch
from peft import LoraConfig
import torch.utils.data as data


os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"
MODEL_NAME = "hf-internal-testing/tiny-random-GPTJForCausalLM"

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class DummyDataset(data.Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.max_prompt_length = 128
        self.max_length = 256
        prompt = "you are a AI assistant that created by Intel."
        chosen = "intel-extension-for-transformers is based in SH"
        reject = "Where is intel-extension-for-transformers based? NYC or SH"
        prompt_tokens = self.tokenizer.tokenize(prompt)
        if len(prompt_tokens) > self.max_prompt_length:
            prompt_tokens = prompt_tokens[:self.max_prompt_length]
        prompt_ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
        prompt_mask = [1] * len(prompt_ids)
        max_resp = self.max_length - len(prompt_ids)

        chosen_tokens = self.tokenizer.tokenize(chosen)
        chosen_tokens = chosen_tokens[:max_resp - 1]
        chosen_tokens.append(self.tokenizer.eos_token)
        chosen_ids = self.tokenizer.convert_tokens_to_ids(chosen_tokens)
        chosen_mask = [1] * len(chosen_ids)

        reject_tokens = self.tokenizer.tokenize(reject)
        reject_tokens = reject_tokens[:max_resp - 1]
        reject_tokens.append(self.tokenizer.eos_token)
        reject_ids = self.tokenizer.convert_tokens_to_ids(reject_tokens)
        reject_mask = [1] * len(reject_ids)

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_attention_mask = prompt_mask + chosen_mask
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids

        reject_input_ids = prompt_ids + reject_ids
        reject_attention_mask = prompt_mask + reject_mask
        reject_labels = [-100] * len(prompt_ids) + reject_ids

        self.encoded_dict = {}

        self.encoded_dict["chosen_input_ids"] = chosen_input_ids
        self.encoded_dict["chosen_attention_mask"] = chosen_attention_mask
        self.encoded_dict["chosen_labels"] = chosen_labels
        self.encoded_dict["rejected_input_ids"] = reject_input_ids
        self.encoded_dict["rejected_attention_mask"] = reject_attention_mask
        self.encoded_dict["rejected_labels"] = reject_labels

    def __len__(self):
        return 10

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        if index < 10:
            return self.encoded_dict


class TestDPO(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, load_in_4bit=True, low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.model_ref = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, load_in_4bit=True, low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.train_dataset = DummyDataset()
        self.eval_dataset = DummyDataset()
        # 4. initialize training arguments:
        training_args = TrainingArguments(
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                max_steps=10,
                logging_steps=1,
                save_steps=10,
                gradient_accumulation_steps=1,
                learning_rate=5e-4,
                evaluation_strategy="steps",
                eval_steps=5,
                output_dir="./tmp_dpo",
                bf16=True,
                remove_unused_columns=False,
                )

        target_modules = find_all_linear_names(self.model)
        peft_config = LoraConfig(
                r=16,
                lora_alpha=8,
                lora_dropout=0.05,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
                )

        self.trainer = DPOTrainer(
                self.model,
                self.model_ref,
                args=training_args,
                data_collator=self.collate_fn,
                beta=0.01,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.train_dataset.tokenizer,
                peft_config=peft_config,
                max_length=self.train_dataset.max_length,
                )

    @staticmethod
    def collate_fn(batch):
        input_ids = [torch.tensor(ins["chosen_input_ids"]) for ins in batch] +\
                [torch.tensor(ins["rejected_input_ids"]) for ins in batch]
        labels = [torch.tensor(ins["chosen_labels"]) for ins in batch] +\
                [torch.tensor(ins["rejected_labels"]) for ins in batch]
        attention_mask = [torch.tensor(ins["chosen_attention_mask"]) for ins in batch] +\
                [torch.tensor(ins["rejected_attention_mask"]) for ins in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=2)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    def test_dpo(self):
        self.trainer.train()

if __name__ == "__main__":
    unittest.main()
