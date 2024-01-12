#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import transformers
from itertools import chain
from dataclasses import dataclass, field
import evaluate
import math
import json
import sigopt
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    pipeline,
    TrainingArguments,
    Trainer,
    set_seed,
    GPTJForCausalLM,
)
from typing import Optional, List

#import torch.distributed as dist
#import torch

os.environ["WANDB_DISABLED"] = "true"
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    validation_split_percentage: Optional[int] = field(
        default=20,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )

    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep in memory the loaded dataset. Defaults to False."
        },
    )
    dataset_seed: int = field(
        default=42,
        metadata={
            "help": "Seed to use in dataset processing, different seeds might yield different datasets. This seed and the seed in training arguments are not related"
        },
    )
    dataset_cache_directory: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory where the processed dataset will be saved. If path exists, try to load processed dataset from this path."
        }
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

    dataset_concatenation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to concatenate the sentence for more efficient training."
        }
    )


@dataclass
class FinetuneArguments:
    """
    Arguments of finetune we are going to apply on the model.
    """

    lora_rank: int = field(
        default=8,
        metadata={
            "help": "Rank parameter in the LoRA method."
        },
    )
    lora_alpha: int = field(
        default=32,
        metadata={
            "help": "Alpha parameter in the LoRA method."
        },
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout parameter in the LoRA method."
        },
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={
            "help": "Target modules for the LoRA method."
        },
    )


def training_causal_text(example):
    hypothesis = example["hypothesis"]
    premise = example["premise"]
    class_label = ["entailment", "neutral", "contradiction"][example["label"]]

    example["text"] = f"mnli hypothesis: {hypothesis} premise: {premise} target: {class_label}<|endoftext|>"
    return example


def validation_causal_text(example):
    hypothesis = example["hypothesis"]
    premise = example["premise"]

    example["text"] = f"mnli hypothesis: {hypothesis} premise: {premise} target:"
    # class_label = ["entailment", "neutral", "contradiction"][example["label"]]
    # example["label_id"] = class_label
    return example


def tokenize_text_fn(tokenizer: AutoTokenizer):
    def func(dataset):
        tokenized = tokenizer(dataset["text"], return_attention_mask=False)
        return tokenized

    return func


def tokenize_validation(dataset, tokenizer):
    tokenized_examples = []
    for example in dataset["text"]:
        tokenized_example = tokenizer.encode(example, return_tensors="pt").squeeze()
        tokenized_examples.append(tokenized_example)
    return {"input_ids": tokenized_examples, "label": dataset["label"]}


def batch_causal_input(seq_len):
    def func(examples):
        # Concatenate all texts.
        inputs = list(chain(*examples["input_ids"]))
        total_length = len(inputs)
        # We drop the small remainder instead of padding
        if total_length >= seq_len:
            new_total_length = (total_length // seq_len) * seq_len
            # Make sure to leave at least some elements from the end
            if new_total_length == total_length:
                new_total_length -= seq_len
            total_length = new_total_length

        input_batches = [inputs[i: i + seq_len] for i in range(0, total_length, seq_len)]
        label_batches = [inputs[i: i + seq_len] for i in range(1, total_length + 1, seq_len)]
        return {
            "input_ids": input_batches,
            "labels": label_batches,
        }

    return func


def extract_class_label(s: str) -> str:
    """Extract a class label from the generated text.
    This is done by matching the label as there is often no space between the label and subsequent output."""
    s = s.strip()
    # no need if decoded using skip special tokens
    s = s.replace("<|endoftext|>", " ")
    class_labels = ["entailment", "neutral", "contradiction"]
    words = s.split()
    words = words[-5:]
    for word in words:
        if word in class_labels:
            return word
    return "unknown"


def postprocess_mnli_predictions(generated_sentences):
    labels_to_ids = {"entailment": 0, "neutral": 1, "contradiction": 2, "unknown": -1}
    predictions = []
    for s in generated_sentences:
        generated_sentence = s[0]['generated_text']
        answer = extract_class_label(generated_sentence)
        predictions.append(labels_to_ids[answer])
    return predictions


#def print_module_names_recur(module):
#    for name,mod in module.named_modules():
#        print("Name of modules : ", name)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, FinetuneArguments))
    #model_args, data_args, training_args, finetune_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args, finetune_args, optim_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    b16 = training_args.fp16 or training_args.bf16
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}"
        + f"\ndistributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {b16}"
    )
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    #sigopt.log_dataset("GPTJ MNLI fine tuning with LoRA and 30k data v2")
    #sigopt.log_model("GPTJ")
    #sigopt.params.setdefault("learning_rate", 0.0002)
    #sigopt.params.setdefault("target_modules", "q_proj,v_proj")
    #sigopt.params.setdefault("epochs", 3)
    #sigopt.params.setdefault("lora_alpha", 54)
    #sigopt.params.setdefault("temperature", 0.4)
    #sigopt.params.setdefault("top_p", 0.8)
    #sigopt.params.setdefault("top_k", 70)
    
    #training_args.learning_rate = sigopt.params.learning_rate    
    #training_args.num_train_epochs = sigopt.params.epochs    
    #target_modules = sigopt.params.target_modules.split(",")
    
    # Get the datasets: You can just provide the name of one of the public datasets available on the hub at
    # https://huggingface.co/datasets/ -the dataset will be downloaded automatically from the datasets Hub).
    #
    # Downloading and loading a dataset from the hub.
    train_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        split="train[:80000]",
    )

    print("Raw_datasets type :", type(train_dataset))
    print("One sample row in train dataset raw : ", train_dataset[0])

    train_dataset = train_dataset.map(
        training_causal_text,
        remove_columns=["hypothesis", "premise", "label", "idx"],
        load_from_cache_file=True,
        desc="Generating training causal text",
    )
    print("One sample row in train dataset after causal text transform : ", train_dataset[0])

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    tokenizer.add_special_tokens({"pad_token": "<|extratoken_1|>"})

    train_dataset = train_dataset.map(
        tokenize_text_fn(tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=True,
        desc="Tokenizing training text",
    )
    print("One sample row in train dataset after tokenize : ", train_dataset[0])

    train_dataset = train_dataset.map(
        batch_causal_input(512),
        batched=True,
        batch_size=1000,
        num_proc=1,
        load_from_cache_file=True,
        desc="Packing sequences",
    )

    print("One sample row in train dataset after group : ", train_dataset[0])

    eval_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        split="validation_matched[:500]",
    )
    eval_dataset = eval_dataset.map(
        validation_causal_text,
        remove_columns=["hypothesis", "premise", "idx"],
        load_from_cache_file=True,
        desc="Generating validation causal text",
    )
    print("eval dataset sample row after causal transform : ", eval_dataset[0])

    eval_prompts = eval_dataset["text"]
    eval_labels = eval_dataset["label"]
    eval_dataset = eval_dataset.map(
         tokenize_validation,
         batched=True,
         remove_columns=eval_dataset.column_names,
         load_from_cache_file=False,
         fn_kwargs={"tokenizer": tokenizer},
     )
    # print("validation dataset row 1 after tokenize : ", eval_dataset[0])

    # Load model
    #config = model_args.config_name
    #config = transformers.PretrainedConfig.from_json_file(model_args.config_name)
    config = transformers.GPTJConfig.from_json_file(model_args.config_name)
    #config = json.load(model_args.config_name)
    #config = transformers.GPTJConfig.get_config_dict(model_args.config_name)
    #config = transformers.PretrainedConfig.to_dict(config_orig)
    print("Config orig", config)
    config.temperature = 0.2
    config.top_k = 70
    config.top_p = 0.4
    config.task_specific_params["text-generation"]["temperature"] = 0.2
    config.task_specific_params["text-generation"]["top_k"] = 70
    config.task_specific_params["text-generation"]["top_p"] = 0.4
    print("config new", config)
    

    model = GPTJForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config = config,
        cache_dir=model_args.cache_dir,
    )


    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=finetune_args.lora_rank,
        #lora_alpha=sigopt.params.lora_alpha,
        lora_alpha=54,
        lora_dropout=finetune_args.lora_dropout,
        #target_modules=target_modules,
        target_modules=["q_proj","v_proj","k_proj","out_proj"],#"fc_in","fc_out"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        #modules_to_save=["act","ln_f"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # predictions = trainer.predict(eval_dataset)

    print("Fine-tuning the model")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    print("Saving the fine-tuned model to ", training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config = config,cache_dir=model_args.cache_dir)
    
    model = PeftModel.from_pretrained(model,training_args.output_dir)


    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=5)
    eval_preds = generator(eval_prompts, pad_token_id=50256)
    print("A sample eval pred : ", eval_preds[0], " <Original> : ", eval_prompts[0])

    accuracy_metric = evaluate.load("accuracy")
    results = accuracy_metric.compute(references=eval_labels, predictions=postprocess_mnli_predictions(eval_preds))
    sigopt.log_metric("accuracy", results['accuracy'])
    print(results)



if __name__ == "__main__":
    main()
