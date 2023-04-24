import argparse
import os
import time
import json
import fnmatch
import re
import numpy as np
import torch
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import intel_extension_for_pytorch as ipex
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6b"
)
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--quantize", action="store_true")
parser.add_argument(
    "--ipex_bf16",
    action="store_true",
    help="to enable ipex amp bf16 (work on platforms like SPR)",
)
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--sq", action="store_true")
parser.add_argument("--alpha", default="auto",
                    help="Smooth quant parameter.")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--accuracy_only", action="store_true")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--pad_max_length", default=196, type=int,
                    help="Pad input ids to max length.")

args = parser.parse_args()
calib_size = 1


class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, pad_val=1, pad_max=196, is_calib=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max
        self.is_calib = is_calib

        # tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def tokenize_function(self, examples):
        example = self.tokenizer(examples["text"])
        return example

    @torch.no_grad()
    def collate_batch(self, batch):

        input_ids_padded = []
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            if self.is_calib:
                input_ids = input_ids[:self.pad_max] if len(input_ids) > self.pad_max else input_ids
            else:
                input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            input_ids_padded.append(input_ids)

        return (torch.vstack(input_ids_padded), torch.tensor(last_ind))

    @torch.no_grad()
    def evaluate(self, model):

        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        latency = 0
        test_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        for i, (input_ids, last_ind) in enumerate(test_dataloader):
            label = input_ids[torch.arange(len(last_ind)), last_ind]
            input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
            pad_len = self.pad_max - last_ind - 1

            start = time.time()
            outputs = model(input_ids)
            latency += time.time() - start

            last_token_logits = outputs[0][torch.arange(len(last_ind)), -2 - pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if i % 50 == 0:
                print(hit / total)
                print("Processed minibatch:", i)

        acc = hit / total
        print("Accuracy: ", acc)
        lantecy = latency / len(self.dataset)
        print("Latency: ", latency)
        return acc

user_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torchscript=True,  # torchscript will force `return_dict=False` to avoid jit errors
)
tokenizer = AutoTokenizer.from_pretrained(args.model)


# to channels last
user_model = user_model.to(memory_format=torch.channels_last)
user_model.eval()


if args.quantize:
    # dataset
    calib_dataset = load_dataset(args.dataset, split="train")
    calib_dataset = calib_dataset.shuffle(seed=42)
    calib_evaluator = Evaluator(calib_dataset, tokenizer, args.batch_size, pad_max=args.pad_max_length, is_calib=True)
    calib_dataloader = DataLoader(
        calib_evaluator.dataset,
        batch_size=calib_size,
        shuffle=False,
        collate_fn=calib_evaluator.collate_batch,
    )

    def calib_func(prepared_model):
        for i, calib_input in enumerate(calib_dataloader):
            if i > 100:
                break
            prepared_model(calib_input[0])

    from neural_compressor import PostTrainingQuantConfig, quantization
    if re.search("gpt", args.model):
        op_type_dict = {
            "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        }
    else:
        op_type_dict = {}
    excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
    if args.sq:
        args.alpha = args.alpha if args.alpha == "auto" else float(args.alpha)
        recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': args.alpha}}
        conf = PostTrainingQuantConfig(
            backend="ipex",
            excluded_precisions=excluded_precisions,
            op_type_dict=op_type_dict,
            recipes=recipes,
        )
    else:
        conf = PostTrainingQuantConfig(
            backend="ipex",
            excluded_precisions=excluded_precisions,
            op_type_dict=op_type_dict,
        )

    q_model = quantization.fit(
        user_model,
        conf,
        calib_dataloader=calib_dataloader,
        calib_func=calib_func,
    )

    q_model.save(args.output_dir)
    exit(0)

amp_enabled = False
amp_dtype = None
if args.ipex_bf16:
    amp_enabled = True
    amp_dtype = torch.bfloat16
    user_model = ipex.optimize(user_model, dtype=torch.bfloat16)
    with torch.no_grad(), torch.cpu.amp.autocast():
        user_model = torch.jit.trace(user_model, user_model.dummy_inputs["input_ids"])
        user_model = torch.jit.freeze(user_model)
    user_model.eval()

if args.int8 or args.int8_bf16_mixed:
    print("load int8 model")
    from neural_compressor.utils.pytorch import load
    user_model = load(os.path.abspath(os.path.expanduser(args.output_dir)))
    user_model.eval()

if args.accuracy_only:
    user_model.eval()
    def eval_func(user_model):
        from intel_extension_for_transformers.evaluation.lm_evaluation_harness.evaluator import evaluate
        results = evaluate(
            model="hf-causal",
            model_args='pretrained='+args.model+',tokenizer='+args.model+',dtype=float32',
            user_model=user_model,
            batch_size=args.batch_size,
            tasks=["lambada_openai"]
        )
        dumped = json.dumps(results, indent=2)
        if args.save_accuracy_path:
            with open(args.save_accuracy_path, "w") as f:
                f.write(dumped)
        print('Accuracy for lambada_openai is ', results["results"]["lambada_openai"]["acc"])

    with torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
        eval_func(user_model)



