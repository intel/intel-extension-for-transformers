import argparse
import os
import time
import json
import re
import sys
sys.path.insert(0, '/home/hengguo/code/intel-extension-for-transformers')
import torch
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6b"
)
parser.add_argument(
    "--trust_remote_code", default=True,
    help="Transformers parameter: use the external repo")
parser.add_argument(
    "--revision", default=None,
    help="Transformers parameter: set the model hub commit number")
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--quantize", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="By default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--approach", type=str, default='static', 
                    help="Select from ['dynamic', 'static', 'weight_only']")
parser.add_argument("--sq", action="store_true")
parser.add_argument("--alpha", default="ll0.5",
                    help="Smooth quant parameter.")
parser.add_argument("--layer_wise", action="store_true")
parser.add_argument("--weight_only_algo", default="RTN", choices=['RTN', 'AWQ', 'TEQ', 'GPTQ'],
                    help="Weight-only parameter.")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--weight_only_sym_full_range", action="store_true")
parser.add_argument("--ipex", action="store_true", help="Use intel extension for pytorch.")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--pad_max_length", default=512, type=int,
                    help="Pad input ids to max length.")
parser.add_argument("--calib_iters", default=512, type=int,
                    help="calibration iters.")
parser.add_argument("--tasks", nargs='+', default=["lambada_openai",
    "hellaswag","winogrande","piqa","wikitext"],
    type=str, help="tasks list for accuracy validation")
parser.add_argument("--weight_only_bits", type=int, default=8)
parser.add_argument("--weight_only_group", type=int, default=-1)
parser.add_argument("--weight_only_scheme", default="sym")

args = parser.parse_args()
if args.ipex:
    import intel_extension_for_pytorch as ipex
calib_size = 500

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
        if args.weight_only_algo in ['AWQ', 'TEQ']:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            example = self.tokenizer(examples["text"], padding="max_length", max_length=self.pad_max)
        else:
            # if self.tokenizer.pad_token is None:
            #     if self.tokenizer.eos_token:
            #         self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token = ' '
            example = self.tokenizer(examples["text"], padding="max_length", max_length=self.pad_max)
            # example = self.tokenizer(examples["text"], truncation=True, max_length=self.pad_max)
            # example = self.tokenizer(examples["text"])
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

torchscript = False
if args.sq or args.weight_only_algo in ['AWQ', 'TEQ']:
    torchscript = True

from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, size=5, shape=128):
        self.len = size
        self.input_ids = torch.randint(low=0, high=30522, size=(size, shape), dtype=torch.int64)

    def __getitem__(self, index):
        return self.input_ids[index]

    def __len__(self):
        return self.len
    
from memory_profiler import profile

@profile(precision=4)
def run():
    if args.layer_wise:
        print('load empty shell model...')
        from neural_compressor.adaptor.torch_utils.layer_wise_quant import load_empty_model
        user_model = load_empty_model(args.model, torchscript=torchscript)
    else:
        user_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torchscript=torchscript,  # torchscript will force `return_dict=False` to avoid jit errors
            revision=args.revision
            )
    if re.search("llama", args.model):
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    # to channels last
    user_model = user_model.to(memory_format=torch.channels_last)
    user_model.eval()

    weight_only = args.approach == 'weight_only'

    if args.quantize:
        # dataset
        # print(f'load dataset from {args.dataset}')
        # calib_dataset = load_dataset(args.dataset, split="train")
        # calib_dataset = calib_dataset.shuffle(seed=42)
        # calib_evaluator = Evaluator(calib_dataset, tokenizer, args.batch_size, pad_max=args.pad_max_length, is_calib=True)
        # calib_dataloader = DataLoader(
        #     calib_evaluator.dataset,
        #     batch_size=calib_size,
        #     shuffle=False,
        #     collate_fn=calib_evaluator.collate_batch,
        # )

        calib_dataset = TestDataset()
        calib_dataloader = DataLoader(calib_dataset, batch_size=8)

        def calib_func(prepared_model):
            for i, calib_input in enumerate(calib_dataloader):
                if i > args.calib_iters:
                    break
                prepared_model(calib_input[0])

        recipes = {}
        from neural_compressor import PostTrainingQuantConfig, quantization
        if args.approach == 'weight_only':
            op_type_dict = {
                '.*':{ 	# re.match
                    "weight": {
                        'bits': args.weight_only_bits, # 1-8 bits 
                        'group_size': args.weight_only_group,  # -1 (per-channel)
                        'scheme': args.weight_only_scheme, # sym/asym
                        'algorithm': args.weight_only_algo, # RTN/AWQ/TEQ
                    },
                },
            }
            if args.weight_only_sym_full_range:
                recipes.update({"rtn_args": {"sym_full_range": True}})
        else:
            if re.search("gpt", user_model.config.model_type):
                op_type_dict = {
                    "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
                }
            else:
                op_type_dict = {}
        excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
        if args.layer_wise:
            recipes.update({
                "layer_wise_quant": True,
            })
        conf = PostTrainingQuantConfig(
            backend="ipex" if args.ipex else "default",
            approach=args.approach,
            excluded_precisions=excluded_precisions,
            op_type_dict=op_type_dict,
            recipes=recipes,
        )
        print(conf)

        if args.weight_only_algo == 'TEQ':
            # set calib_func=None, use default training func as calib_func
            calib_func = None

        st = time.time()
        print(f'start quantization, time: {st}')
        from neural_compressor import print_memo
        print_memo('start quantization')
        q_model = quantization.fit(
            user_model,
            conf,
            calib_dataloader=calib_dataloader,
            calib_func=calib_func,
        )

        print(f'finished quantizaiton, time: {time.time()}')
        # q_model.save(args.output_dir)
        print(f'finished save model, time: {time.time()}')

    if args.int8 or args.int8_bf16_mixed:
        if args.layer_wise:
            user_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torchscript=torchscript,  # torchscript will force `return_dict=False` to avoid jit errors
            revision=args.revision
            )
        print("load int8 model")
        from neural_compressor.utils.pytorch import load
        if args.ipex:
            user_model = load(os.path.abspath(os.path.expanduser(args.output_dir)))
        else:
            user_model = load(os.path.abspath(os.path.expanduser(args.output_dir)), user_model, weight_only=weight_only)
        user_model.eval()

    if args.accuracy:
        user_model.eval()
        from intel_extension_for_transformers.evaluation.lm_eval import evaluate
        results = evaluate(
            model="hf-causal",
            model_args='pretrained='+args.model+',tokenizer='+args.model+',dtype=float32',
            user_model=user_model,
            batch_size=args.batch_size,
            tasks=args.tasks,
        )
        dumped = json.dumps(results, indent=2)
        if args.save_accuracy_path:
            with open(args.save_accuracy_path, "w") as f:
                f.write(dumped)
        for task_name in args.tasks:
            if task_name == "wikitext":
                print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["word_perplexity"]))
            else:
                print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["acc"]))

if __name__ == '__main__':
    run()