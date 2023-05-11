import os

os.environ["DNNL_GRAPH_VERBOSE"] = "0"
import numpy as np
import torch
from transformers import GPT2Tokenizer, OPTForCausalLM, AutoTokenizer, BloomForCausalLM
from torch.nn.functional import pad
from torch.utils.data import DataLoader

import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig

import time
from datasets import load_dataset, load_from_disk
import pickle

from pathlib import Path
import re
import gc

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--method", nargs="?", default="int8", const="int8")
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6B", const="facebook/opt-125m"
)
parser.add_argument("--dataset", nargs="?", default="lambada", const="lambada")
parser.add_argument("--split", nargs="?", default="validation", const="validation")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--accuracy_only", action="store_true")
parser.add_argument("--ir_path",
                    type=str,
                    help="path to bfloat16 or int8 IR files",
                    )
args = parser.parse_args()

model = args.model
fname_res = "../output/res_" + args.model + "_" + args.method + ".pkl"
dataset_path = "../datasets/lambada/"
dataset = args.dataset
split = args.split
calib_size = 8
batch_size = 8
device = "cpu"

class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, pad_val=1, pad_max=196):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max

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
        from intel_extension_for_transformers.backends.neural_engine.compile import compile
        graph = compile(args.ir_path)
        for i, (input_ids, last_ind) in enumerate(test_dataloader):
            label = input_ids[torch.arange(len(last_ind)), last_ind]
            input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
            pad_len = self.pad_max - last_ind - 1

            start = time.time()
            past_k_v =  np.ones([1,0,16,256]).astype(np.float32) 
            attmask = np.ones([input_ids.shape[0], input_ids.shape[1]]).astype(np.int32)
            newinput = input_ids.numpy()
            output = graph.inference([newinput] + [past_k_v for _ in range(2 * 28)] + [attmask])
            outputs = [torch.from_numpy(list(output.values())[0].reshape(input_ids.shape[0], input_ids.shape[1], -1))]
            latency += time.time() - start

            last_token_logits = outputs[0][torch.arange(len(last_ind)), -2 - pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if i % 50 == 0:
                print(hit / total)
                print("Processed minibatch:", i)

        acc = hit / total
        print(acc)
        lantecy = latency / len(self.dataset)
        return acc, lantecy


if re.search("bloom", model):
    user_model = BloomForCausalLM.from_pretrained(
        model, torchscript=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
if re.search("opt", model):
    user_model = OPTForCausalLM.from_pretrained(
        model, torchscript=True, low_cpu_mem_usage=True
    )
    tokenizer = GPT2Tokenizer.from_pretrained(model)

if re.search("gpt", model):
    import transformers

    user_model = transformers.AutoModelForCausalLM.from_pretrained(
        model,
        torchscript=True,  # torchscript will force `return_dict=False` to avoid jit errors
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

print("Data type of the model:", user_model.dtype)

if os.environ.get("HF_DATASETS_OFFLINE") == "1":
    dataset = load_from_disk(dataset_path)
    print("offline")
else:
    full_dataset = load_dataset(dataset)
    dataset = full_dataset["validation"]
    calib_dataset = full_dataset["train"]
    dataset.save_to_disk(dataset_path)

user_model.eval()
evaluator = Evaluator(dataset, tokenizer, batch_size)
calib_evaluator = Evaluator(calib_dataset, tokenizer, batch_size)

calib_dataloader = DataLoader(
    calib_evaluator.dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=evaluator.collate_batch,
)
calib_inputs = next(iter(calib_dataloader))[0]


def calib_func(prepared_model):
    prepared_model(calib_inputs)



def eval_func(traced_model):
    acc, latency = evaluator.evaluate(traced_model)
    print("Accuracy:", acc)
    print("Latency (sec):", latency)
    return acc


if args.quantize:
    from neural_compressor import PostTrainingQuantConfig
    from neural_compressor import quantization

    op_type_list = {
        "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        "linear": {
            "weight": {
                "dtype": ["int8"],
                "scheme": ["sym"],
                "granularity": ["per_channel"],
                "algorithm": ["minmax"],
            },
            "activation": {
                "dtype": ["uint8"],
                "scheme": ["asym"],
                "granularity": ["per_tensor"],
                "algorithm": ["kl"],
            },
        },
    }
    conf = PostTrainingQuantConfig(backend="ipex", op_type_list=op_type_list)
    conf.performance_only = True

    q_model = quantization.fit(
        user_model,
        conf,
        calib_dataloader=calib_dataloader,
        eval_func=eval_func,
        calib_func=calib_func,
    )

    q_model.save(args.output_dir)

if args.accuracy_only:
    user_model.eval()
    if args.int8:
        from neural_compressor.utils.pytorch import load

        user_model = load(os.path.abspath(os.path.expanduser(args.output_dir)), user_model)
    eval_func(user_model)
