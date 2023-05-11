import os

os.environ["DNNL_GRAPH_VERBOSE"] = "0"
import argparse
import gc
import pickle
import re
import time
import json
from pathlib import Path
import pathlib
import intel_extension_for_pytorch as ipex
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from intel_extension_for_pytorch.quantization import convert, prepare
from torch.ao.quantization import (MinMaxObserver, PerChannelMinMaxObserver,
                                   QConfig)
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, BloomForCausalLM, GPT2Tokenizer,
                          OPTForCausalLM)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="decapoda-research/llama-7b-hf"
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "xpu", "cuda"],
    help="cpu, xpu or cuda",
    default="cpu",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16", "float16", "int8"],
    help="bfloat16 or float32 or float16 or int8",
    default="bfloat16",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--dataset", nargs="?", default="lambada", const="lambada")
parser.add_argument("--split", nargs="?", default="validation", const="validation")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--ipex_dynamic_quantize", action="store_true")
parser.add_argument("--ipex_static_quantize", action="store_true")
parser.add_argument("--quantize_with_inc", action="store_true")
parser.add_argument("--ipex_smooth_quant", action="store_true")
parser.add_argument("--jit", action="store_true")
parser.add_argument("--int8", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--quantized_model_path", default="./saved_result/best_model.pt")
parser.add_argument("--lambada", action="store_true")
parser.add_argument("--accuracy_only", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--ir_path",
                    type=str,
                    help="path to bfloat16 or int8 IR files",
                    )
parser.add_argument("--model_type", default=None, type=str)
args = parser.parse_args()
layer_num = 32
if args.model_type == "13b":
    layer_num = 40
model = args.model
dataset_path = "../datasets/lambada/"
dataset = args.dataset
split = args.split
calib_size = 1
batch_size =1
device = "cpu"
args.dtype = "int8" if args.int8 or args.int8_bf16_mixed else args.dtype

import transformers
user_model = transformers.LlamaForCausalLM.from_pretrained(
    args.model,
    torchscript=args.jit,  # torchscript will force `return_dict=False` to avoid jit errors
    low_cpu_mem_usage=True
)
if args.dtype == "bfloat16" or args.dtype == "float32":
    user_model = ipex.optimize(user_model.eval(), dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float)
tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model)
print("Data type of the model:", user_model.dtype)
global_past_key_value = [(torch.zeros([1,user_model.config.num_attention_heads,1,int(user_model.config.hidden_size/user_model.config.num_attention_heads)]), 
                          torch.zeros([1,user_model.config.num_attention_heads,1,int(user_model.config.hidden_size/user_model.config.num_attention_heads)])) for i in range(user_model.config.num_hidden_layers)]

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
        attention_mask_padded = []
        for text in batch:
            # we cut the sentence if it exceeds pad_max, we are using tuned max 196 from gptj model; TODO: tune best pad_max 
            input_ids = text["input_ids"] if text["input_ids"].shape[0] <= self.pad_max else text["input_ids"][0:int(self.pad_max-1)]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids) + 1)
            attention_mask[0] = 0
            input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            input_ids_padded.append(input_ids)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            attention_mask_padded.append(attention_mask)

        return (
            (
                torch.vstack(input_ids_padded),
                tuple(global_past_key_value),
                torch.vstack(attention_mask_padded),
            ),
            torch.tensor(last_ind),
        )

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
     
        for i, (
            (input_ids, past_key_values, attention_mask),
            last_ind,
        ) in enumerate(test_dataloader):
            label = input_ids[torch.arange(len(last_ind)), last_ind]
            input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
            pad_len = self.pad_max - last_ind - 1
            start = time.time()
            past_k_v = [past_key_values[i][j] for i in range(layer_num) for j in range(2)]
            output = graph.inference([input_ids, attention_mask] + past_k_v)
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

# amp autocast
if args.dtype == "bfloat16":
    amp_enabled = True
    amp_dtype = torch.bfloat16
elif args.dtype == "float16":
    amp_enabled = True
    amp_dtype = torch.float16
else:
    amp_enabled = False
    amp_dtype = None

if args.device == 'xpu':
    autocast = torch.xpu.amp.autocast
elif args.device == 'cuda':
    autocast = torch.cuda.amp.autocast
else:
    autocast = torch.cpu.amp.autocast

if args.lambada:
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
        batch_size=1,
        shuffle=False,
        collate_fn=evaluator.collate_batch,
    )

    test_dataloader = DataLoader(
        evaluator.dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=evaluator.collate_batch,
    )

# beam search = 4
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

def calib_func(prepared_model):
    for i, (
        (input_ids, past_key_values, attention_mask),
        last_ind,
    ) in enumerate(calib_dataloader):
        if i == 8:
            break
        prepared_model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )


def eval_func(traced_model):
    acc, latency = evaluator.evaluate(traced_model)
    print("Accuracy:", acc)
    print("Latency (sec):", latency)
    return acc


if args.jit:
    torch._C._jit_set_texpr_fuser_enabled(False)
    generate_kwargs["jit"] = True
    if args.int8 or args.int8_bf16_mixed:
        generate_kwargs["ipex_int8"] = True
        generate_kwargs["quantized_model_path"] = args.quantized_model_path


if args.ipex_dynamic_quantize:
    example_inputs=None
    input_ids = torch.ones(32).to(torch.long)
    attention_mask = torch.ones(len(input_ids) + 1)
    attention_mask[0] = 0
    last_ind = input_ids.shape[0] - 1
    example_inputs=(input_ids.unsqueeze(0), tuple(global_past_key_value), attention_mask.unsqueeze(0))

    from intel_extension_for_pytorch.quantization import prepare, convert
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig,HistogramObserver
    qconfig = ipex.quantization.default_dynamic_qconfig
    prepared_model = prepare(user_model.eval(), qconfig, example_inputs=example_inputs)
    with torch.no_grad():
        convert_model = convert(prepared_model.eval()).eval()
        self_jit = torch.jit.trace(convert_model.eval(), example_inputs, strict=False)
        self_jit = torch.jit.freeze(self_jit.eval())
        self_jit.save(args.output_dir+"/ipex_dynamic_quantize_model.pt")


if args.ipex_static_quantize or args.ipex_smooth_quant:
    example_inputs=None
    for i, (
        (input_ids, past_key_values, attention_mask),
        last_ind,
    ) in enumerate(calib_dataloader):
        example_inputs=(input_ids, past_key_values, attention_mask)
        break
    from intel_extension_for_pytorch.quantization import prepare, convert
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig,HistogramObserver
    qconfig = ipex.quantization.default_static_qconfig
    if args.ipex_smooth_quant:
        qconfig = ipex.quantization.get_smooth_quant_static_qconfig()
    prepared_model = prepare(user_model.eval(), qconfig, example_inputs=example_inputs)
    with torch.no_grad():
        for i, (
            (input_ids, past_key_values, attention_mask),
            last_ind,
        ) in enumerate(calib_dataloader):
            if i == 8:
                break
            prepared_model(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask)
    with torch.no_grad(), autocast(enabled=amp_enabled or args.int8_bf16_mixed, dtype=amp_dtype):
        convert_model = convert(prepared_model.eval()).eval()
        if args.ipex_smooth_quant:
            convert_model(*example_inputs)
        self_jit = torch.jit.trace(convert_model.eval(), example_inputs, strict=False)
        self_jit = torch.jit.freeze(self_jit.eval())
        self_jit.save(args.output_dir+"/ipex_static_quantize_model.pt")

if args.quantize_with_inc:
    from neural_compressor import PostTrainingQuantConfig, quantization

    op_type_dict = {
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

    excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
    conf = PostTrainingQuantConfig(
        backend="ipex",
        excluded_precisions=excluded_precisions,
        op_type_dict=op_type_dict,
    )
    q_model = quantization.fit(
        user_model,
        conf,
        calib_dataloader=calib_dataloader,
        eval_func=eval_func,
        calib_func=calib_func,
    )


    q_model.save(args.output_dir)

if args.accuracy_only:
    # backend setting for dynamic quantization: torch.backends.quantized.engine = 'onednn', default is fbgemm
    if args.int8 or args.int8_bf16_mixed:
        user_model = torch.jit.load(
            args.quantized_model_path
        )
        user_model = torch.jit.freeze(user_model.eval())

    if args.jit and (args.dtype == "bfloat16" or args.dtype == "float32"):
        input_ids = torch.ones(32).to(torch.long)
        attention_mask = torch.ones(len(input_ids) + 1)
        attention_mask[0] = 0
        last_ind = input_ids.shape[0] - 1
        example_inputs=(input_ids.unsqueeze(0), tuple(global_past_key_value), attention_mask.unsqueeze(0))
        with torch.no_grad(), autocast(enabled=amp_enabled, dtype=amp_dtype):
            user_model = torch.jit.trace(user_model.eval(), example_inputs, strict=False)
            user_model = torch.jit.freeze(user_model.eval())

    with autocast(enabled=amp_enabled or args.int8_bf16_mixed, dtype=amp_dtype):
      eval_func(user_model)

if args.benchmark:
    # input prompt
    current_path = pathlib.Path(__file__).parent.resolve()
    with open(str(current_path) + '/prompt.json') as f:
        prompt_pool = json.load(f)
    if args.prompt is not None:
        prompt = args.prompt
    elif args.input_tokens in prompt_pool:
        prompt = prompt_pool[args.input_tokens]
    else:
        raise SystemExit('[ERROR] Plese use --prompt if want to use custom input.')

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)
    if input_size + args.max_new_tokens > 2049:
        raise SystemExit('[WARN] Token indices sequence length is longer than the specified maximum ' + 
                        'sequence length for this model (2049 > 2048). Running this sequence through the model will result in indexing errors')

    total_time = 0.0
    num_iter = 20
    num_warmup = 10

    with autocast(enabled=amp_enabled or args.int8_bf16_mixed, dtype=amp_dtype):
       for i in range(num_iter):
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            gen_tokens = user_model.generate(
                input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
            )
            gen_text = tokenizer.batch_decode(gen_tokens)[0]
            if args.device == "xpu":
                torch.xpu.synchronize()
            if args.device == "cuda":
                torch.cuda.synchronize()
            toc = time.time()
            print(gen_text, flush=True)
            if i >= num_warmup:
                total_time += toc - tic

    print("Inference latency: %.3f sec." % (total_time / (num_iter - num_warmup)))
