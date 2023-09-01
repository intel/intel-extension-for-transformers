import os
import psutil
import argparse
import time
import json
from pathlib import Path
import pathlib

from datasets import load_dataset, load_from_disk
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader

import intel_extension_for_pytorch as ipex


parser = argparse.ArgumentParser("LLaMA generation script (int8 path)", add_help=False)
parser.add_argument(
    "-m", "--model-id", default=None, type=str, required=True, help="your llama model"
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu"],
    help="cpu",
    default="cpu",
)
parser.add_argument("--dtype", type=str, default="int8")
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="lambada")
parser.add_argument("--split", nargs="?", default="validation", const="validation")
parser.add_argument("--output-dir", nargs="?", default="./saved_results")
parser.add_argument("--ipex-smooth-quant", action="store_true")
parser.add_argument("--jit", action="store_true")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--greedy", action="store_true")
parser.add_argument(
    "--int8-bf16-mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--quantized-model-path", default="./saved_results/best_model.pt")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
args = parser.parse_args()


# disable
try:
    ipex._C.disable_jit_linear_repack()
except Exception:
    pass

device = torch.device(args.device)
args.dtype = "int8" if args.int8 or args.int8_bf16_mixed else args.dtype

# amp autocast
if args.int8_bf16_mixed:
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = torch.float32

# beam numbers
num_beams = 1 if args.greedy else 4


# load model
config = AutoConfig.from_pretrained(args.model_id, torchscript=args.jit)

user_model = LlamaForCausalLM.from_pretrained(
        args.model_id, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float
    )

tokenizer = LlamaTokenizer.from_pretrained(args.model_id)
print("Data type of the model:", user_model.dtype)
# calling _optimize_transformers for int8 path
user_model = ipex._optimize_transformers(
    user_model.eval(), dtype=torch.int8, inplace=True
)
# dummy past key value
beam_idx_tmp = torch.zeros(
    (2048, int(args.batch_size * num_beams)), dtype=torch.long
).contiguous()
global_past_key_value = [
    (
        torch.zeros(
            [
                1,
                user_model.config.num_attention_heads,
                1,
                int(
                    user_model.config.hidden_size
                    / user_model.config.num_attention_heads
                ),
            ]
        ).contiguous(),
        torch.zeros(
            [
                1,
                user_model.config.num_attention_heads,
                1,
                int(
                    user_model.config.hidden_size
                    / user_model.config.num_attention_heads
                ),
            ]
        ).contiguous(),
        beam_idx_tmp,
        torch.zeros(1, dtype=torch.long).contiguous(),
    )
    for i in range(user_model.config.num_hidden_layers)
]


class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=1, pad_val=1, pad_max=512):
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
        position_ids_padded = []
        input_ids_padded = []
        last_ind = []
        attention_mask_padded = []
        for text in batch:
            # we cut the sentence if it exceeds pad_max, we are using tuned max 196 from gptj model; TODO: tune best pad_max
            input_ids = (
                text["input_ids"]
                if text["input_ids"].shape[0] <= self.pad_max
                else text["input_ids"][0 : int(self.pad_max - 1)]
            )
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            position_ids = torch.arange(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            input_ids_padded.append(input_ids)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            attention_mask_padded.append(attention_mask)
            position_ids = pad(position_ids, (0, pad_len), value=self.pad_val)
            position_ids_padded.append(position_ids)
        return (
            (
                torch.vstack(input_ids_padded),
                torch.vstack(attention_mask_padded),
                torch.vstack(position_ids_padded),
                tuple(global_past_key_value),
            ),
            torch.tensor(last_ind),
        )



calib_dataset = load_dataset(args.dataset)["train"]
user_model.eval()
calib_evaluator = Evaluator(calib_dataset, tokenizer, args.batch_size)

calib_dataloader = DataLoader(
    calib_evaluator.dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=calib_evaluator.collate_batch,
)




if args.ipex_smooth_quant:
    example_inputs = None
    for i, (
        (input_ids, attention_mask, position_ids, past_key_values),
        last_ind,
    ) in enumerate(calib_dataloader):
        example_inputs = (input_ids, attention_mask, position_ids, past_key_values)
        break
    from intel_extension_for_pytorch.quantization import prepare, convert

    qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=1.0)
    prepared_model = prepare(user_model.eval(), qconfig, example_inputs=example_inputs, inplace=True)
    with torch.no_grad():
        for i, (
            (input_ids, attention_mask, position_ids, past_key_values),
            last_ind,
        ) in enumerate(calib_dataloader):
            if i == 100:
                break
            prepared_model(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
    with torch.no_grad(), torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=torch.bfloat16 if amp_enabled else None,
    ):
        convert_model = convert(prepared_model.eval(), inplace=True).eval()
        self_jit = torch.jit.trace(convert_model.eval(), example_inputs, strict=False)
        self_jit = torch.jit.freeze(self_jit.eval())
        os.makedirs(args.output_dir, exist_ok=True)
        self_jit.save(args.output_dir + "/best_model.pt")



