from transformers import AutoTokenizer, GPTJModel, AutoModelForCausalLM
import torch
import argparse
import os
import sys

parser = argparse.ArgumentParser('GPT-J Generation ir', add_help=False)
parser.add_argument("--model",
        type=str,
        help="path to bfloat16 or int8 IR files",
        default="EleutherAI/gpt-j-6B",
    )
parser.add_argument('--dtype', default=None, type=str)
parser.add_argument('--output_model', default="./ir", type=str)
parser.add_argument('--model_type', default="gpt-j", type=str)
parser.add_argument('--pt_file', type=str)
args = parser.parse_args()
print(args)

model_id = args.model
model_type = args.model_type
if 'llama' in model_type:
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Once upon a time, there existed a little girl, who liked to have adventures." + \
        " She wanted to go to places and meet new people, and have fun."
init_input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
input_ids = init_input_ids.clone()
attention_mask = torch.ones(len(input_ids)+1)
attention_mask[0] = 0
past_key_value_torch = tuple([(torch.zeros([1,16,32,256]), torch.zeros([1,16,32,256])) for i in range(28)])
input_ids = input_ids[0:1].unsqueeze(0)
attention_mask = attention_mask.unsqueeze(0)


traced_model = None
if 'llama' in model_type:
    past_key_value_torch = tuple([(torch.zeros([1,32,34,128]), torch.zeros([1,32,34,128])) for i in range(32)])
if 'llama_13b' in model_type:
    past_key_value_torch = tuple([(torch.zeros([1,40,34,128]), torch.zeros([1,40,34,128])) for i in range(40)])

if args.pt_file and os.path.exists(args.pt_file):
    print('PT model exists, compile will be executed.')
    traced_model = torch.jit.load(args.pt_file)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, return_dict=False)
    model.eval()
    if args.dtype in ['fp32', 'bf16']:
        if 'llama' in model_type:
            traced_model = torch.jit.trace(model, (input_ids, attention_mask, past_key_value_torch))
            print("Traced model is saved as {}".format(args.pt_file))
        else:
            traced_model = torch.jit.trace(model, (input_ids, past_key_value_torch, attention_mask))
            print("Traced model is saved as {}".format(args.pt_file))
    else:
        print("Model with {} can't be traced, please provide one.".format(args.dtype))
        sys.exit(1)

from intel_extension_for_transformers.backends.neural_engine.compile import compile, autocast
if 'llama' not in model_type:
    if args.dtype == "bf16":
        with autocast("bf16"):
            graph = compile(traced_model)
    elif args.dtype == "int8":
        graph = compile(traced_model, './int8_pattern.conf')
    else:
        graph = compile(traced_model)
else:
    if args.dtype == "bf16":
        with autocast("bf16"):
            graph = compile(traced_model, './llama_pattern.conf')
    elif args.dtype == "int8":
        graph = compile(traced_model, './llama_int8_pattern.conf')
    else:
        graph = compile(traced_model, './llama_pattern.conf')
        
graph.save(args.output_model)
print('Neural Engine ir is saved as {}'.format(args.output_model))
