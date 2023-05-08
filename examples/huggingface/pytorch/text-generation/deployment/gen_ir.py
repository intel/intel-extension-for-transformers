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
parser.add_argument('--pt_file', default="temp.pt", type=str)
args = parser.parse_args()
print(args)

model_id = args.model
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

clean_model = False
if 'llama' in model_id:
    past_key_value_torch = tuple([(torch.zeros([1,32,32,256]), torch.zeros([1,32,32,256])) for i in range(32)])
if os.path.exists(args.pt_file):
    print('PT model exists, compile will be executed.')
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, return_dict=False)
    model.eval()
    outputs = model(input_ids, past_key_value_torch, attention_mask)
    if args.dtype in ['fp32', 'bf16']:
        traced_model = torch.jit.trace(model, (input_ids, past_key_value_torch, attention_mask))
        torch.jit.save(traced_model, args.pt_file)
        print("Traced model is saved as {}".format(args.pt_file))
        clean_model = True
    else:
        print("Model with {} can't be traced, please provide one.".format(args.dtype))
        sys.exit(1)

from intel_extension_for_transformers.backends.neural_engine.compile import compile, autocast
if 'llama' not in model_id:
    if args.dtype == "bf16":
        with autocast("bf16"):
            graph = compile(args.pt_file)
    elif args.dtype == "int8":
        graph = compile(args.pt_file, './int8_pattern.conf')
    else:
        graph = compile(args.pt_file)
else:
    if args.dtype == "bf16":
        with autocast("bf16"):
            graph = compile(args.pt_file, './llama_pattern.conf')
    elif args.dtype == "int8":
        graph = compile(args.pt_file, './llama_int8_pattern.conf')
    else:
        graph = compile(args.pt_file, './llama_pattern.conf')
        
graph.save(args.output_model)
print('Neural Engine ir is saved as {}'.format(args.output_model))
if clean_model:
    os.remove(args.pt_file)
