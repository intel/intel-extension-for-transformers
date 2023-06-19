from transformers import AutoTokenizer, GPTJModel, AutoModelForCausalLM
import torch
import argparse
import os
import sys
from optimum.utils import NormalizedConfigManager

class Net(torch.nn.Module):
    def __init__(self, ori_model):
        super(Net, self).__init__()
        self.model = ori_model
    def forward(self, input_ids, pastkv, mask):
        return self.model(input_ids=input_ids, attention_mask=mask, past_key_values=pastkv, return_dict=False)

parser = argparse.ArgumentParser('GPT-J Generation ir', add_help=False)
parser.add_argument("--model",
        type=str,
        help="path to original config and weight files",
        default="EleutherAI/gpt-j-6B",
    )
parser.add_argument('--dtype', default=None, type=str)
parser.add_argument('--output_model', default="./ir", type=str)
parser.add_argument('--pt_file', type=str)
args = parser.parse_args()
print(args)

model_id = args.model
model = AutoModelForCausalLM.from_pretrained(model_id, return_dict=False)
model.eval()

normalized_config = NormalizedConfigManager.get_normalized_config_class(model.config.model_type)(model.config)
num_layers = normalized_config.num_layers
num_attention_heads = normalized_config.num_attention_heads
hidden_size = normalized_config.hidden_size
d_k = hidden_size // num_attention_heads
model_type = model.config.model_type

if 'llama' in model_type:
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Once upon a time, there existed a little girl, who liked to have adventures." + \
        " She wanted to go to places and meet new people, and have fun."
init_input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
input_ids = init_input_ids.clone().unsqueeze(0)
attention_mask = torch.ones(len(input_ids)).unsqueeze(0)
past_key_value = tuple([(torch.zeros([1,num_attention_heads,0,d_k]),
                         torch.zeros([1,num_attention_heads,0,d_k])) for i in range(num_layers)])

if 'llama' in model_type:
    input_ids = init_input_ids.clone()
    attention_mask = torch.ones(len(input_ids)+1)
    attention_mask[0] = 0
    input_ids = input_ids[0:1].unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    past_key_value = tuple([(torch.zeros([1,32,34,128]), torch.zeros([1,32,34,128])) for i in range(32)])
    if 'llama_13b' in model_type:
        past_key_value = tuple([(torch.zeros([1,40,34,128]), torch.zeros([1,40,34,128])) for i in range(40)])

traced_model = None

if args.pt_file and os.path.exists(args.pt_file):
    print('PT model exists, compile will be executed.')
    del model
    traced_model = torch.jit.load(args.pt_file)
else:
    assert args.dtype in ['fp32', 'bf16'], "Model with {} can't be traced, please provide one.".format(args.dtype)
    if 'llama' in model_type:
        net = model
        traced_model = torch.jit.trace(net, (input_ids, attention_mask, past_key_value))
    else:
        net = Net(model)
        traced_model = torch.jit.trace(net, (input_ids, past_key_value, attention_mask))

from intel_extension_for_transformers.backends.neural_engine.compile import compile, autocast
if 'llama'  in model_type:
    if args.dtype == "bf16":
        with autocast("bf16"):
            graph = compile(traced_model, './llama_pattern.conf')
    elif args.dtype == "int8":
        graph = compile(traced_model, './llama_int8_pattern.conf')
    else:
        graph = compile(traced_model, './llama_pattern.conf')
elif 'gpt_neox' in model_type:
    if args.dtype == "bf16":
        with autocast("bf16"):
            graph = compile(traced_model, './gpt_neox_pattern.conf')
    else:
        graph = compile(traced_model, './gpt_neox_pattern.conf')
else:
    if args.dtype == "bf16":
        with autocast("bf16"):
            graph = compile(traced_model)
    elif args.dtype == "int8":
        graph = compile(traced_model, './int8_pattern.conf')
    else:
        graph = compile(traced_model)
        
graph.save(args.output_model)
print('Neural Engine ir is saved as {}'.format(args.output_model))
