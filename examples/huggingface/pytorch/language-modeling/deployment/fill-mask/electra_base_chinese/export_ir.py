from transformers import AutoTokenizer, ElectraForMaskedLM, ElectraModel, ElectraForPreTraining
from intel_extension_for_transformers.backends.neural_engine.compile import compile, autocast
import torch
import argparse
import os
import sys

parser = argparse.ArgumentParser('Electra-Base Generation ir', add_help=False)
parser.add_argument("--model_name",
        type=str,
        help="path to pytorch model file",
        default="hfl/chinese-legal-electra-base-generator",
    )
parser.add_argument('--dtype', default="fp32", type=str)
parser.add_argument('--output_model', default="./ir", type=str)
parser.add_argument('--pt_file', default="./model.pt", type=str)
args = parser.parse_args()
print(args)

model_id = args.model_name
is_generator = True if 'generator' in model_id else False

if os.path.exists(args.pt_file):
    print('PT model exists, compile will be executed.')
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if is_generator:
        pt_model = ElectraForMaskedLM.from_pretrained(model_id, torchscript=True)
    else:
        pt_model = ElectraForPreTraining.from_pretrained(model_id, torchscript=True)
    pt_model.eval()
    text = "我喜欢写[MASK]码"
    inputs = tokenizer(text, return_tensors="pt")
    if args.dtype in ['fp32', 'bf16']:
        jit_model = torch.jit.trace(pt_model, (inputs.input_ids, 
                                    inputs.attention_mask, inputs.token_type_ids))
        torch.jit.save(jit_model, args.pt_file)
        print("Traced model is saved as {}".format(args.pt_file))
    else:
        print("Model with {} can't be traced, please provide one.".format(args.dtype))
        sys.exit(1)

if args.dtype == "bf16":
    with autocast("bf16"):
        graph = compile(args.pt_file, './bf16_pattern.conf')
elif args.dtype == "fp32":
    graph = compile(args.pt_file)
else:
    print("Only supports fp32 and bf16 dtype.")
    sys.exit(1)
        
graph.save(args.output_model)
print('Neural Engine ir is saved as {}'.format(args.output_model))
