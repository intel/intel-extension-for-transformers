import argparse
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6b"
)
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument('--precision', default='bf16', type=str, help="fp32 or bf16")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--ipex", action="store_true", help="use intel extension for pytorch.")
parser.add_argument("--jit", action="store_true", help="convert model to torchscript mode.")
parser.add_argument("--tasks", nargs='+', default=["winogrande", "copa", "piqa", "rte", "hellaswag", \
                    "openbookqa", "lambada_openai", "lambada_standard", "wikitext"], type=str, \
                    help="tasks list for accuracy validation")

args = parser.parse_args()

if args.ipex:
    try:
        import intel_extension_for_pytorch as ipex
    except:
        assert False,"Please install intel_extension_for_pytorch, `pip install intel_extension_for_pytorch`"

if re.search("llama", args.model):
    from transformers import LlamaForCausalLM, LlamaTokenizer
    user_model = LlamaForCausalLM.from_pretrained(
        args.model,
        torchscript=True if args.ipex else False,  # torchscript will force `return_dict=False` to avoid jit errors
    )
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
else:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torchscript=True if args.ipex else False,  # torchscript will force `return_dict=False` to avoid jit errors
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)


# To channels last
user_model = user_model.to(memory_format=torch.channels_last)
user_model.eval()

# Dummy input ids    
input_ids = user_model.dummy_inputs["input_ids"]

amp_enabled = False
amp_dtype = None

# Bfloat16 model
if args.precision == "bf16":
    amp_enabled = True
    amp_dtype = torch.bfloat16
    if args.ipex:
        user_model = ipex.optimize(user_model, dtype=torch.bfloat16)
        if args.jit:
            with torch.no_grad(), torch.cpu.amp.autocast():
                user_model = torch.jit.trace(user_model, input_ids, strict=False)
                user_model = torch.jit.freeze(user_model)
    user_model.eval()

# Inference
input_ids, label = input_ids[:,0:-1], input_ids[:, -1]
outputs = user_model(input_ids)
if isinstance(outputs, tuple):
    last_token_logits = outputs[0]
else: 
    last_token_logits = outputs["logits"]
last_token_logits = last_token_logits[:, -1, :]
pred = last_token_logits.argmax(dim=-1)


# Accuracy
if args.accuracy:
    user_model.eval()
    def eval_func(user_model):
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

    with torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
        eval_func(user_model)

