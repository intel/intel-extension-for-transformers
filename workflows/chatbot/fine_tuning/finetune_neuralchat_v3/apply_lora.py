import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def apply_lora(base_model_path, lora_path):
    print(f"Loading the base model from {base_model_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    return base, model, base_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--lora-model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    args = parser.parse_args()

    base, target, base_tokenizer = apply_lora(args.base_model_path, args.lora_model_path)
    target.save_pretrained(args.output_path)
    base_tokenizer.save_pretrained(args.output_path)
