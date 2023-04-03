import argparse
import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

PROMPT_DICT = {
    "prompt_with_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_without_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bm", "--base_model_path", type=str, default="")
    parser.add_argument("-lm", "--lora_model_path", type=str, default="")
    parser.add_argument(
        "-ins",
        "--instructions",
        type=str,
        nargs="+",
        default=["Tell me about alpacas.", "Tell me five words that rhyme with 'shock'."]
    )
    args = parser.parse_args()
    return args

def create_prompts(examples):
    prompts = []
    for example in examples:
        prompt_template = PROMPT_DICT["prompt_with_input"] \
            if example["input"] != "" else PROMPT_DICT["prompt_without_input"]
        prompt = prompt_template.format_map(example)
        prompts.append(prompt)
    return prompts

def main():
    args = parse_args()
    base_model_path = args.base_model_path
    lora_model_path = args.lora_model_path
    prompts = create_prompts(
        [{'instruction':instruction, 'input':''} for instruction in args.instructions]
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if "llama" in base_model_path:
        model = AutoModelForCausalLM.from_pretrained(base_model_path)
    elif "flan-t5" in base_model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
    else:
        raise ValueError(f"Unsupported model {base_model_path}, only supports LLaMA and FLAN-T5 now.")

    if lora_model_path:
        model = PeftModel.from_pretrained(model, lora_model_path)

    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    def evaluate(
        prompt,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        input = tokenizer(prompt, return_tensors="pt")
        input_ids = input["input_ids"].to(model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        sequence = generation_output.sequences[0]
        output = tokenizer.decode(sequence)
        if "### Response:" in output:
            return output.split("### Response:")[1].strip()
        elif "<pad> " in output:
            return output.split("<pad> ")[1].strip()
        else:
            return output

    for idx, tp in enumerate(zip(prompts, args.instructions)):
        prompt, instruction = tp
        idxs = f"{idx+1}"
        print("="*30 + idxs + "="*30 + "\n")
        print("Instruction:", instruction)
        print("Response:", evaluate(prompt))
        print("="*(60 + len(idxs)) + "\n")

if __name__ == "__main__":
    main()