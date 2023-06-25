import argparse
import torch
from peft import PeftModel
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import re

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
    parser.add_argument("-pm", "--peft_model_path", type=str, default="")
    parser.add_argument(
        "-ins",
        "--instructions",
        type=str,
        nargs="+",
        default=[
            "Tell me about alpacas.",
            "Tell me five words that rhyme with 'shock'.",
        ],
    )
    # Add arguments for temperature, top_p, top_k and repetition_penalty
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="The value used to control the randomness of sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.75,
        help="The cumulative probability of tokens to keep for sampling.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="The number of highest probability tokens to keep for sampling.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="The penalty applied to repeated tokens.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default=None, help="specify tokenizer name"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="enable when use custom model architecture that is not yet part of the Hugging Face transformers package like MPT",
    )
    args = parser.parse_args()
    return args


def create_prompts(examples):
    prompts = []
    for example in examples:
        prompt_template = (
            PROMPT_DICT["prompt_with_input"]
            if example["input"] != ""
            else PROMPT_DICT["prompt_without_input"]
        )
        prompt = prompt_template.format_map(example)
        prompts.append(prompt)
    return prompts


def main():
    args = parse_args()
    base_model_path = args.base_model_path
    peft_model_path = args.peft_model_path
    prompts = create_prompts(
        [{"instruction": instruction, "input": ""} for instruction in args.instructions]
    )

    # Check the validity of the arguments
    if not 0 < args.temperature <= 1.0:
        raise ValueError("Temperature must be between 0 and 1.")
    if not 0 <= args.top_p <= 1.0:
        raise ValueError("Top-p must be between 0 and 1.")
    if not 0 <= args.top_k <= 200:
        raise ValueError("Top-k must be between 0 and 200.")
    if not 1.0 <= args.repetition_penalty <= 2.0:
        raise ValueError("Repetition penalty must be between 1 and 2.")

    tokenizer_path = (
        args.tokenizer_name if args.tokenizer_name is not None else base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=not args.use_slow_tokenizer
    )

    if re.search("flan-t5", base_model_path, re.IGNORECASE):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_path, trust_remote_code=True if args.trust_remote_code else None
        )
    elif re.search("llama", base_model_path, re.IGNORECASE) or re.search(
        "mpt", base_model_path, re.IGNORECASE
    ):
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True if args.trust_remote_code else None
        )
    else:
        raise ValueError(
            f"Unsupported model {base_model_path}, only supports FLAN-T5/LLAMA/MPT now."
        )

    if re.search("llama", model.config.architectures[0], re.IGNORECASE):
        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2

    if (
        hasattr(model.generation_config, "pad_token_id")
        and model.generation_config.pad_token_id is not None
    ):
        tokenizer.pad_token_id = model.generation_config.pad_token_id
    if (
        hasattr(model.generation_config, "eos_token_id")
        and model.generation_config.eos_token_id is not None
    ):
        tokenizer.eos_token_id = model.generation_config.eos_token_id
    if (
        hasattr(model.generation_config, "bos_token_id")
        and model.generation_config.bos_token_id is not None
    ):
        tokenizer.bos_token_id = model.generation_config.bos_token_id

    if tokenizer.pad_token_id is None:
        model.generation_config.pad_token_id = (
            tokenizer.pad_token_id
        ) = tokenizer.eos_token_id

    if peft_model_path:
        model = PeftModel.from_pretrained(model, peft_model_path)

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    model.eval()

    def evaluate(
        prompt,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
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
            repetition_penalty=repetition_penalty,
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
        output = tokenizer.decode(sequence, skip_special_tokens=True)
        if "### Response:" in output:
            return output.split("### Response:")[1].strip()
        elif "<pad> " in output:
            return output.split("<pad> ")[1].strip()
        else:
            return output

    for idx, tp in enumerate(zip(prompts, args.instructions)):
        prompt, instruction = tp
        idxs = f"{idx+1}"
        print("=" * 30 + idxs + "=" * 30 + "\n")
        print("Instruction:", instruction)
        print(
            "Response:",
            evaluate(
                prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=2,
            ),
        )
        print("=" * (60 + len(idxs)) + "\n")


if __name__ == "__main__":
    main()
