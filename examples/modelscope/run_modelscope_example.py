from transformers import TextStreamer
from modelscope import AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from typing import List, Optional
import argparse

def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name: String", required=True, default="qwen/Qwen-7B")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to start generation with: String (default: empty)",
        default="你好，你可以做点什么？",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--use_neural_speed", action="store_true")
    args = parser.parse_args(args_in)
    print(args)
    model_name = args.model     # Modelscope model_id or local model
    prompt = args.prompt
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, model_hub="modelscope")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    streamer = TextStreamer(tokenizer)
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
    
if __name__ == "__main__":
    main()
