# Import necessary libraries
import os
import time
import argparse
from tqdm import tqdm
from pprint import pprint
from eagle.modeling_eagle import EAGLE
from transformers import AutoModelForCausalLM,LlamaModel,MistralModel,AutoTokenizer,MixtralForCausalLM,LlamaConfig,LlamaForCausalLM
import torch
from fastchat.model import get_conversation_template

# Define a dictionary to map string representations of data types to their corresponding PyTorch data types
DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# Function to parse command-line arguments
def parse_argument():
    # Construct the argument parser and add the arguments
    arg_parser = argparse.ArgumentParser(description="Test EAGLE with Llama-2")
    arg_parser.add_argument("-d", '--device', type=str, default="cpu", choices=["cpu", "xpu", "cuda"],
                            help="Target device for text generation")
    arg_parser.add_argument("-t", '--dtype', type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                            help="Data type for text generation")
    arg_parser.add_argument('--max_new_tokens', type=int, default=512,
                            help="Number of max new tokens for text generation.")
    arg_parser.add_argument('--use_eagle', action='store_true', help='Use EAGLE model for generation')
    return arg_parser.parse_args()


if __name__ == "__main__":
    # Define paths to the base and eagle models
    # Note - You can use your choice of base model, we are using llama-2-7b-chat in this example
    base_model_path = "meta-llama/Llama-2-7b-chat-hf"
    ea_model_path = "yuhuili/EAGLE-llama2-chat-7B"

    # Parse command-line arguments
    args = parse_argument()

    # Extract arguments
    device = args.device
    dtype = DTYPE_MAP[args.dtype]
    max_new_tokens = args.max_new_tokens
    use_eagle = args.use_eagle

    if device == "xpu":
        import intel_extension_for_pytorch as ipex

    if device == "cpu":
        assert dtype == torch.float32, f"CPU can only support float32. Got dtype = {args.dtype}"
     
    tokenizer=AutoTokenizer.from_pretrained(base_model_path)

    # Define a message, conversation template and system message to be processed by the model
    your_message="Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
    conv = get_conversation_template("llama-2-chat")
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt1 = conv.get_prompt()+" "

    your_message="Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point."
    conv = get_conversation_template("llama-2-chat")
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt2 = conv.get_prompt()+" "

    your_message = "Write a brief technical documentation for a new software feature, explaining its functionality, benefits, and how to implement it."
    conv = get_conversation_template("llama-2-chat")
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt3 = conv.get_prompt()+" "

    your_message = "Craft a marketing campaign plan for a new product launch, including the target audience, key messages, and promotional strategies."
    conv = get_conversation_template("llama-2-chat")
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt4 = conv.get_prompt()+" "

    # Pick the prompt you want to use to run the example or write your own prompt, initialize it as a list
    text = [prompt1]

    # Load the model and set parameters
    model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                 torch_dtype=dtype,
                                                 ).eval().to(device)
    # model = torch.xpu.optimize(model)
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    eagle = EAGLE(model, ea_model_path)

    # Iterate over each prompt in the text list
    for prompt in text:
        # Initialize counters
        t_total = 0.
        total_new_tokens = 0
        for _ in tqdm(range(10)):
            t_b = time.time()
            
            # inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
            inputs = tokenizer(text, return_tensors="pt").to(device)
          
            # Generate text using the EAGLE model if the flag is set, otherwise use the base model
            if use_eagle:
                outs = eagle.generate(**inputs, max_new_tokens=max_new_tokens,temperature=0.5)
            else:
                outs=model.generate(**inputs, max_new_tokens=max_new_tokens,temperature=0.5)
            output=tokenizer.batch_decode(outs)
            t_e = time.time()

          
            # Update counters
            t_total += t_e - t_b
            total_new_tokens += len(outs) - inputs.input_ids.shape[-1]

            # Print total new tokens and generated text
            print("Total new tokens", total_new_tokens)
            pprint(output)
        print(f"TPS:  {total_new_tokens / t_total}")

    del model

