import argparse
import json
import os.path
import sys
sys.path.insert(0, '/home/hengguo/code/intel-extension-for-transformers')

import tqdm
import torch

from rouge import Rouge
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from intel_extension_for_transformers.transformers.modeling.kv_cache_compression.h2o import convert_model



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument('--h2o', action='store_true')
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument("--h2o_min_seqlen", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument('--mean', action='store_true')


    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    try:
        device_str = int(args.device)
        device_str = f'cuda:{args.device}'
    except:
        device_str = args.device
    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)

    if args.batch_size>1:
        tokenizer.pad_token = tokenizer.eos_token

    if args.h2o:
        print('Enabling H2O KV cache')
        model = convert_model(
            model,
            heavy_ratio=args.heavy_ratio,
            recent_ratio=args.recent_ratio,
            h2o_min_seqlen=args.h2o_min_seqlen,
            real_drop=True,
            is_gen=True,
            mean=args.mean)
        model.clean_cache()

    model = model.half().eval().to(device_str)

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples from {} samples'.format(args.sample_num, len(requests)))
    requests = requests[:args.sample_num]

    results = []
    rouge = Rouge()
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []

    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['article']
            label = request['summary_gt']
            temperature = request['temperature']
            stop = request['stop']

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=request['max_tokens'] + len(input_ids[0]),
                temperature=temperature,
                top_k=args.k,
                top_p=request['top_p'],
                do_sample=True,
                num_return_sequences=request['n'],
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )

            if args.h2o:
                model.clean_cache()

            tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
            logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
            top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

            generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
            generate_text = generate_text[: generate_text.find(stop[0])]

            scores = rouge.get_scores(generate_text, label)[0]
            rouge1_score_list.append(scores['rouge-1']['f'])
            rouge2_score_list.append(scores['rouge-2']['f'])
            rougel_score_list.append(scores['rouge-l']['f'])

            result['result'] = {
                "choices": [
                    {
                        "text": generate_text,
                        "logprobs": {
                            "tokens": tokens, 
                            "token_logprobs": logprobs, 
                            "top_logprobs": top_logprobs, 
                            "text_offset": []
                        }, 
                        "finish_reason": "length"
                    }
                ], 
                "request_time": {
                    "batch_time": 0, 
                    "batch_size": 1}
            }
            
            results.append(result)
            print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}, prompt length: {}, generate text length: {}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list), input_ids.size(-1), output_sequences['sequences'].size(-1)))

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    main()