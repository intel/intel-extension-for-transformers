import argparse
import json, tqdm
import torch
import copy

import sys
sys.path.insert(0, '/root/hengguo/intel-extension-for-transformers')

from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from intel_extension_for_transformers.transformers.modeling.kv_cache_compression.h2o_sim_drop.modify_llama import convert_kvcache_llama_heavy_recent
from intel_extension_for_transformers.transformers.modeling.kv_cache_compression.h2o_sim_drop.modify_opt import convert_kvcache_opt_heavy_recent
from intel_extension_for_transformers.transformers.modeling.kv_cache_compression.h2o_sim_drop.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent

from tasks import EvalHarnessAdaptor

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
    "opt": convert_kvcache_opt_heavy_recent,
    "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')
    parser.add_argument("--tasks", nargs='+', default=["lambada_openai",
                                                   "hellaswag", "winogrande", "piqa", "wikitext"],
                    type=str, help="tasks list for accuracy validation")
    parser.add_argument('--num_fewshot', type=int, default=0)

    parser.add_argument('--enable_small_cache', action='store_true')
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m')
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--seq_len", type=int, default=1024)

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    batch_size = 1
    pe = 'fixed'
    seq = args.seq_len

    # build data
    requests = []
    class DryRunner:
        def eval(self, batch):
            for text in batch['text']:
                item = {
                    "best_of": 1, 
                    "echo": True, 
                    "logprobs": 1, 
                    "max_tokens": 0, 
                    "model": "x", 
                    "n": 1, 
                    "prompt": text, 
                    "request_type": "language-model-inference", 
                    "stop": None, 
                    "temperature": 0, 
                    "top_p": 1
                }
                requests.append(item)
            out = {
                'mask_loss': [1.0] * len(batch),
                'each_correct': [True] * len(batch),
            }
            return out
    t = DryRunner()
    adaptor = EvalHarnessAdaptor(t, seq, batch_size, shrink=pe != "fixed")
    result = evaluator.evaluate(adaptor, tasks.get_task_dict(args.tasks), False, args.num_fewshot, None)

    model_name = args.model_name
    if 'cpu' in args.device:
        device = args.device
    else:
        device = f"cuda:{args.device}"

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)

    if args.enable_small_cache:
        print('Enable Small Cache Size')
        # checkpoint = copy.deepcopy(model.state_dict())
        # model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_type](model, config)
        from intel_extension_for_transformers.transformers.modeling.kv_cache_compression import convert_model
        model = convert_model(model, heavy_ratio=args.heavy_ratio, recent_ratio=args.recent_ratio, h2o_min_seqlen=0)
        # model.load_state_dict(checkpoint)

    model = model.to(device)
    print('using device: ', device)
    model.eval()
    # model.half().eval()

    results = []
    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            logits = model(input_ids).logits.log_softmax(dim=-1)

            values, indices = logits.squeeze(0).topk(dim=-1, k=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
            
            gold_indices = input_ids[:, 1:] # skip first
            logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().tolist()
            top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1), indices.squeeze(-1))]
            
            result['result'] = {
                "choices": [
                    {
                        "text": prompt, 
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

    # evaluate
    class RealRunner:
        def __init__(self, args):
            self.results = {}
            for item in results:
                request = item['request']
                result = item['result']
                self.results[json.dumps(request)] = result
            print(f"{len(self.results)} items in the cache")
        
        def eval(self, batch):
            from tasks.eval_harness import tokenizer
            mask_loss = []
            each_correct = []
            for i, text in enumerate(batch['text']):
                request = {
                        "best_of": 1, 
                        "echo": True, 
                        "logprobs": 1, 
                        "max_tokens": 0, 
                        "model": "x", 
                        "n": 1, 
                        "prompt": text, 
                        "request_type": "language-model-inference", 
                        "stop": None, 
                        "temperature": 0, 
                        "top_p": 1
                    }
                
                key = json.dumps(request)
                correct = True
                
                if key in self.results:
                    result = self.results[key]
                    token_logprobs = result['choices'][0]['logprobs']['token_logprobs']
                    tokens = result['choices'][0]['logprobs']['tokens']
                    top_logprobs = result['choices'][0]['logprobs']['top_logprobs']
                    assert token_logprobs[0] is None
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    obs = batch['obs'][i]
                    target = batch['target'][i]
                    eval_mask = batch['eval_mask'][i]
                    
                    n_positive = 0
                    sum_lobprob = 0
                    if args.debug:
                        print(target)
                    for i, mask in enumerate(eval_mask):
                        try:
                            if i+1 >= len(tokens):
                                break
                            if mask == True:
                                if args.debug:
                                    print(tokens[i+1], next(iter(top_logprobs[i+1].keys())))
                                correct = correct and (tokens[i+1] == next(iter(top_logprobs[i+1].keys())))
                                sum_lobprob += token_logprobs[i+1]
                                n_positive += 1
                        except Exception as e:
                            raise e
                    # avg_logprob = sum(token_logprobs[1:]) / (len(token_logprobs) - 1)
                    avg_logprob = sum_lobprob / n_positive
                    mask_loss.append( - avg_logprob)
                    each_correct.append( correct )
                else:
                    assert False

            out = {
                'mask_loss': mask_loss,
                'each_correct': each_correct,
            }
            return out

    t = RealRunner(args)

    adaptor = EvalHarnessAdaptor(t, seq, batch_size, shrink=pe != "fixed")
    results = evaluator.evaluate(adaptor, tasks.get_task_dict(args.tasks), False, args.num_fewshot, None)
    
    dumped = json.dumps(results, indent=2)
    print(dumped)
