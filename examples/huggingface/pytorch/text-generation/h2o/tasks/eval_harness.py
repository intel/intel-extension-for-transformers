from functools import partial

import os
import transformers
from lm_eval.base import LM
from tqdm import tqdm
import numpy as np

from .util import sample_batch, shrink_seq
import multiprocessing
import ftfy

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

tokenizer = None

def process_init():
    global tokenizer
    model_name = os.environ.get('MODEL_NAME', 'facebook/opt-1.3b')

    if model_name == "EleutherAI/gpt-neox-20b":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e30)
        tokenizer.pad_token = "<|endoftext|>"
    elif model_name == 'huggyllama/llama-7b':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e30)
        tokenizer.pad_token = "<|endoftext|>"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_bos_token = False

def process_request(x, seq):
    global tokenizer

    ctx, cont = x
#     ctx_tokens = tokenizer.encode("<|endoftext|>" + ftfy.fix_text(ctx, normalization="NFKC"))
    ctx_text = ftfy.fix_text(ctx, normalization="NFKC")
    cont_text = ftfy.fix_text(cont, normalization="NFKC")
    all_text = ctx_text + cont_text

    ctx_tokens = tokenizer(ctx_text, add_special_tokens=False)['input_ids']
    cont_tokens = tokenizer(cont_text, add_special_tokens=False)['input_ids']

    all_tokens = ctx_tokens + cont_tokens
    all_tokens = np.array(all_tokens)[-seq:]  # truncate sequence at seq length

    provided_ctx = len(all_tokens) - 1
    pad_amount = seq - provided_ctx

    return {
        "obs": np.pad(all_tokens[:-1], ((0, pad_amount),), constant_values=tokenizer.pad_token_id),
        "target": np.pad(all_tokens[1:], ((0, pad_amount),), constant_values=tokenizer.pad_token_id),
        "ctx_length": seq,
        "eval_mask": np.logical_and(
            np.arange(0, seq) > len(all_tokens) - len(cont_tokens) - 2,
            np.arange(0, seq) < len(all_tokens) - 1
        ),
        "prompt": ctx_text,
        "target": cont_text,
        "text": all_text,
    }


class EvalHarnessAdaptor(LM):
    def greedy_until(self, requests):
        raise Exception("unimplemented")

    def loglikelihood_rolling(self, requests):
        raise Exception("unimplemented")

    def __init__(self, tpu_cluster, seq, batch, shrink, min_seq=None):
        super().__init__()
        self.tpu = tpu_cluster
        self.seq = seq
        self.batch = batch
        self.shrink = shrink
        self.min_seq = min_seq

        self.pool = multiprocessing.Pool(processes=1, initializer=process_init)
        # self.pool = multiprocessing.Pool(initializer=process_init)
        process_init()

    def convert_requests(self, requests):
        return self.pool.imap(partial(process_request, seq=self.seq), requests)

    def loglikelihood(self, requests):
        output = []

        r = self.convert_requests(requests)
        self.pool.close()
        zero_example = process_request(requests[0], self.seq)

        for b in tqdm(sample_batch(r, self.batch, zero_example),
                      desc="LM eval harness",
                      total=len(requests) // self.batch):

            if self.shrink:
                b = shrink_seq(b, min_seq=self.min_seq)

            out = self.tpu.eval(b)

            for loss, correct in zip(out["mask_loss"], out["each_correct"]):
                output.append((float(-loss), bool(correct)))

        return output


