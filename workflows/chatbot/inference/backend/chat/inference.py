"""Inference for LLM models."""
import abc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig

from conversation import conv_templates, SeparatorStyle
from compression import compress_module

import time
import psutil

from torch import nn
from torch.nn import functional as F
import intel_extension_for_transformers.backends.neural_engine.compile as itrex_compile
import numpy as np
import random

class ItrexWrapper(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        print("itrex")
        input_ids_1 = input_ids.cpu().numpy().astype(np.int32)
        if attention_mask is None and past_key_values is None:
            attention_mask = torch.ones([input_ids.shape[0], input_ids.shape[1]])
            # attention_mask_1 = torch.cat([torch.zeros([1, 1]), attention_mask], dim=-1)
            # attention_mask_1 = attention_mask_1.cpu().numpy().astype(np.int32)
            # only for llama-7b
            # past_key_values = tuple([(torch.zeros([1,32,1,128]), torch.zeros([1,32,1,128])) for i in range(32)])
            # past_k_v = [past_key_values[i][j].cpu().numpy() for i in range(32) for j in range(2)]
            attention_mask_1 = attention_mask.cpu().numpy().astype(np.int32)
            past_key_values = tuple([(torch.zeros([1,0,16,256]), torch.zeros([1,0,16,256])) for i in range(28)])
            past_k_v = [past_key_values[i][j].cpu().numpy() for i in range(28) for j in range(2)]
        else:
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-3] + 1)
            # attention_mask_1 = torch.cat([torch.zeros([1, 1]), attention_mask[:, 1:]], dim=-1)
            # attention_mask_1 = attention_mask_1.cpu().numpy().astype(np.int32)
            attention_mask_1 = attention_mask.cpu().numpy().astype(np.int32)
            input_ids_1 = input_ids.cpu().numpy().astype(np.int32)

            # numpy
            past_k_v = past_key_values
        print(input_ids_1.shape)
        print(attention_mask_1.shape)
        print(past_k_v[0].shape)

        # predictions = self.graph.inference([input_ids_1, attention_mask_1] + past_k_v)
        predictions = self.graph.inference([input_ids_1] + past_k_v + [attention_mask_1])
        outs = []
        for key in predictions:
            outs.append(predictions[key])

        logits = outs[0]
        logits = torch.from_numpy(logits)

        # only for gpt-j-6b
        logits = torch.unsqueeze(logits, dim=1)

        past_k_v = outs[1:]

        return logits, past_k_v

class IpexWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        # print("ipex-bf16")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            out = self.model(
                input_ids, 
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True)

        return out


def load_model(model_name, device, num_gpus, load_8bit=False, itrex=False, ipex=False, debug=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
    else:
        raise ValueError(f"Invalid device: {device}")

    if 'mpt' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        config = AutoConfig.from_pretrained(model_name)

    if ipex:
        print('=='*10, "ipex")
        import intel_extension_for_pytorch as intel_ipex
        if "mpt" in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    low_cpu_mem_usage=True,
                                                    return_dict=True,
                                                    torch_dtype=torch.bfloat16,
                                                    trust_remote_code=True,
                                                    max_seq_len=8192)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                         low_cpu_mem_usage=True,
                                                         return_dict=False)
        model = intel_ipex.optimize(model.eval(), dtype=torch.bfloat16, inplace=True)
        model = IpexWrapper(model)
        setattr(model, "config", model.model.config)
    elif itrex:
        print('=='*10, "itrex")
        graph = itrex_compile.compile(model_name)
        model = ItrexWrapper(graph)
        setattr(model, "config", config)
    else:
        print('=='*10, "normal fp32")
        if "mpt" in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                        low_cpu_mem_usage=True,
                                                        torch_dtype=torch.bfloat16,
                                                        trust_remote_code=True,
                                                        max_seq_len=8192)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                        low_cpu_mem_usage=True)

    if not "mpt" in model_name:
        special_tokens_dict = {}
        special_tokens_dict["eos_token"] = "</s>"
        special_tokens_dict["pad_token"] = "</s>"
        # </s>
        tokenizer.eos_token = "</s>"

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

@torch.inference_mode()
def generate_stream(model, tokenizer, params, device,
                    context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    stop_str = "</s>"
    topk = int(params.get("topk", 1))
    print('topk: ', topk)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    memory_allocated = round(psutil.Process().memory_info().rss / 1024**3, 3)
    print("memory used total:", memory_allocated, "GB")

    for i in range(max_new_tokens):
        if i == 0:
            st=time.time()
            out = model(
                torch.as_tensor([input_ids], device=device))
            # logits = out.logits
            # past_key_values = out.past_key_values
            logits = out[0]
            past_key_values = out[1]
            print("first token latency:", time.time()-st)
        else:
            st_n=time.time()
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
            # for itrex gpt-j-6b bf16 [batch_size, seq_len, num_heads, head_size]
            # attention_mask = torch.ones(1, past_key_values[0][0].shape[-3] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            # logits = out.logits
            # past_key_values = out.past_key_values
            logits = out[0]
            past_key_values = out[1]
            print("Token:",i,"latency:",time.time()-st_n)

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if topk == 1:
            token = int(torch.argmax(last_token_logits))
        else:
            # print("topk > 1")
            # probs = torch.softmax(last_token_logits / temperature, dim=-1)
            # token = int(torch.multinomial(probs, num_samples=1))


            logits = logits[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(logits, top_k=topk, top_p=0.9)
            probabilities = F.softmax(filtered_logits, dim=-1)
            # print(torch.nonzero(probabilities))

            # torch.manual_seed(100)
            token = int(torch.multinomial(probabilities, 1))
            
        output_ids.append(token)


        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            if i == 0 and tokenizer.decode([token]) in [".", ",", "?"]:
                print("=="*20)
                print(output)
                output = output[:-1]
                output_ids[-1] = tokenizer.convert_tokens_to_ids("")

            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            pos = output.rfind("Human", l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break

    del past_key_values


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream, skip_echo_len: int):
        """Stream output."""


def chat_loop(model_name: str, device: str, num_gpus: str, load_8bit: bool,
              conv_template: str, temperature: float, max_new_tokens: int,
              chatio: ChatIO, debug: bool):
    # Model
    model, tokenizer = load_model(model_name, device,
        num_gpus, load_8bit, debug)

    # Chat
    conv = conv_templates[conv_template].copy()
    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        generate_stream_func = generate_stream
        prompt = conv.get_prompt()
        skip_echo_len = len(prompt) + 1

        params = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, params, device)
        outputs = chatio.stream_output(output_stream, skip_echo_len)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()

        # if debug:
        if True:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
