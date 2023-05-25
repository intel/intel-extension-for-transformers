"""Inference for LLM models."""
import abc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig

from conversation import conv_templates, SeparatorStyle
from compression import compress_module

from torch import nn
from torch.nn import functional as F
import time

class IpexWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        #print("ipex-bf16")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True)

        return out


def load_model(model_name, device, num_gpus, load_8bit=False, ipex=False, debug=False):
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if ipex:
        print('=='*10, "ipex")
        import intel_extension_for_pytorch as intel_ipex
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, return_dict=True, torch_dtype=torch.bfloat16,
            trust_remote_code=True, max_seq_len=8192)
        model = intel_ipex.optimize(model.eval(), dtype=torch.bfloat16, inplace=True)
        """
        example_inputs = []
        input_ids = torch.tensor([[0]*32])
        attention_mask = torch.ones(1, input_ids.shape[1]+1)
        # only for llama
        # shape = [1, config.n_heads, 1, config.d_model // config.n_heads]
        shape = [1, 1, config.d_model]
        layers = config.n_layers

        print(shape)

        past_key_value_torch = [(torch.zeros(shape), torch.zeros(shape)) for i in range(layers)]
        example_inputs.append(input_ids)
        example_inputs.append(past_key_value_torch)
        example_inputs.append(attention_mask)
        self_jit = torch.jit.trace(model, tuple(example_inputs), strict=False)
        self_jit = torch.jit.freeze(self_jit.eval())
        setattr(model, "trace_graph", self_jit)
        """
        model = IpexWrapper(model)
        setattr(model, "config", model.model.config)

    else:
        print('=='*10, "normal fp32")
        model = AutoModelForCausalLM.from_pretrained(model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                max_seq_len=8192)


    # print(tokenizer.eos_token_id)
    # print(tokenizer.decode(tokenizer.eos_token_id))
    # import os
    # os._exit(0)
    """
    special_tokens_dict = {}
    special_tokens_dict["eos_token"] = "</s>"
    special_tokens_dict["pad_token"] = "</s>"
    tokenizer.eos_token = "</s>"
    """

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


@torch.inference_mode()
def generate_stream(model, tokenizer, params, device,
                    context_len=2048, stream_interval=2):
    #print("Generate Streams")
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            st=time.time()
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            # logits = out.logits
            # past_key_values = out.past_key_values
            print("first token latency:", time.time()-st)
        else:
            st_n=time.time()
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            # logits = out.logits
            # past_key_values = out.past_key_values

            print("Token:",i,"latency:",time.time()-st_n)

        # logits = out[0]
        # past_key_values = out[1]
        logits = out.logits
        past_key_values = out.past_key_values
        print(past_key_values[0][0].shape)

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            pos = output.rfind(stop_str, l_prompt)
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

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
