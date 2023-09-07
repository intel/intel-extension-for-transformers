import argparse
import copy, time
from datetime import datetime
import torch
from queue import Queue
import re, os, logging
from threading import Thread
import contextlib
from typing import List
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    TextIteratorStreamer,
    StoppingCriteriaList,
    StoppingCriteria,
)

from checkpoint_utils import (
    get_ds_injection_policy,
    get_repo_root,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)
from .checkpoint_utils import (
    get_ds_injection_policy,
    get_repo_root,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)

# Set necessary env variables
os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")
from transformers.deepspeed import is_deepspeed_available

if is_deepspeed_available():
    import deepspeed

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

instruction_template = {
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

chat_template = """### System:
- You are a helpful assistant chatbot trained by Intel.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.{eos_token}
### User:\n{instruction}{eos_token}
### Assistant:
"""

summarization_template = "{instruction}\nSummarize the highlights of this article.\n"


template_maps = {
    "completion": instruction_template,
    "chat": chat_template,
    "summarization": summarization_template,
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
        "--max_new_tokens",
        type=int,
        default=128,
        help="The maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--hf_access_token",
        type=str,
        default=None,
        help="Huggingface token to access model",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=0,
        help="The number of beams for beam search.",
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

    # habana parameters
    parser.add_argument(
        "--habana",
        action="store_true",
        help="Whether run on habana",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--jit",
        action="store_true",
        help="Whether to use jit trace. It should speed up generation.",
    )
    parser.add_argument(
        "--seed",
        default=27,
        type=int,
        help="Seed to use for random generation. Useful to reproduce your runs with `--do_sample`.",
    )
    parser.add_argument(
        "--bad_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that are not allowed to be generated.",
    )
    parser.add_argument(
        "--force_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that must be generated.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--local_rank", type=int, default=-1, metavar="N", help="Local process rank."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        choices=["completion", "chat", "summarization"],
        help="task name, different task means different templates.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
        help="bfloat16, float32 or float16",
    )
    parser.add_argument(
        "--return_stats", action='store_true', default=False,)
    args = parser.parse_args()
    return args


class StopOnTokens(StoppingCriteria):
    def __init__(self, min_length: int, start_length: int, stop_token_id: List[int]):
        self.min_length = min_length
        self.start_length = start_length
        self.stop_token_id = stop_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if scores is not None:
            if len(scores) > self.min_length:
                for stop_id in self.stop_token_id:
                    if input_ids[0][self.start_length - 1 + len(scores)] == stop_id:
                        return True
        elif input_ids.shape[-1] - self.start_length > self.min_length:
            for stop_id in self.stop_token_id:
                if input_ids[0][input_ids.shape[-1] - 1] == stop_id:
                    return True
        return False


def max_input_len(input_text_length):
    if input_text_length <= 128:
        return 128
    elif input_text_length <= 512:
        return 512
    elif input_text_length <= 2048:
        return 2048
    else:
        logger.warning("Max support length is 4096")
        return 4096


def add_template(example, template_name):
    if "prompt_with_input" in template_name:
        prompt_template = (
            template_name["prompt_with_input"]
            if example["input"] != ""
            else template_name["prompt_without_input"]
        )
    else:
        prompt_template = template_name
    prompt = prompt_template.format_map(example)
    return prompt


MODELS = {}


def smart_context_manager(use_deepspeed=False, model_dtype=torch.bfloat16):
    if use_deepspeed:
        ctx_manager = deepspeed.OnDevice(dtype=model_dtype, device="cpu")
    else:
        ctx_manager = contextlib.nullcontext()
    return ctx_manager


def import_deepspeed():
    if not is_deepspeed_available():
        raise ImportError(
            "This script requires deepspeed: `pip install"
            " git+https://github.com/HabanaAI/DeepSpeed.git@1.11.0`."
        )
    # Initialize process(es) for DeepSpeed
    deepspeed.init_distributed(dist_backend="hccl")
    logger.info("DeepSpeed is enabled.")


def init_deepspeed_inference(model, model_name_or_path, use_hpu_graphs, is_meta):
    # Initialize the model
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

    world_size, rank, local_rank = initialize_distributed_hpu()

    model = model.eval()
    ds_inference_kwargs = {"dtype": torch.bfloat16}
    ds_inference_kwargs["tensor_parallel"] = {"tp_size": world_size}
    ds_inference_kwargs["enable_cuda_graph"] = use_hpu_graphs
    # Make sure all devices/nodes have access to the model checkpoints
    if is_meta:
        checkpoints_json = "checkpoints.json"
        write_checkpoints_json(model_name_or_path, local_rank, checkpoints_json)

    torch.distributed.barrier()

    if is_meta:
        ds_inference_kwargs["checkpoint"] = checkpoints_json

    ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(model.config)
    model = deepspeed.init_inference(model, **ds_inference_kwargs)
    return model.module


def set_cpu_running_env():
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_BF16"


def load_model(
    model_name,
    tokenizer_name,
    device="cpu",
    use_hpu_graphs=False,
    cpu_jit=False,
    use_cache=True,
    peft_path=None,
    use_deepspeed=False,
    hf_access_token=None,
    dtype=torch.bfloat16
):
    """
    Load the model and initialize the tokenizer.

    Args:
        model_name (str): The name of the model.
        device (str, optional): The device for the model. Defaults to 'cpu'. The valid value is 'cpu' or 'hpu'.
        use_hpu_graphs (bool, optional): Whether to use HPU graphs. Defaults to False. Only set when device is hpu.

    Returns:
        None

    Raises:
        ValueError: If the model is not supported, ValueError is raised.
    """
    print("Loading model {}".format(model_name))
    if device == "hpu":
        if use_deepspeed:
            import_deepspeed()
        # Tweak generation so that it runs faster on Gaudi
        from optimum.habana.transformers.modeling_utils import (
            adapt_transformers_to_gaudi,
        )

        adapt_transformers_to_gaudi()
    else:
        set_cpu_running_env()
    MODELS[model_name] = {}
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False
        if (
            re.search("llama", model_name, re.IGNORECASE)
            or re.search("neural-chat-7b-v2", model_name, re.IGNORECASE)
        )
        else True,
        use_auth_token=hf_access_token,
    )
    config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_access_token)
    load_to_meta = model_on_meta(config)
    if peft_path and device == "hpu" and use_deepspeed and load_to_meta:
        logger.warn("PEFT could not work in deepspeed sharded checkpt loading mode, set load_to_meta to False")
        load_to_meta = False
    if device == "hpu" and use_deepspeed and load_to_meta:
        with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    elif re.search("flan-t5", model_name, re.IGNORECASE):
        with smart_context_manager(use_deepspeed=use_deepspeed):
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_auth_token=hf_access_token,
                quantization_config=bitsandbytes_quant_config,
            )
    elif (
        re.search("gpt", model_name, re.IGNORECASE)
        or re.search("mpt", model_name, re.IGNORECASE)
        or re.search("bloom", model_name, re.IGNORECASE)
        or re.search("llama", model_name, re.IGNORECASE)
        or re.search("opt", model_name, re.IGNORECASE)
        or re.search("neural-chat-7b-v1", model_name, re.IGNORECASE)
        or re.search("neural-chat-7b-v2", model_name, re.IGNORECASE)
    ):
        with smart_context_manager(use_deepspeed=use_deepspeed):
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_auth_token=hf_access_token,
            )
    else:
        raise ValueError(
            f"Unsupported model {model_name}, only supports FLAN-T5/LLAMA/MPT/GPT/BLOOM/OPT/NEURAL-CHAT now."
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

    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if device == "hpu":
        if peft_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, peft_path)
            model = model.to(torch.bfloat16)
            model = model.merge_and_unload()

        if not use_deepspeed:
            model = model.eval().to("hpu")
            if use_hpu_graphs:
                from habana_frameworks.torch.hpu import wrap_in_hpu_graph
                model = wrap_in_hpu_graph(model)

        if use_deepspeed:
            model = init_deepspeed_inference(
                model=model,
                model_name_or_path=model_name,
                use_hpu_graphs=use_hpu_graphs,
                is_meta=load_to_meta,
            )
    else:
        if peft_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, peft_path)
            model = model.to(torch.bfloat16) if dtype == torch.bfloat16 else model.to(torch.float32)

        if device == "cpu" and torch_dtype == torch.bfloat16:
            import intel_extension_for_pytorch as intel_ipex

            model = intel_ipex.optimize(
                model.eval(),
                dtype=torch.bfloat16,
                inplace=True,
                level="O1",
                auto_kernel_selection=True,
            )
            if cpu_jit and (re.search("mpt-7b", model_name, re.IGNORECASE)
                            or re.search("neural-chat-7b-v1", model_name, re.IGNORECASE)):
                from .models.mpt.mpt_trace import jit_trace_mpt_7b, MPTTSModelForCausalLM

                model.config.use_cache = use_cache
                model = jit_trace_mpt_7b(model)
                config = AutoConfig.from_pretrained(
                    model_name, use_auth_token=hf_access_token
                )
                model = MPTTSModelForCausalLM(
                    model, config, use_cache=use_cache, model_dtype=torch.bfloat16
                )

    if not model.config.is_encoder_decoder:
        tokenizer.padding_side = "left"

    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    MODELS[model_name]["model"] = model
    MODELS[model_name]["tokenizer"] = tokenizer
    print("model loaded")


output_token_len = 0
def predict_stream(**params):
    """
    Generates streaming text based on the given parameters and prompt.

    Args:
        params (dict): A dictionary containing the parameters for text generation.
        `device` (string): Specifies the device type for text generation. It can be either "cpu" or "hpu".
        `prompt` (string): Represents the initial input or context provided to the text generation model.
        `temperature` (float): Controls the randomness of the generated text.
                               Higher values result in more diverse outputs.
        `top_p` (float): Specifies the cumulative probability threshold for using in the top-p sampling strategy.
                         Smaller values make the output more focused.
        `top_k` (int): Specifies the number of highest probability tokens to consider in the top-k sampling strategy.
        `repetition_penalty` (float): Controls the penalty applied to repeated tokens in the generated text.
                                      Higher values discourage repetition.
        `max_new_tokens` (int): Limits the maximum number of tokens to be generated.
        `do_sample` (bool): Determines whether to use sampling-based text generation.
                            If set to True, the output will be sampled; otherwise,
                            it will be determined by the model's top-k or top-p strategy.
        `num_beams` (int): Controls the number of beams used in beam search.
                           Higher values increase the diversity but also the computation time.
        `model_name` (string): Specifies the name of the pre-trained model to use for text generation.
                               If not provided, the default model is "mosaicml/mpt-7b-chat".
        `num_return_sequences` (int): Specifies the number of alternative sequences to generate.
        `bad_words_ids` (list or None): Contains a list of token IDs that should not appear in the generated text.
        `force_words_ids` (list or None): Contains a list of token IDs that must be included in the generated text.
        `use_hpu_graphs` (bool): Determines whether to utilize Habana Processing Units (HPUs) for accelerated generation.
        `use_cache` (bool): Determines whether to utilize kv cache for accelerated generation.
        `dtype`(object): default is torch.bfloat16

    Returns:
        generator: A generator that yields the generated streaming text.
    """
    start_time = datetime.now()
    device = params["device"] if "device" in params else "cpu"
    temperature = float(params["temperature"]) if "temperature" in params else 0.9
    top_p = float(params["top_p"]) if "top_p" in params else 0.75
    top_k = int(params["top_k"]) if "top_k" in params else 1
    repetition_penalty = (
        float(params["repetition_penalty"]) if "repetition_penalty" in params else 1.1
    )
    max_new_tokens = (
        int(params["max_new_tokens"]) if "max_new_tokens" in params else 256
    )
    do_sample = params["do_sample"] if "do_sample" in params else True
    num_beams = int(params["num_beams"]) if "num_beams" in params else 0
    model_name = (
        params["model_name"] if "model_name" in params else "mosaicml/mpt-7b-chat"
    )
    num_return_sequences = (
        params["num_return_sequences"] if "num_return_sequences" in params else 1
    )
    bad_words_ids = params["bad_words_ids"] if "bad_words_ids" in params else None
    force_words_ids = params["force_words_ids"] if "force_words_ids" in params else None
    use_hpu_graphs = params["use_hpu_graphs"] if "use_hpu_graphs" in params else False
    use_cache = params["use_cache"] if "use_cache" in params else True
    return_stats = params["return_stats"]
    prompt = params["prompt"]
    model = MODELS[model_name]["model"]
    tokenizer = MODELS[model_name]["tokenizer"]
    errors_queue = Queue()
    task = params.get("task", "")
    dtype = params["dtype"]
    amp_dtype = torch.bfloat16 if dtype != torch.float32 else None
    if task != "":
        # add template
        if template_maps.get(task) is not None:
            template_name = template_maps.get(task)
        else:
            NotImplementedError(f"task template is not exist.")
        prompt = add_template(
            {"instruction": prompt, "input": "", "eos_token": tokenizer.eos_token},
            template_name,
        )
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    if num_beams == 0:
        num_beams = 1
        do_sample = True
    if device == "cpu":
        input_tokens = tokenizer.batch_encode_plus(
            [prompt], return_tensors="pt", padding=True
        )
        input_token_len = input_tokens.input_ids.shape[-1]
        if isinstance(model.generation_config.eos_token_id, list):
            stop_token_ids = copy.deepcopy(model.generation_config.eos_token_id)
        else:
            stop_token_ids = [model.generation_config.eos_token_id]
        end_token_id = torch.flatten(tokenizer("go.", return_tensors="pt").input_ids)[
            -1
        ]
        stop_token_ids.append(end_token_id)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=use_cache,
            num_return_sequences=num_return_sequences,
        )
        amp_enabled = True if dtype != torch.float32 else False

        def generate_output():
            try:
                with torch.no_grad():
                    with torch.cpu.amp.autocast(
                        enabled=amp_enabled, dtype=amp_dtype, cache_enabled=amp_enabled
                    ):
                        generation_kwargs = dict(
                            streamer=streamer,
                            generation_config=generation_config,
                            return_dict_in_generate=True,
                        )
                        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                            [
                                StopOnTokens(
                                    min_length=max(max_new_tokens - 20, 0),
                                    start_length=input_token_len,
                                    stop_token_id=stop_token_ids,
                                )
                            ]
                        )
                        global output_token_len
                        output_token=model.generate(**input_tokens, **generation_kwargs)
                        output_token_len=output_token.sequences[0].shape[-1]
                        return output_token
            except Exception as e:
                errors_queue.put(e)
        generation_thread = Thread(target=generate_output)
        generation_thread.start()
    elif device == "hpu":
        input_tokens_no_pad = tokenizer([prompt], return_tensors="pt")
        input_token_len = input_tokens_no_pad.input_ids.shape[-1]
        input_tokens = tokenizer.batch_encode_plus(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            max_length=max_input_len(input_token_len),
        )
        if isinstance(model.generation_config.eos_token_id, list):
            stop_token_ids = copy.deepcopy(model.generation_config.eos_token_id)
        else:
            stop_token_ids = [model.generation_config.eos_token_id]
        end_token_id = torch.flatten(tokenizer("go.", return_tensors="pt").input_ids)[
            -1
        ]
        stop_token_ids.append(end_token_id)
        generate_kwargs = {
            "stopping_criteria": StoppingCriteriaList(
                [
                    StopOnTokens(
                        min_length=max(max_new_tokens - 20, 0),
                        start_length=input_token_len,
                        stop_token_id=stop_token_ids,
                    )
                ]
            )
        }
        # Move inputs to target device(s)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(model.device)

        # Generation configuration
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.use_cache = use_cache
        generation_config.do_sample = do_sample
        generation_config.num_beams = num_beams
        generation_config.bad_words_ids = bad_words_ids
        generation_config.force_words_ids = force_words_ids
        generation_config.num_return_sequences = num_return_sequences
        generation_config.static_shapes = model_is_optimized(model.config)
        generation_config.top_k = top_k
        # TODO there is an issue when top_p is used in Habana
        # generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.repetition_penalty = repetition_penalty

        def generate_output():
            try:
                with torch.no_grad():
                    output_token=model.generate(
                        **input_tokens,
                        **generate_kwargs,
                        streamer=streamer,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=max_new_tokens,
                        lazy_mode=True,
                        hpu_graphs=use_hpu_graphs,
                        ignore_eos=False,
                    )
                    global output_token_len
                    output_token_len=output_token.sequences[0].shape[-1]
                    return output_token
            except Exception as e:
                errors_queue.put(e)

        generation_thread = Thread(target=generate_output)
        generation_thread.start()
    else:
        raise ValueError(
            f"Unsupported device type {device}, only supports cpu and hpu now."
        )
    output_word_len = 0
    generation_thread.join(0.1)
    if generation_thread.is_alive():
        pass
    else:
        thread_exception = errors_queue.get()
        raise thread_exception
    # prevent crash if no words are coming out
    first_word_output_time = datetime.now()
    for new_text in streamer:
        if len(new_text) == 0:
            continue
        if output_word_len == 0:
            first_word_output_time = datetime.now()
        output_word_len += 1
        yield new_text

    end_time = datetime.now()

    time.sleep(0.1)
    duration = int((end_time - start_time).total_seconds() * 1000)
    first_token_latency = int(
        (first_word_output_time - start_time).total_seconds() * 1000 * 3/4
    )

    msecond_per_token = (
        duration  / (output_token_len - input_token_len)
        if output_word_len != 1
        else 0
    )
    if return_stats:
        stats = {
            "input_token_len": str(input_token_len),
            "output_token_len": str(output_token_len),
            "duration": str(duration) + " ms",
            "first_token_latency": str(first_token_latency) + " ms",
            "msecond_per_token": str(msecond_per_token) + " ms",
        }
        yield "\n| {:<22} | {:<27} |\n".format("Key", "Value")
        yield "| " + "-"*22 + " | " + "-"*27 + " |" + "\n"
        for key, value in stats.items():
            yield "| {:<22} | {:<27} |\n".format(key, value)


def predict(**params):
    """
    Generates streaming text based on the given parameters and prompt.

    Args:
        params (dict): A dictionary containing the parameters for text generation.
        `device` (string): Specifies the device type for text generation. It can be either "cpu" or "hpu".
        `prompt` (string): Represents the initial input or context provided to the text generation model.
        `temperature` (float): Controls the randomness of the generated text.
                               Higher values result in more diverse outputs.
        `top_p` (float): Specifies the cumulative probability threshold for using in the top-p sampling strategy.
                         Smaller values make the output more focused.
        `top_k` (int): Specifies the number of highest probability tokens to consider in the top-k sampling strategy.
        `repetition_penalty` (float): Controls the penalty applied to repeated tokens in the generated text.
                                      Higher values discourage repetition.
        `max_new_tokens` (int): Limits the maximum number of tokens to be generated.
        `do_sample` (bool): Determines whether to use sampling-based text generation.
                            If set to True, the output will be sampled; otherwise,
                            it will be determined by the model's top-k or top-p strategy.
        `num_beams` (int): Controls the number of beams used in beam search.
                           Higher values increase the diversity but also the computation time.
        `model_name` (string): Specifies the name of the pre-trained model to use for text generation.
                               If not provided, the default model is "mosaicml/mpt-7b-chat".
        `num_return_sequences` (int): Specifies the number of alternative sequences to generate.
        `bad_words_ids` (list or None): Contains a list of token IDs that should not appear in the generated text.
        `force_words_ids` (list or None): Contains a list of token IDs that must be included in the generated text.
        `use_hpu_graphs` (bool): Determines whether to utilize Habana Processing Units (HPUs) for accelerated generation.
        `use_cache` (bool): Determines whether to utilize kv cache for accelerated generation.
        `dtype`(object): default is torch.bfloat16

    Returns:
        generator: A generator that yields the generated streaming text.
    """
    device = params["device"] if "device" in params else "cpu"
    temperature = float(params["temperature"]) if "temperature" in params else 0.9
    top_p = float(params["top_p"]) if "top_p" in params else 0.75
    top_k = int(params["top_k"]) if "top_k" in params else 1
    repetition_penalty = (
        float(params["repetition_penalty"]) if "repetition_penalty" in params else 1.1
    )
    max_new_tokens = (
        int(params["max_new_tokens"]) if "max_new_tokens" in params else 256
    )
    do_sample = params["do_sample"] if "do_sample" in params else True
    num_beams = int(params["num_beams"]) if "num_beams" in params else 0
    model_name = (
        params["model_name"] if "model_name" in params else "mosaicml/mpt-7b-chat"
    )
    num_return_sequences = (
        params["num_return_sequences"] if "num_return_sequences" in params else 1
    )
    bad_words_ids = params["bad_words_ids"] if "bad_words_ids" in params else None
    force_words_ids = params["force_words_ids"] if "force_words_ids" in params else None
    use_hpu_graphs = params["use_hpu_graphs"] if "use_hpu_graphs" in params else False
    use_cache = params["use_cache"] if "use_cache" in params else False
    prompt = params["prompt"]
    model = MODELS[model_name]["model"]
    tokenizer = MODELS[model_name]["tokenizer"]
    dtype = params["dtype"]
    amp_dtype = torch.bfloat16 if dtype != torch.float32 else None

    task = params.get("task", "")

    if task != "":
        # add template
        if template_maps.get(task) is not None:
            template_name = template_maps.get(task)
        else:
            NotImplementedError(f"task template is not exist.")
        prompt = add_template(
            {"instruction": prompt, "input": "", "eos_token": tokenizer.eos_token},
            template_name,
        )

    if num_beams == 0:
        num_beams = 1
        do_sample = True
    if device == "cpu":
        input_tokens = tokenizer.batch_encode_plus(
            [prompt], return_tensors="pt", padding=True
        )
        input_token_len = input_tokens.input_ids.shape[-1]
        if isinstance(model.generation_config.eos_token_id, list):
            stop_token_ids = copy.deepcopy(model.generation_config.eos_token_id)
        else:
            stop_token_ids = [model.generation_config.eos_token_id]
        end_token_id = torch.flatten(tokenizer("go.", return_tensors="pt").input_ids)[
            -1
        ]
        stop_token_ids.append(end_token_id)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=use_cache,
            num_return_sequences=num_return_sequences,
        )
        amp_enabled = True if dtype != torch.float32 else False

        with torch.no_grad():
            with torch.cpu.amp.autocast(
                enabled=amp_enabled, dtype=amp_dtype, cache_enabled=amp_enabled
            ):
                generation_kwargs = dict(
                    generation_config=generation_config, return_dict_in_generate=True
                )
                generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    [
                        StopOnTokens(
                            min_length=max(max_new_tokens - 20, 0),
                            start_length=input_token_len,
                            stop_token_id=stop_token_ids,
                        )
                    ]
                )
                generation_output = model.generate(**input_tokens, **generation_kwargs)
    elif device == "hpu":
        input_tokens_no_pad = tokenizer([prompt], return_tensors="pt")
        input_token_len = input_tokens_no_pad.input_ids.shape[-1]
        input_tokens = tokenizer.batch_encode_plus(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            max_length=max_input_len(input_token_len),
        )
        if isinstance(model.generation_config.eos_token_id, list):
            stop_token_ids = copy.deepcopy(model.generation_config.eos_token_id)
        else:
            stop_token_ids = [model.generation_config.eos_token_id]
        end_token_id = torch.flatten(tokenizer("go.", return_tensors="pt").input_ids)[
            -1
        ]
        stop_token_ids.append(end_token_id)
        generate_kwargs = {
            "stopping_criteria": StoppingCriteriaList(
                [
                    StopOnTokens(
                        min_length=max(max_new_tokens - 20, 0),
                        start_length=input_token_len,
                        stop_token_id=stop_token_ids,
                    )
                ]
            )
        }
        # Move inputs to target device(s)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(model.device)

        # Generation configuration
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.use_cache = use_cache
        generation_config.do_sample = do_sample
        generation_config.num_beams = num_beams
        generation_config.bad_words_ids = bad_words_ids
        generation_config.force_words_ids = force_words_ids
        generation_config.num_return_sequences = num_return_sequences
        generation_config.static_shapes = model_is_optimized(model.config)
        generation_config.top_k = top_k
        # TODO there is an issue when top_p is used in Habana
        # generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.repetition_penalty = repetition_penalty

        with torch.no_grad():
            generation_output = model.generate(
                **input_tokens,
                **generate_kwargs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                lazy_mode=True,
                hpu_graphs=use_hpu_graphs,
                ignore_eos=False,
            )
    output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    if "### Response:" in output:
        return output.split("### Response:")[1].strip()
    return output


def main():
    args = parse_args()
    base_model_path = args.base_model_path

    # Check the validity of the arguments
    if not 0 < args.temperature <= 1.0:
        raise ValueError("Temperature must be between 0 and 1.")
    if not 0 <= args.top_p <= 1.0:
        raise ValueError("Top-p must be between 0 and 1.")
    if not 0 <= args.top_k <= 200:
        raise ValueError("Top-k must be between 0 and 200.")
    if not 1.0 <= args.repetition_penalty <= 2.0:
        raise ValueError("Repetition penalty must be between 1 and 2.")
    if not 0 <= args.num_beams <= 8:
        raise ValueError("Number of beams must be between 0 and 8.")
    if not 32 <= args.max_new_tokens <= 1024:
        raise ValueError(
            "The maximum number of new tokens must be between 32 and 1024."
        )

    # User can use DeepSpeed to speedup the inference On Habana Gaudi processors.
    # If the DeepSpeed launcher is used, the env variable _ will be equal to /usr/local/bin/deepspeed
    # For multi node, the value of the env variable WORLD_SIZE should be larger than 8
    use_deepspeed = (
        "deepspeed" in os.environ["_"]
        or ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 8)
        and args.habana
    )

    if args.habana:
        # Set seed before initializing model.
        from optimum.habana.utils import set_seed

        set_seed(args.seed)
    else:
        from transformers import set_seed

        set_seed(args.seed)

    tokenizer_path = (
        args.tokenizer_name if args.tokenizer_name is not None else base_model_path
    )
    datatype = torch.bfloat16 if args.dtype != "float32" else torch.float32
    load_model(
        base_model_path,
        tokenizer_path,
        device="hpu" if args.habana else "cpu",
        use_hpu_graphs=args.use_hpu_graphs,
        cpu_jit=args.jit,
        use_cache=args.use_kv_cache,
        peft_path=args.peft_model_path,
        use_deepspeed=True if use_deepspeed and args.habana else False,
        hf_access_token=args.hf_access_token,
        dtype=datatype
    )

    if args.habana:
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

        world_size, rank, args.local_rank = initialize_distributed_hpu()
    if args.habana and rank in [-1, 0]:
        logger.info(f"Args: {args}")
        logger.info(f"n_hpu: {world_size}, bf16")
    # warmup, the first time inference take longer because of graph compilation

    for idx, instruction in enumerate(args.instructions):
        set_seed(args.seed)
        idxs = f"{idx+1}"
        out = predict(
            model_name=base_model_path,
            device="hpu" if args.habana else "cpu",
            prompt=instruction,
            task=args.task,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0.0,
            use_hpu_graphs=args.use_hpu_graphs,
            use_cache=args.use_kv_cache,
            num_return_sequences=args.num_return_sequences,
            dtype=datatype,
        )

    for idx, instruction in enumerate(args.instructions):
        set_seed(args.seed)
        idxs = f"{idx+1}"
        if args.local_rank in [-1, 0]:
            logger.info("=" * 30 + idxs + "=" * 30)
            logger.info(f"Instruction: {instruction}")
            logger.info("Response: ")
        for new_text in predict_stream(
            model_name=base_model_path,
            device="hpu" if args.habana else "cpu",
            prompt=instruction,
            task=args.task,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0.0,
            use_hpu_graphs=args.use_hpu_graphs,
            use_cache=args.use_kv_cache,
            num_return_sequences=args.num_return_sequences,
            dtype=datatype,
            return_stats= args.return_stats,
        ):
            if args.local_rank in [-1, 0]:
                print(new_text, end="", flush=True)
        if args.local_rank in [-1, 0]:
            logger.info("=" * (60 + len(idxs)))

    for idx, instruction in enumerate(args.instructions):
        set_seed(args.seed)
        idxs = f"{idx+1}"
        if args.local_rank in [-1, 0]:
            logger.info("=" * 30 + idxs + "=" * 30)
            logger.info(f"Instruction: {instruction}")
            start_time = time.time()
            logger.info("Response: ")
        out = predict(
            model_name=base_model_path,
            device="hpu" if args.habana else "cpu",
            prompt=instruction,
            task=args.task,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0.0,
            use_hpu_graphs=args.use_hpu_graphs,
            use_cache=args.use_kv_cache,
            num_return_sequences=args.num_return_sequences,
            dtype=datatype
        )
        if args.local_rank in [-1, 0]:
            print(f"whole sentence out = {out}")
            logger.info(f"duration: {time.time() - start_time}" + ' s')
            logger.info("=" * (60 + len(idxs)))


if __name__ == "__main__":
    main()
