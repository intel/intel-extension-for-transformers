import copy
import glob
import os
import sys
import shutil
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import check_min_version

from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    get_repo_root,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)
from optimum.habana.utils import check_habana_frameworks_version
from optimum.habana.utils import check_optimum_habana_min_version
from optimum.habana.utils import set_seed

"Compute 'sliding window' perplexity on a dataset. Validated against the calculations reported in arXiv 2306.15595"
def compute_perplexity(model, tokenizer, inputs, samples_num=None, add_start_token=True, max_length=None, sliding_window=256, truncate=False):

    if samples_num:
        encodings = inputs[: samples_num]

    device='hpu'
    max_tokenized_len = max_length - 1 if add_start_token else max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
        sliding_window = max_tokenized_len

    nlls = []
    t_ppl = time.perf_counter()
    for encoding_index in range(0, len(encoded_texts)):
        labels = torch.tensor(encoded_texts[encoding_index:encoding_index+1])
        seq_len = labels.size(1)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, sliding_window):

            end_loc = min(begin_loc + max_tokenized_len, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc].to(device)

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * input_ids.size(dim=0)).to(device)
                input_ids = torch.cat(
                    [bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)

            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    ppl_duration = time.perf_counter() - t_ppl
    return {'max_length': max_length, 'ppl': ppl, 'duration': ppl_duration, 'samples_num': samples_num}


def print_memory_stats(p_info=""):
    from optimum.habana.utils import get_hpu_memory_stats
    separator = "-" * 90
    print(separator)
    print("{}".format(p_info))
    mem = get_hpu_memory_stats()
    for k, v in mem.items():
        print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))
    print(separator)

def adjust_batch(batch, size):
    curr_size = batch["input_ids"].shape[1]
    if curr_size >= size:
        adjusted_batch = {
            "input_ids": batch["input_ids"][:, :size],
            "attention_mask": batch["attention_mask"][:, :size],
        }
    else:
        adjusted_batch = {}
        for k in batch.keys():
            last_colm = batch[k][:, -1]
            expanded = last_colm.tile((size - curr_size, 1)).T
            adjusted_batch[k] = torch.concat([batch[k], expanded], 1)
    assert adjusted_batch["input_ids"].shape[1] == size
    assert adjusted_batch["attention_mask"].shape[1] == size
    return adjusted_batch


def count_hpu_graphs():
    return len(glob.glob(".graph_dumps/*PreGraph*"))


def setup_distributed(args):
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "0"))
    args.global_rank = int(os.getenv("RANK", "0"))


def setup_const_serialization(const_serialization_path):
    import uuid

    const_serialization_path = os.path.join(const_serialization_path + uuid.uuid4().hex)
    os.makedirs(const_serialization_path)
    from habana_frameworks.torch.hpu import enable_const_section_serialization

    print("Serializing const params to {}".format(const_serialization_path))
    enable_const_section_serialization(const_serialization_path, False, True)


def setup_env(args):
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version("4.34.0")
    # check_optimum_habana_min_version("1.9.0.dev0")
    # TODO: SW-167588 - WA for memory issue in hqt prep_model
    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")
    # TODO let's set the lazy mode on
    # os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    # os.environ.setdefault("PT_HPU_MAX_COMPOUND_OP_SIZE", "1")

    if args.global_rank == 0 and not args.torch_compile:
        os.environ.setdefault("GRAPH_VISUALIZATION", "true")
        shutil.rmtree(".graph_dumps", ignore_errors=True)

    if args.world_size > 0:
        os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")

    # Tweak generation so that it runs faster on Gaudi
    from intel_extension_for_transformers.transformers.modeling.modeling_gaudi import adapt_transformers_to_gaudi

    adapt_transformers_to_gaudi()


def setup_device(args):
    if args.device == "hpu":
        import habana_frameworks.torch.core as htcore

        if args.fp8:
            htcore.hpu_set_env()
    return torch.device(args.device)

# patching LinearAllreduce to use ScopedLinearAllReduce
def patch_scoped_linear_all_reduce(model):
    from deepspeed.module_inject.layers import LinearAllreduce

    from optimum.habana.transformers.models.modeling_all_models import ScopedLinearAllReduce

    for name, module in model.named_children():
        if type(module) is LinearAllreduce:
            SL = ScopedLinearAllReduce(mod=module)
            setattr(model, name, SL)
        patch_scoped_linear_all_reduce(module)


def get_torch_compiled_model(model):
    # model.model = torch.compile(model.model, backend="aot_hpu_inference_backend")
    model.model = torch.compile(model.model)
    return model


def setup_model(args, model_dtype, model_kwargs):

    config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            torch_dtype=model_dtype,
            **model_kwargs)
    # config.max_position_embeddings = max(config.max_position_embeddings, 20000)
    config.tensor_split = False
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=model_dtype,
            **model_kwargs)
    if args.quant_config:
        import habana_quantization_toolkit

        habana_quantization_toolkit.prep_model(model)

    model = model.eval()
    model = model.to("hpu")

    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph

        if check_habana_frameworks_version("1.13.0") and model.config.model_type == "falcon":
            model = wrap_in_hpu_graph(model, hash_with_views=False)
        else:
            model = wrap_in_hpu_graph(model)

    if args.torch_compile and model.config.model_type == "llama":
        model = get_torch_compiled_model(model)

    return model


def setup_distributed_model(args, model_dtype, model_kwargs):
    import deepspeed

    deepspeed.init_distributed(dist_backend="hccl")
    config = AutoConfig.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs)
    config.tensor_split = False
    load_to_meta = model_on_meta(config)

    if load_to_meta:
        # Construct model with fake meta tensors, later will be replaced on devices during ds-inference ckpt load
        with deepspeed.OnDevice(dtype=model_dtype, device="meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)

        # Model loaded to meta is managed differently
        checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")

        # For PEFT models, write the merged model on disk to be able to load it on the meta device
        # if args.peft_model is not None:
        #     merged_model_dir = "/tmp/text_generation_merged_peft_model"
        #     if args.local_rank == 0:
        #         if Path(merged_model_dir).is_dir():
        #             shutil.rmtree(merged_model_dir)
        #         peft_model(args, model_dtype, **model_kwargs).save_pretrained(merged_model_dir)
        #     torch.distributed.barrier()

        write_checkpoints_json(
            args.model_name_or_path,
            # merged_model_dir if args.peft_model is not None else args.model_name_or_path,
            args.local_rank,
            checkpoints_json,
            token=None,
        )
    else:
        # TODO: revisit placement on CPU when auto-injection is possible
        with deepspeed.OnDevice(dtype=model_dtype, device="cpu"):
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs
            )
    model.eval()

    # Initialize the model
    ds_inference_kwargs = {"dtype": model_dtype}
    ds_inference_kwargs["tensor_parallel"] = {"tp_size": args.world_size}
    ds_inference_kwargs["enable_cuda_graph"] = args.use_hpu_graphs
    ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(config)
    if load_to_meta:
        ds_inference_kwargs["checkpoint"] = checkpoints_json.name

    model = deepspeed.init_inference(model, **ds_inference_kwargs)
    model = model.module
    if model.config.model_type in ["llama", "falcon"]:
        patch_scoped_linear_all_reduce(model)

    if args.quant_config:
        import habana_quantization_toolkit

        habana_quantization_toolkit.prep_model(model)

    if args.torch_compile and model.config.model_type == "llama":
        model = get_torch_compiled_model(model)

    return model


def peft_model(args, model_dtype, **model_kwargs):
    import importlib.util

    if importlib.util.find_spec("peft") is None:
        raise ImportError("The `peft` package is not installed, please run: `pip install peft`.")
    from peft import AutoPeftModelForCausalLM
    from peft.config import PeftConfigMixin

    base_model_name = PeftConfigMixin.from_pretrained(
        args.peft_model,
        token=model_kwargs["token"] if "token" in model_kwargs else None,
    ).base_model_name_or_path

    base_model_is_local = Path(base_model_name).is_dir()
    if not base_model_is_local:
        # Check if the base model path to a remote repository on the HF Hub exists
        from huggingface_hub import list_repo_files

        try:
            list_repo_files(base_model_name)
            base_model_is_remote = True
        except Exception:
            base_model_is_remote = False

    if base_model_is_local or base_model_is_remote:
        model = AutoPeftModelForCausalLM.from_pretrained(args.peft_model, torch_dtype=model_dtype, **model_kwargs)
    else:
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs)
        model = PeftModel.from_pretrained(model, args.peft_model, torch_dtype=model_dtype, **model_kwargs)

    return model.merge_and_unload()


def setup_tokenizer(args, model):
    tokenizer_kwargs = {
        "revision": "main",
        "token": None, 
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    if not model.config.is_encoder_decoder:
        tokenizer.padding_side = "left"
    # Some models like GPT2 do not have a PAD token so we have to set it if necessary
    if model.config.model_type == "llama":
        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        tokenizer.bos_token_id = model.generation_config.bos_token_id
        tokenizer.eos_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token_id = model.generation_config.pad_token_id
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return tokenizer, model


def setup_generation_config(args, model, tokenizer):
    bad_words_ids = None
    force_words_ids = None

    attention_sink_size = None
    attention_sink_window_size = None
    if hasattr(args, "attention_sink_size"):
        attention_sink_size = args.attention_sink_size
    if hasattr(args, "attention_sink_window_size"):
        attention_sink_window_size = args.attention_sink_window_size

    is_optimized = model_is_optimized(model.config)
    # Generation configuration
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.use_cache = args.use_kv_cache
    generation_config.static_shapes = True
    generation_config.bucket_size = args.bucket_size if is_optimized else -1
    # generation_config.bucket_size = args.bucket_size
    # (TODO) bucket internal will increase the shape length
    generation_config.bucket_internal = False
    # generation_config.do_sample = args.do_sample
    generation_config.num_beams = args.num_beams
    generation_config.bad_words_ids = bad_words_ids
    generation_config.force_words_ids = force_words_ids
    # generation_config.num_return_sequences = args.num_return_sequences
    generation_config.trim_logits = args.trim_logits
    # TODO notice here why can't use softmax_bf16
    generation_config.attn_softmax_bf16 = True
    generation_config.limit_hpu_graphs = args.limit_hpu_graphs
    # TODO why reuse cache  and reduce recompile false
    generation_config.reuse_cache = True
    generation_config.reduce_recompile = False
    if generation_config.reduce_recompile:
        assert generation_config.bucket_size > 0
    # TODO this will also influence
    generation_config.use_flash_attention = False
    # attention_sinks
    generation_config.attention_sink_size = attention_sink_size
    generation_config.attention_sink_window_size = attention_sink_window_size
    return generation_config


def initialize_model(args):
    args.quant_config = os.getenv("QUANT_CONFIG", "")
    init_start = time.perf_counter()
    setup_distributed(args)
    # override_prints(args.global_rank == 0 or args.verbose_workers,)
    setup_env(args)
    setup_device(args)
    set_seed(27)
    get_repo_root(args.model_name_or_path, local_rank=args.local_rank, token=None)
    use_deepspeed = args.world_size > 1
    # import pdb; pdb.set_trace()
    if use_deepspeed or args.bf16 or args.fp8:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float
        args.attn_softmax_bf16 = False

    model_kwargs = {
        "revision": "main",
        "token":None,
    }

    model_kwargs["device_map"] = "auto"
    model_kwargs["offload_folder"] = "/tmp/offload_folder/"

    if hasattr(args, "attention_sink_size") and hasattr(args, "attention_sink_window_size"):
        model_kwargs["attention_sink_size"] = args.attention_sink_size
        model_kwargs["attention_sink_window_size"] = args.attention_sink_window_size

    model = (
        setup_model(args, model_dtype, model_kwargs)
        if not use_deepspeed
        else setup_distributed_model(args, model_dtype, model_kwargs)
    )
    tokenizer, model = setup_tokenizer(args, model)
    generation_config = setup_generation_config(args, model, tokenizer)

    # if args.const_serialization_path:
    #     setup_const_serialization(args.const_serialization_path)
    if args.fp8:
        import habana_frameworks.torch.core as htcore

        print("Initializing inference mode")
        # const_marking = os.getenv("ENABLE_CONST_MARKING", "True")
        # if const_marking == "True":
            # TODO always initialize model
        htcore.hpu_initialize(model)
    init_end = time.perf_counter()
    return model, tokenizer, generation_config
