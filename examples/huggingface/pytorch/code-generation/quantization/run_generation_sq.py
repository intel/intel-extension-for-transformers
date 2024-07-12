import argparse
import time
import os
import torch
from transformers import AutoTokenizer, AutoConfig
from optimum.utils import NormalizedConfigManager
from intel_extension_for_transformers.transformers import (
    MixedPrecisionConfig,
    SmoothQuantConfig,
)
from intel_extension_for_transformers.transformers import (
    AutoModelForCausalLM,
)

parser = argparse.ArgumentParser()

# ============Main configs============
parser.add_argument(
    "--model", nargs="?", default="bigcode/starcoderbase", const="bigcode/starcoderbase"
)
parser.add_argument("--trust_remote_code", action="store_true")
parser.add_argument("--_commit_hash", default=None, type=str)
parser.add_argument("--dataset", nargs="?", default="mbpp", const="mbpp")
parser.add_argument("--dtype", type=str, default="int8")
parser.add_argument(
    "--max_new_tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
# ============Benchmark configs==============
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--benchmark_iters", default=100, type=int, help="num iter")
parser.add_argument("--num_warmup", default=10, type=int, help="num warmup")
parser.add_argument(
    "--prompt_size", default=32, type=int, help="generate dummy input_ids size"
)
# ============MixedPrecision configs==============
parser.add_argument("--mixed_precision", action="store_true")
# ============SmoothQuant configs==============
parser.add_argument("--sq", action="store_true")
parser.add_argument("--alpha", default=0.5, help="Smooth quant parameter.")
parser.add_argument(
    "--calib_n_samples", default=100, type=int, help="Smooth quant calibration samples."
)
parser.add_argument(
    "--seq_len", default=512, type=int, help="Smooth quant calibration input length."
)
parser.add_argument("--batch_size", default=1, type=int, help="batch size num.")
parser.add_argument("--padding", action="store_true")
parser.add_argument("--shuffle", action="store_true")
# sq alpha "auto" parameters
parser.add_argument("--scale_sharing", action="store_true")
parser.add_argument(
    "--init_alpha", default=0.5, type=float, help="Smooth quant parameter."
)
parser.add_argument(
    "--alpha_min", default=0.0, type=float, help="Smooth quant parameter."
)
parser.add_argument(
    "--alpha_max", default=1.0, type=float, help="Smooth quant parameter."
)
parser.add_argument(
    "--alpha_step", default=0.1, type=float, help="Smooth quant parameter."
)
parser.add_argument("--shared_criterion", default="max", type=str)
parser.add_argument("--do_blockwise", action="store_true")
parser.add_argument(
    "--restore_sq_model_from_json",
    action="store_true",
    help="restore ipex quantized model from output_dir/best_configure.json",
)
# ============Harness configs============
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--tasks", default=None, help="Evaluation tasks")
parser.add_argument(
    "--limit", default=None, type=int, help="Limit number of samples to eval"
)
parser.add_argument("--allow_code_execution", action="store_true")
parser.add_argument("--generation_only", action="store_true")
parser.add_argument("--postprocess", action="store_false")
parser.add_argument("--save_references", action="store_true")
parser.add_argument("--save_generations", action="store_true")
parser.add_argument("--instruction_tokens", default=None)
parser.add_argument("--save_generations_path", default="generations.json")
parser.add_argument("--load_generations_path", default=None)
parser.add_argument("--metric_output_path", default="evaluation_results.json")
parser.add_argument(
    "--load_generations_intermediate_paths",
    type=str,
    nargs="*",
    help="List of paths for saving the intermediate code generations",
)
# ============Generation config============
parser.add_argument("--max_length_generation", default=512, type=int)
parser.add_argument("--check_references", action="store_true")
parser.add_argument("--max_memory_per_gpu", type=str, default=None)
parser.add_argument(
    "--prompt",
    type=str,
    default="prompt",
    help="Prompt type to use for generation in HumanEvalPack tasks",
)
parser.add_argument(
    "--modeltype",
    default="causal",
    help="AutoModel to use, it can be causal or seq2seq",
)
parser.add_argument(
    "--limit_start",
    type=int,
    default=0,
    help="Optional offset to start from when limiting the number of samples",
)
parser.add_argument(
    "--save_every_k_tasks",
    type=int,
    default=-1,
    help="Optional saving after every k tasks",
)
parser.add_argument(
    "--left_padding",
    action="store_true",
    help="Force left padding, needed for models like chatglm3-6b",
)
parser.add_argument(
    "--load_data_path",
    type=str,
    default=None,
    help="Path of additional data to load for the tasks",
)
# ============Evaluation configs==============
parser.add_argument("--prefix", default="")
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--temperature", default=0.2, type=float)
parser.add_argument("--top_p", default=0.95, type=float)
parser.add_argument("--top_k", default=0, type=int)
parser.add_argument("--n_samples", default=1, type=int)
parser.add_argument("--eos", default="<|endoftext|>", type=str)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    truncation_side="left",
    padding_side="right",
    trust_remote_code=args.trust_remote_code,
    _commit_hash=args._commit_hash,
)

config = AutoConfig.from_pretrained(
    args.model,
    torchscript=(
        True
        if (
            args.sq
            or args.woq_algo in ["Awq", "Teq"]
            or (args.int8 or args.int8_bf16_mixed or args.benchmark)
        )
        else False
    ),  # torchscript will force `return_dict=False` to avoid jit errors
    use_cache=True,  # to use kv cache.
    trust_remote_code=args.trust_remote_code,
    _commit_hash=args._commit_hash,
)
if not tokenizer.eos_token:
    if tokenizer.bos_token:
        tokenizer.eos_token = tokenizer.bos_token
        print("bos_token used as eos_token")
    else:
        raise ValueError("No eos_token or bos_token found")

tokenizer.pad_token = tokenizer.eos_token

# Generation
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

# mp/sq/woq/bitsandbytes config setting
quantization_config = None
if args.mixed_precision:
    quantization_config = MixedPrecisionConfig(dtype="bfloat16")  # default is bfloat16
elif args.sq:
    excluded_precisions = ["bf16"]
    quantization_config = SmoothQuantConfig(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        n_samples=args.calib_n_samples,
        batch_size=args.batch_size,
        excluded_precisions=excluded_precisions,
        alpha=args.alpha if args.alpha == "auto" else float(args.alpha),
        scale_sharing=args.scale_sharing,
        init_alpha=args.init_alpha,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        shared_criterion=args.shared_criterion,
        do_blockwise=args.do_blockwise,
        shuffle=args.shuffle,
        padding=args.padding,
        num_beams=generate_kwargs["num_beams"],
    )

# get optimized model
if quantization_config is not None:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
        _commit_hash=args._commit_hash,
    )


# save model
if args.output_dir is not None:
    tokenizer.save_pretrained(args.output_dir)
    if args.sq:
        quantization_config.remove_redundant_parameters()
        config.quantization_config = quantization_config
        config.save_pretrained(args.output_dir)
        user_model.save(args.output_dir)
        user_model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            trust_remote_code=args.trust_remote_code,
            _commit_hash=args._commit_hash,
        )
    elif args.mixed_precision:
        user_model.save_pretrained(args.output_dir)

if args.restore_sq_model_from_json:
    from intel_extension_for_transformers.transformers.llm.quantization.sq_utils import (
        recover_model_from_json,
    )
    user_model = recover_model_from_json(
        args.model,
        os.path.join(args.output_dir, "qconfig.json"),
        args.trust_remote_code,
    )

elif not (args.sq or args.mixed_precision):
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        _commit_hash=args._commit_hash,
    )

if args.benchmark:
    normalized_config = NormalizedConfigManager.get_normalized_config_class(
        user_model.config.model_type
    )(user_model.config)
    num_layers = normalized_config.num_layers
    num_attention_heads = normalized_config.num_attention_heads
    hidden_size = normalized_config.hidden_size
    d_k = hidden_size // num_attention_heads
    num_beams = 1
    if hasattr(normalized_config, "num_key_value_heads"):
        num_key_value_heads = normalized_config.num_key_value_heads
    if hasattr(normalized_config, "multi_query_group_num"):
        num_key_value_heads = normalized_config.multi_query_group_num

    num_iter = args.benchmark_iters
    num_warmup = args.num_warmup

    total_latency = 0
    for j in range(args.max_new_tokens):
        total_time = 0.0
        with torch.inference_mode(), torch.no_grad():
            for i in range(num_iter):
                tic = time.time()
                if j == 0:
                    input_ids = torch.randint(
                        1,
                        tokenizer.vocab_size,
                        size=(args.batch_size, args.prompt_size),
                    )
                    input_bs, input_len = input_ids.shape
                    attention_mask = torch.ones(input_bs, input_len)
                    position_ids = (
                        torch.arange(input_len).unsqueeze(0).expand(input_bs, -1)
                    )
                    if user_model.config.model_type == "gpt_bigcode":
                        new_shape = [input_bs, 0, d_k * 2]
                        dummy_tensor = torch.zeros(size=new_shape)
                        past_key_values = tuple([dummy_tensor] * num_layers)
                    else:
                        if not (args.int8 or args.int8_bf16_mixed):
                            new_shape = [input_bs, num_key_value_heads, 0, d_k]
                            past_key_values = [
                                (
                                    torch.zeros(size=new_shape).contiguous(),
                                    torch.zeros(size=new_shape).contiguous(),
                                )
                                for _ in range(num_layers)
                            ]
                            past_key_values = tuple(past_key_values)

                        else:
                            new_shape = [input_bs, num_key_value_heads, 1, d_k]
                            beam_idx_tmp = torch.zeros(
                                (2048, int(input_bs * num_beams)), dtype=torch.long
                            ).contiguous()
                            past_key_values = [
                                (
                                    torch.zeros(
                                        1, 0, 0, 1, dtype=torch.long
                                    ).contiguous(),
                                    torch.zeros(size=new_shape).contiguous(),
                                    torch.zeros(size=new_shape).contiguous(),
                                    beam_idx_tmp,
                                )
                                for _ in range(num_layers)
                            ]
                            past_key_values = tuple(past_key_values)

                inp = {
                    "input_ids": input_ids,
                    "past_key_values": past_key_values,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
                out = user_model(**inp)
                gen_id = torch.argmax(out[0][:, -1:, :], axis=-1)
                gen_text = tokenizer.batch_decode(gen_id, skip_special_tokens=True)
                toc = time.time()
                if i >= num_warmup:
                    total_time += toc - tic

        print("\n", "-" * 10, "Summary:", "-" * 10)
        print("Generated token index:", j + 1)
        latency = total_time / (num_iter - num_warmup)
        print("Inference latency: %.5f sec." % latency)
        throughput = (num_iter - num_warmup) / total_time
        print("Throughput: {} samples/sec".format(throughput))

        input_ids = gen_id
        past_key_values = out[1]
        attention_mask = torch.ones(
            (attention_mask.shape[0], attention_mask.shape[1] + 1)
        )
        position_ids = torch.tensor([[len(inp["position_ids"])]])
        total_latency += latency

    average_latency = total_latency / args.max_new_tokens
    print("Average inference latency: %.5f sec." % latency)
    average_throughput = args.max_new_tokens / total_latency
    print("Average throughput: {} samples/sec".format(throughput))


if args.accuracy:
    from intel_extension_for_transformers.transformers.llm.evaluation.bigcode_eval import evaluate

    results = evaluate(
        model=user_model,
        tokenizer=tokenizer,
        tasks=args.tasks,
        batch_size=args.eval_batch_size,
        args=args,
    )
    print(results)
