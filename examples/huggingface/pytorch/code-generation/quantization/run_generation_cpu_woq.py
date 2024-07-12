import argparse
import time
import os
import torch
from transformers import AutoTokenizer, AutoConfig
from optimum.utils import NormalizedConfigManager
from intel_extension_for_transformers.transformers import (
    BitsAndBytesConfig,
    RtnConfig,
    AwqConfig,
    TeqConfig,
    GPTQConfig,
    AutoRoundConfig,
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
parser.add_argument("--use_neural_speed", action="store_true")
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
# ============BitsAndBytes configs==============
parser.add_argument("--bitsandbytes", action="store_true")
# ============WeightOnlyQuant configs===============
parser.add_argument("--woq", action="store_true")
parser.add_argument(
    "--woq_algo",
    default="Rtn",
    choices=["Rtn", "Awq", "Teq", "GPTQ", "AutoRound"],
    help="Weight-only algorithm.",
)
parser.add_argument(
    "--bits",
    type=int,
    default=8,
    choices=[4, 8],
)
parser.add_argument(
    "--weight_dtype",
    type=str,
    default="int8",
    choices=[
        "int8",
        "int4",  # int4 == int4_clip
        "int4_clip",
        "fp4",  # fp4 == fp4_e2m1
        "fp4_e2m1_bnb",
        "fp4_e2m1",
        "nf4",
        "fp8",  # fp8 == fp8_e4m3
        "fp8_e5m2",
        "fp8_e4m3",
    ],
)
parser.add_argument(
    "--scale_dtype",
    type=str,
    default="fp32",
    choices=["fp32", "bf16", "fp8"],
)
parser.add_argument(
    "--compute_dtype",
    type=str,
    default="fp32",
    choices=["fp32", "bf16", "int8"],
)
parser.add_argument("--group_size", type=int, default=128)
parser.add_argument("--scheme", default=None)
parser.add_argument(
    "--layer_wise",
    action="store_true",
    help="Use layer wise to do quantization",
)
parser.add_argument(
    "--n_samples", type=int, default=512, help="Number of calibration data samples."
)
parser.add_argument(
    "--seq_len",
    type=int,
    default=2048,
    help="Calibration dataset sequence max length, this should align with your model config",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Calibration batchsize.",
)
# ============GPTQ configs==============
parser.add_argument(
    "--desc_act",
    action="store_true",
    help="Whether to apply the activation order GPTQ heuristic.",
)
parser.add_argument(
    "--damp_percent",
    type=float,
    default=0.01,
    help="Percent of the average Hessian diagonal to use for dampening.",
)
parser.add_argument(
    "--true_sequential",
    action="store_true",
    help="Whether to quantize layers within a transformer block in their original order.",
)
parser.add_argument(
    "--blocksize",
    type=int,
    default=128,
    help="Block size. sub weight matrix size to run GPTQ.",
)
parser.add_argument(
    "--static_groups",
    action="store_true",
    help="Use determined group to do quantization",
)
# ============AUTOROUND configs==============
parser.add_argument(
    "--lr",
    type=float,
    default=None,
    help="learning rate, if None, it will be set to 1.0/iters automatically",
)
parser.add_argument(
    "--minmax_lr",
    type=float,
    default=None,
    help="minmax learning rate, if None,it will beset to be the same with lr",
)
parser.add_argument("--autoround_iters", default=200, type=int, help="num iters for autoround calibration.")
parser.add_argument(
    "--disable_quanted_input",
    action="store_true",
    help="whether to use the output of quantized block to tune the next block",
)
parser.add_argument(
    "--quant_lm_head",
    action="store_true",
    help="whether to quant the lm head layer",
)
# ============Harness configs============
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
parser.add_argument("--accuracy", action="store_true")
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
if args.use_neural_speed:
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1)
else:
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

# woq/bitsandbytes config setting
quantization_config = None
if args.woq:
    if args.woq_algo == "Rtn":
        quantization_config = RtnConfig(
            bits=args.bits,
            sym=True if args.scheme == "sym" else False,
            group_size=args.group_size,
            compute_dtype=args.compute_dtype,
            scale_dtype=args.scale_dtype,
            weight_dtype=args.weight_dtype,
            layer_wise=args.layer_wise,
            use_ipex=args.use_ipex,
        )
    elif args.woq_algo == "Awq":
        quantization_config = AwqConfig(
            tokenizer=tokenizer,
            dataset=args.dataset,
            bits=args.bits,
            zero_point=False if args.scheme == "sym" else True,
            group_size=args.group_size,
            seq_len=args.seq_len,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            compute_dtype=args.compute_dtype,
            scale_dtype=args.scale_dtype,
            weight_dtype=args.weight_dtype,
            use_ipex=args.use_ipex,
        )
    elif args.woq_algo == "Teq":
        quantization_config = TeqConfig(
            tokenizer=tokenizer,
            dataset=args.dataset,
            bits=args.bits,
            sym=True if args.scheme == "sym" else False,
            group_size=args.group_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            compute_dtype=args.compute_dtype,
            scale_dtype=args.scale_dtype,
            weight_dtype=args.weight_dtype,
            use_ipex=args.use_ipex,
        )
    elif args.woq_algo == "GPTQ":
        quantization_config = GPTQConfig(
            tokenizer=tokenizer,
            dataset=args.dataset,
            bits=args.bits,
            desc_act=args.desc_act,
            damp_percent=args.damp_percent,
            sym=True if args.scheme == "sym" else False,
            blocksize=args.blocksize,
            static_groups=args.static_groups,
            group_size=args.group_size,
            n_samples=args.n_samples,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            compute_dtype=args.compute_dtype,
            scale_dtype=args.scale_dtype,
            weight_dtype=args.weight_dtype,
            layer_wise=args.layer_wise,
            true_sequential=args.true_sequential,
            use_ipex=args.use_ipex,
        )
    elif args.woq_algo == "AutoRound":
        quantization_config = AutoRoundConfig(
            tokenizer=tokenizer,
            dataset=args.dataset,
            bits=args.bits,
            sym=True if args.scheme == "sym" else False,
            n_samples=args.n_samples,
            group_size=args.group_size,
            compute_dtype=args.compute_dtype,
            scale_dtype=args.scale_dtype,
            weight_dtype=args.weight_dtype,
            iters=args.autoround_iters,
            seq_len=args.seq_len,
            lr=args.lr,
            minmax_lr=args.minmax_lr,
            disable_quanted_input=args.disable_quanted_input,
            quant_lm_head = args.quant_lm_head,
            use_ipex=args.use_ipex,
        )
    else:
        assert False, "Please set the correct '--woq_algo'"
# bitsandbytes
elif args.bitsandbytes:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )
else:
    print("The quantization_config is None.")

# get optimized model
if quantization_config is not None:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        trust_remote_code=args.trust_remote_code,
        _commit_hash=args._commit_hash,
        use_neural_speed=args.use_neural_speed,
    )
elif args.load_in_4bit or args.load_in_8bit:
    # CPU device usage is provided by intel-extension-for-transformers.
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        _commit_hash=args._commit_hash,
        use_neural_speed=args.use_neural_speed,
    )
else:
    print("Didn't do Weight Only Quantization.")

# save model
if args.output_dir is not None and ((args.woq or args.load_in_4bit or args.load_in_8bit) and not args.use_neural_speed):
    user_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # to validate woq model accuracy 
    args.model = args.output_dir

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
