import torch
import inspect
from functools import wraps
torch.ops.load_library("build/libweight_only_jblasop.so")


def capture_args(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(f)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arg_strs = []
        for name, value in bound_args.arguments.items():
            arg_strs.append(f'{name}={value}')
        result = ', '.join(arg_strs)
        print(result)
        return f(*args, **kwargs)
    return wrapper


@capture_args
def test_fp32in_fp32out(m, n, k, blocksize, compute_type, weight_type, transpose, add_bias, dump_tensor_info=False):
    torch.manual_seed(0)
    activation = torch.rand(m, k, dtype=torch.float)
    wei_row = k
    wei_col = n
    if transpose:
        wei_row, wei_col = wei_col, wei_row
    raw_wei = torch.rand(wei_row, wei_col, dtype=torch.float)
    if dump_tensor_info:
        print(raw_wei)
    compress_wei = torch.ops.weight_only_jblasop.qbits_quantize(
        raw_wei, transpose, blocksize, compute_type, weight_type)
    revert_wei = torch.zeros(wei_row, wei_col, dtype=torch.float)
    torch.ops.weight_only_jblasop.qbits_dequantize(
        compress_wei, revert_wei, transpose, compute_type, weight_type)
    bias = torch.rand(n, dtype=torch.float)*10
    if dump_tensor_info:
        print(revert_wei)
    tar_dst = torch.zeros(m, n, dtype=torch.float)
    if transpose:
        revert_wei = torch.transpose(revert_wei, 0, 1)
    ref_dst = torch.matmul(activation, revert_wei)
    if add_bias:
        torch.ops.weight_only_jblasop.qbits_f32in_f32out_linear_with_bias(
            activation, compress_wei, bias, tar_dst, k, n, compute_type, weight_type)
    else:
        torch.ops.weight_only_jblasop.qbits_f32in_f32out_linear_without_bias(
            activation, compress_wei, tar_dst, n, k, n, compute_type, weight_type)
    if add_bias:
        ref_dst += bias
    if dump_tensor_info:
        print(tar_dst)
        print(ref_dst)
    if torch.allclose(tar_dst, ref_dst, rtol=0.03):
        print("ok")
    else:
        print("fail")


configs = {"s8_scalef32": {"int8", "fp32"}, "s4clip_scalef32": {"int8", "fp32", "bf16"}, "s4fullrange_scalef32": {
    "int8", "fp32", "bf16"}, "fp4bnb_scalef32": {"fp32", "bf16"}, "fp4e2m1_scalef32": {"fp32", "bf16"}, "nf4_scalef32": {"fp32", "bf16"}}

blocksizes = [8, 12, 64]
do_trans = [False, True]
add_bias = [False, True]

for weight_type in configs:
    m = 255
    n = 1023
    k = 512 # contain unalign calc error bug currently. 
    for compute_type in configs[weight_type]:
        for blocksize in blocksizes:
            if compute_type == "int8" and blocksize % 8 != 0:
                continue
            for trans in do_trans:
                for bias in add_bias:
                    test_fp32in_fp32out(m, n, k, blocksize,
                                        compute_type, weight_type, trans, bias)
