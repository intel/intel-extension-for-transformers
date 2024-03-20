# QBits
## Introduction
QBits is a computing extension module introduced by ITREX to accelerate pytorch operators, especially low-bits gemm on x86 platform. QBtis based on a BLAS library named [BesTLA](https://github.com/intel/neural-speed/tree/main/bestla) which has been highly optimized in terms of thread parallelism, instruction parallelism, data parallelism, and cache reuse.  

Users can accelerate weight-only-quantization on CPU platform via the operators provided by QBits, which already included in ITREX's woq feature, see [here](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/huggingface/pytorch/text-generation/quantization/README.md) for more details.
## Example
```python
import intel_extension_for_transformers.qbits as qbits

# replace linear weight option1: RTN quantize
"""
    RTN quantize the input weight tensor.

    Args:
        fp32_weight (torch.Tensor): Input fp32 weight tensor.
        transpose (bool): Whether to transpose the weight tensor (required for woq_quantize with KxN weight shape).
        blocksize (int): Blocksize for weight-only quantization.
        compute_type (str): Computation type (fp32/bf16/int8). fp32 will leverage AVX2/AVX512F to compute, bf16 will be AMX_BF16, int8 will be VNNI/AMX_INT8.
        weight_type (str): Quantization type (int8/int4_clip/int4_fullrange/nf4/fp4_e2m1).
        scale_type (str): Scale type (fp32/bf16).
        asym (bool): Whether to use asymmetric quantization.

    Returns:
        torch.Tensor: Quantized weight tensor.
"""
pack_weight = qbits.woq_quantize(
        fp32_weight, transpose, blocksize, compute_type, weight_type, scale_type, asym) # pack_weight can use to replace linear layer weight.

# replace linear weight option2: repack weight
"""
    Repack quantized weight, scale, zp and gidx tensor to QBits needed format.

    Args:
        qweight (torch.Tensor):  quantized weight tensor(KxN shape), dtype must be int8
        scale (torch.Tensor): scale tensor, dtype must be fp32
        zp (torch.Tensor): zero-point tensor, dtype must be int8
        g_idx (torch.Tensor): shuffle index used by GPTQ, dtype must be int32.
        blocksize (int): Blocksize for weight-only quantization.
        compute_type (str): Computation type (fp32/bf16/int8). fp32 will leverage AVX2/AVX512F to compute, bf16 will be AMX_BF16, int8 will be VNNI/AMX_INT8.
        weight_type (str): Quantization type (int8/int4_clip/int4_fullrange/nf4/fp4_e2m1).
        scale_type (str): Scale type (fp32/bf16).
        asym (bool): Whether to use asymmetric quantization.

    Returns:
        torch.Tensor: Quantized weight tensor.
"""
pack_weight = qbits.woq_packq(
        quantized_weight, scale, zp, g_idx, weight_type, scale_type, compute_type, asym, blocksize) # pack_weight can use to replace linear layer weight.

# low-bit gemm compute
"""
    Low-bits gemm provided by QBits.

    Args:
        activation (torch.Tensor): Input activation tensor, support fp32/bf16.
        pack_weight (torch.Tensor): Woq weight created by qbits.woq_quantize
        bias (torch.Tensor): Bias tensor, must be fp32, if bias is empty woq_linear will not add bias.
        output (torch.Tensor): Output tensor, support fp32/bf16, shape must be MxN.
        compute_type (str): Computation type (fp32/bf16/int8).fp32 will leverage AVX2/AVX512F to compute, bf16 will leverage AMX_BF16 to compute, int8 will leverage VNNI/AMX_INT8 to compute.
        weight_type (str): Quantization type (int8/int4_clip/int4_fullrange/nf4/fp4_e2m1).
        scale_type (str): Scale type (fp32/bf16).
        asym (bool): Whether to use asymmetric quantization.
"""
qbits.woq_linear(
        activation, pack_weight, bias, output, n, add_bias, compute_type, weight_type, scale_type, asym)
```
please refer [here](https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/transformers/llm/operator/csrc/qbits_ut) for more QBits operators usage.
