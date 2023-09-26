# Highly Optimized Low Precision Kernels
Our kernels are based on x64 template library [jblas](../../../library/jblas).
## Support Matrix
Limited by the graph framework, we only add kernels which accept float tensor as input and output tensor.

input dtype | output dtype | compute type | compute ISA
--- |---|---|---
float32 | float32 | float32 | AVX2 
float32 | float32 | float32 | AVX512F
float32^1^ | float32^2^ | int8 | AVX512_VNNI
float32^1^ | float32^2^ | int8 | AVX_VNNI
float32^1^ | float32^2^ | int8 | AMX_INT8
float32/bf16 | float32/bf16 | bf16 | AMX_BF16
float32/fp16 | float32/fp16 | fp16 | AVX512_FP16

^1^: per-batch and per-K group-wise dynamic quantization for input tensor, where per-K group-wise also applies to weight quantization
group size of weight tensor; support both symmetric and asymmetric quantization.  
^2^: per-batch dynamic dequantization for output tensor.

### Weight-only Quantization Support
dtype | algo | group size
--- | --- | ---
int4 | symmetric int8 truncated quant^2^ | multipler of 8, -1^1^
int4 | symmetric int4 full range^3^ | multipler of 8, -1^1^
int4 | asymmetric int4 full range^3^ | multipler of 8, -1^1^
int8 | symmetric | multipler of 8, -1^1^
fp4 | | multipler of 8
nf4 | | multipler of 8

^1^: group size=-1 means per channel quantization on output channel (or group size equals to input channel size).  
^2^: truncated quant means keep the high 4 bits of int8 quantization result for model saving and computation.  
^3^: full range is a method that utilizes -8 value of int4 range compared with normal int4 range [-7,7].  

NOTE: AMX_INT8 requires group size is aligend to 128 (best hardware efficiency)

## Fusion Support
We support three kinds of kernel fusion for transformer models: QKV, MHA(multi-head attention) and FFN(feed-forward network) fusion.  

fusion type | models | runtime ISA
--- | --- | ---
QKV | GPT-J<br>LLaMA | AMX_INT8,AVX512_VNNI
MHA | GPT-J<br>LLaMA | AMX_BF16
FFN | GPT-J<br>LLaMA<br>BLOOM<br>ChatGLM<br>Falcon<br>MPT | AMX_INT8, AVX512_VNNI, AVX512F and AMX_BF16


## Fastest Configuration for CPUs
Codename | Weight Config | Runtime ISA
---|---
Sapphire Rapids | any int4<br>group size=-1<br>compute type=int8 | AMX_INT8
Ice Lake<br>Cascade Lake<br>Cooper Lake<br>Tiger Lake<br>Rocket Lake | any int4<br>group size=-1<br>compute type=int8 | AVX512_VNNI
Skylake |  any 4bits<br>group size=-1<br>compute type=fp32 | AVX512F

