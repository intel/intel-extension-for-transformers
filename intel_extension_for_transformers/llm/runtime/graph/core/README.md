# Highly optimized low precision kernels
Our kernels are based on x64 template library JBLAS.
## Support Matrix
Limited by the graph framework, we only add kernels which accept float tensor as input and output tensor.

input dtype | output dtype | compute type | compute ISA
--- |---|---|---
float32 | float32 | float32 | AVX2 
float32 | float32 | float32 | AVX512F
float32(1) | float32(2) | int8 | AVX512_VNNI
float32(1) | float32(2) | int8 | AVX_VNNI
float32(1) | float32(2) | int8 | AMX_INT8
float32/bf16 | float32/bf16 | bf16 | AMX_BF16
float32/fp16 | float32/fp16 | fp16 | AVX512_FP16

(1):dynamic quantization for input tensor, each batch of input tensor has the same quantization
group size of weight tensor;  
(2):dynamic dequantization for output tensor.

### Weight-only quantization support
dtype | algo | group size
--- | --- | ---
int4 | symmetric int8 clip quant | aligned to 8, -1(1)
int4 | symmetric int4 full range | aligned to 8, -1(1)
int4 | asymmetric int4 full range | aligned to 8, -1(1)
int8 | symmetric | aligned to 8, -1(1)
fp4 | | aligned to 8
nf4 | | aligned to 8

(1): group size=-1 means per channel quantization on output channel (or group size equals to input channel size).

NOTE: AMX_INT8 requires group size is aligend to 128(best hardware efficiency)

### Fastest configuration for CPUs
Codename | Weight Config
---|---
Sapphire Rapids<br>Ice Lake<br>Cascade Lake<br>Cooper Lake<br>Tiger Lake<br>Rocket Lake | int4(symmetric clip or full range)<br>group size=-1<br>compute type=int8
Skylake |  int4(symmetric clip or full range)<br>group size=-1<br>compute type=fp32
