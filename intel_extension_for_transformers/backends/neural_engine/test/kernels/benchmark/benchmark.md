- [Benchmark for Kernels](#benchmark-for-kernels)
  - [Build](#build)
  - [Usage](#usage)
    - [sparse\_matmul](#sparse_matmul)
      - [spmm\_avx512f](#spmm_avx512f)
      - [spmm\_vnni](#spmm_vnni)
      - [spmm\_amx\_bf16\_x16](#spmm_amx_bf16_x16)
    - [eltwiseop](#eltwiseop)
    - [layernorm\_ba](#layernorm_ba)
    - [transpose\_matmul](#transpose_matmul)
      - [matmul\_avx512f\_p2031\_p2013](#matmul_avx512f_p2031_p2013)
      - [matmul\_vnni\_noperm\_p2031\_p1302](#matmul_vnni_noperm_p2031_p1302)
    - [softmax](#softmax)
    - [attention](#attention)
    - [Static MHA](#static-mha)
    - [dynamic\_quant\_matmul](#dynamic_quant_matmul)
    - [dynamic\_quant](#dynamic_quant)
- [For developers](#for-developers)

# Benchmark for Kernels
This is a tool to perform accuracy test and performance test for kernel in [kernels](../../../kernels).

## Build
```shell
cd <path/to/this/repo>/intel_extension_for_transformers/backends/neural_engine
mkdir build
cd build
cmake .. -DNE_WITH_SPARSELIB=ON -DNE_WITH_SPARSELIB_ONLY=ON -DNE_WITH_SPARSELIB_BENCHMARK=ON
make -j
```

## Usage
```shell
[<environment_variable>...] ./benchmark <mode> <kernel_type> <kernel_configs>
```
+ `mode = {perf,acc}`: `perf` for perfomance test, `acc` for accuracy test, and `perf,acc` for both perfomance test and accuracy test.
+ `kernel_type` is one of
    + [sparse_matmul](#sparse_matmul)
    + [transpose_matmul](#transpose_matmul)
    + [eltwiseop](#eltwiseop)
    + [layernorm_ba](#layernorm_ba)
    + [softmax](#softmax)
+ `kernel_configs` contains information of the test case, for example tensor shapes. Refer to the kernel introduction for detail.##### Build

+ Environment variables
    + `BENCHMARK_ITER`: how many iterations to run to calculate kernel execution time. The default value is `100`.
    + `BENCHMARK_NO_REFRESH`: by default, we refresh data for src tensor in every iteration before executing the kernel. If the value of the variable is set to 1, we use the same src tensor for every iteration.


### sparse_matmul
Currently we done 2D sparse matrix multiplication: `A(MxK) x B(KxN) = C(MxN)` where `A` is a sparse matrix.

Current algorithms for sparse_matmul:

+ [spmm_avx512f](../../../kernels/docs/kernel_desc/kernel_avx512f.md)
+ [spmm_vnni](../../../kernels/docs/kernel_desc/kernel_vnni.md)
+ [spmm_amx_bf16_x16](../../../kernels/docs/kernel_desc/kernel_amx.md)

#### spmm_avx512f
```shell
[<environment_variable>...] ./benchmark <mode> sparse_matmul avx512f <M> <K> <N> <sparse_ratio> [<post-op>...]
```
`M,K,N` is the shape of matmul
`sparse_ratio` is the sparse ratio of weight.
There can be more than one post-op. Please refer to [eltwiseop](#eltwiseop) to see supported `algorithm`.
##### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf sparse_matmul avx512f 1024 1024 1024 0.7 gelu exp
```

#### spmm_vnni
```shell
[<environment_variable>...] ./benchmark <mode> sparse_matmul vnni <M> <K> <N> <sparse_ratio> <micro_bs> <output_fp32> <has_append_sum> <micro_oc> <sub_func_level> [<post-op>...]
```
- `M,K,N` is the shape of matmul.
- `sparse_ratio` is the sparse ratio of weight.
- `micro_bs` is used in [3D_inference](../../../kernels/docs/kernel_desc/3D_inference.md), and set to `-1` to avoid to use it.
- `output_fp32 = {0,1}` set to `1` to output fp32, while set to 0 to output int8.
- `has_append_sum = {0,1}`  set to  `1` to append sum on output otherwise set to 0.
- `micro_oc` is used to specify the data parallel in OC dim, and set to `-1` to automaticlly calculate.
- `sub_func_level` is a positive integer up to `jd::ssd::subfunc_level::subfunc_level_MAX`. Higher value means more code folding.
- `[<post-op>...]` There can be more than one post-op. Please refer to [eltwiseop](#eltwiseop) to see supported `algorithm`.


You can use `-1` to use default config for `micro_bs`, `micro_oc`,`sub_func_level`.

##### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf sparse_matmul vnni 1024 1024 1024 0.7 -1 0 0 -1 -1 gelu
```

#### spmm_amx_bf16_x16
> **Note** Please make sure amx instructions are supported on your machine. You should build a benchmark with amx.
```shell
mkdir build
cd build
cmake .. -DSPARSE_LIB_USE_AMX=True -DNE_WITH_SPARSELIB=ON -DNE_WITH_SPARSELIB_ONLY=ON -DNE_WITH_SPARSELIB_BENCHMARK=ON
make -j
```

```shell
[<environment_variable>...] ./benchmark <mode> sparse_matmul amx_bf16_x16 <M> <K> <N> <sparse_ratio> <micro_bs> <micro_oc> <output_bf16> [<post-op>...]
```
- `M,K,N` is the shape of matmul.
- `sparse_ratio` is the sparse ratio of weight.
- `micro_bs` is used in [3D_inference](../../../kernels/docs/kernel_desc/3D_inference.md), and set to `-1` to avoid to use it.
- `micro_oc` is used to specify the data parallel in OC dim, and set to `-1` to automatically calculate.
- `output_bf16 = {0,1}` set to `1` to output bf16, while set to 0 to output fp32.
- `sub_func_level` is a positive integer up to `jd::ssd::subfunc_level::subfunc_level_MAX`. Higher value means more code folding.
- `[<post-op>...]` There can be more than one post-op. Please refer to [eltwiseop](#eltwiseop) to see supported `algorithm`.

You can use `-1` to use default config for `micro_bs`, `micro_oc`.
##### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf sparse_matmul amx_bf16_x16 1024 1024 1024 0.9 64 -1 -1 1 gelu
```


### eltwiseop
```shell
[<environment_variable>...] ./benchmark <mode> eltwiseop <M> <N> <jd::data_type>_<algorithm>[+<jd::data_type>_<algorithm>[+...]] <ranges>
```
Eltwiseop is a series of element-wise calculation, and we have appended it to sparse GEMM in Kernels. We use 2D shape specification in eltwiseop.
- `M,N` is the shape of input and output.
- `ranges` specifies the interval where values of src tensor are located. It has the form of `<lower_bound>,<upper_bound>`.
- `<jd::data_type>_<algorithm>` describe a kind of element-wise calculation.There can be more than one postop and they should be concatenated by `+`.
- `jd::data_type = {bf16,fp32}` There should be **only one** `jd::data_type` in each test case, for example, `fp32_gelu+bf16_exp` is invalid.
-  `algorithm`  Current supported :
    + `exp`
    + `gelu`
    + `tanh`
    + `relu`
    + `quantize` : converts `fp32` to `u8`. This `algorithm` doesn't require a `jd::data_type` prefix.
    + `dequantize`: converts `u8` to `fp32`. This `algorithm` doesn't require a `jd::data_type` prefix.

#### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf eltwiseop 1024 1024 dequantize+fp32_relu+quantize -10.0,10.0
```

### layernorm_ba
```shell
[<environment_variable>...] ./benchmark <mode> layernorm_ba <M> <N> <src_dt> <dst_dt>
```
`layernorm_ba` is layernorm for tansposed input.
- `M,N` is the shape of input and output.
- `src_dt={fp32}`is input data type. It only supports fp32 now.
- `dst_dt={fp32,s8,u8}`is output data type.
#### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf layernorm_ba 1024 1024 fp32 fp32
```
### transpose_matmul
`transpose_matmul` are a series 4D matmuls without distinguishing between dense and sparse, they can be calculated with `alpha * src0(bs0, bs1, M, K) x src1(bs0, bs1, K, N) + beta * scr2(M, N) = dst(bs0, bs1, M, N)` where `alpha, beta, src2` are optional. Refer to [transpose_matmul](../../../kernels/docs/kernel_desc/kernel_transpose_matmul.md) for detail.
#### matmul_avx512f_p2031_p2013
```shell
[<environment_variable>...] ./benchmark <mode> transpose_matmul avx512f_p2031_p2013 <M> <K> <N> <bs0> <bs1> <has_binary_add> <alpha> <beta> [tile_m] [tile_n]
```
The data type of its inputs and output is fp32. The permutation of both input is {2,0,3,1}. Output has no permutation.
- `M,K,N,bs0,bs1` are the shape of inputs.
- `has_binary_add = {0,1}`  set to  `1` to append binary add, otherwise set to 0.
- `alpha,beta` are coefficients of matmul and binary_add. `beta` will be not effective if `has_binary_add = 0`.
- `tile_m, tile_n` are optional. They specify the tile shape when calculating matmul.
##### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf transpose_matmul avx512f_p2031_p2013 32 64 32 8 12 1 1 0.25 8 2
```

#### matmul_vnni_noperm_p2031_p1302
```shell
[<environment_variable>...] ./benchmark <mode> transpose_matmul vnni_noperm_p2031_p1302 <M> <K> <N> <bs0> <bs1>
```
The data type of its inputs and output is int8. The permutation of the second input is {2,0,3,1}. The permutation of the output is {1,3,0,2}.
- `M,K,N,bs0,bs1` are the shape of inputs.
- `has_binary_add = {0,1}`  set to  `1` to append binary add, otherwise set to 0.
- `alpha,beta` are coefficients of matmul and binary_add. `beta` will be not effective if `has_binary_add = 0`.
- `tile_m, tile_n` are optional. They specify the tile shape when calculating matmul.
##### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf transpose_matmul vnni_noperm_p2031_p1302 32 32 64 8 12
```

### softmax
```shell
[<environment_variable>...] ./benchmark <mode> softmax <spec_type> <input_shape> <input_dt> <output_dt>
```
Softmax in Kernels only suuport LUT with int8 input and bf16 output.
- `spec_type = {lut}`
- `input_shape = d0xd1x...`: `d0,d1,...` are the dimensions of input, `x` is the delimiter.
- `input_dt = {u8}`: input data type only support unsigned int8 now.
- `output_dt = {bf16}`: output data type only support bf16 now.
#### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf softmax lut 256x256 u8 bf16
```
### attention
```shell
[<environment_variable>...] ./benchmark <mode> attention <head_num> <head_size> <batch_size> <seq_len> <sparsity> <dst_type>
```
#### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf attention 1 64 8 128 0.5 1
```


### Static MHA
```shell
[<environment_variable>...] ./benchmark <mode> mha_dense <batch_size> <seq_len> <head_num> <head_size> [dst_type] [padding_mask] [badd_dim]
```

- `batch_size` / `seq_len` / `head_num` / `head_size` are the the problem size.
- `dst_type` is the output data type, could be one of `u8` / `s8` / `fp32`; default to `u8` if leave blank.
- `padding_mask` a positive integer which is the length of sequence(s) before padding; leave blank pr set to negative value so that it defaults to `seq_len` (no padding).
- `badd_dim` dimension of the tensor to apply binary_add before performing softmax; leave blank pr set to non-positive value to disable the binary add; set to `1`-`4` to apply the binary_add with broadcasting if necessary.
#### Examples
```shell
BENCHMARK_ITER=10 BENCHMARK_NO_REFRESH=0 ./benchmark perf mha_dense 1 2048 16 256 u8 -1 4
```

### dynamic_quant_matmul
```shell
[<environment_variable>...] ./benchmark <mode> dynamic_quant_matmul <batch_size> <m> <n> <k> <large_weight_threshold> <add_bias>
```
- `batch_size` / `m` / `n` / `k` are the the problem size.
- `large_weight_threshold` should be 0-1, if weight*large_weight_threshold large than L2-cache size, kernel will select large_wei jit-path.
- `add_bias` could be one of `true` / `false`, indicate whether add a bias value after gemm.

Please note that dynamic_quant_matmul only supports s8-activation, s8-weight and s8-destination, so the user does not need to specify the data type.

#### Examples
```shell
BENCHMARK_ITER=10 BENCHMARK_NO_REFRESH=0 ./benchmark perf dynamic_quant_matmul 1 512 1280 1280
```

### dynamic_quant
```shell
[<environment_variable>...] ./benchmark <mode> dynamic_quant <channel_num> <quantize_dim_elt_num> <src_data_type>
```
- `channel_num` the number of channels in input-tensor.
- `quantize_dim_elt_num` the number of elements in the quantized dim.
- `src_data_type` could be one of `fp32` / `bf16`, indicate the data type of input-tensor.

Please note that dynamic_quant only supports per-channel symmetric quantization.

#### Examples
```shell
BENCHMARK_ITER=10 BENCHMARK_NO_REFRESH=0 ./benchmark perf dynamic_quant 1280 1280 bf16
```

# For developers
To add benchmark support for a newly-added kernel, you may need to follow several steps:

+ Create a subdir for the kernel under `<benchmark_dir>` and make sure you add files `bench_<kernel_name>.hpp` and `bench_<kernel_name>.cpp`. Implement the `test_<kernel_name>` function as the entrance of benchmark procedure of the kernel. Then Include `bench_<kernel_name>.hpp` in `<benchmark_dir>/benchmark.cpp` and add a branch for the function `test_<kernel_name>` in `main` function.
+ You may want to implement several utility functions in other source files under the subdir for the kernel, for example:
    + `get_true_data_<kernel_name>` : to calculate reference output for the kernel.
    + `check_result_<kernel_name>` : to compare reference output and kernel output.
    + `gen_case_<kernel_name>` : to use the config input by user to generate test case, i.e. operator descriptors and runtime data.
    + `run_bench_<kernel_name>` : benchmark procedure of the kernel.

    Feel free to add other utility functions if you want.
+ Add a case for the kernel in functions `calc_flop` and `get_refresh_data_idx` in `<benchmark_dir>/benchmark_utils.cpp`.
