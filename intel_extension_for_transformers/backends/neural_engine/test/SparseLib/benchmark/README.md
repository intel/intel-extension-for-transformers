# Benchmark for SparseLib
To perform accuracy test and performance test for kernels in [SparseLib](https://github.com/intel/intel-extension-for-transformers/tree/develop/intel_extension_for_transformers/backends/neural_engine/SparseLib).

## Build
```shell
mkdir build
cd build
cmake ..
make -j
```

## Usage
```shell
[<environment_variable>...] ./benchmark <mode> <kernel_type> <kernel_specification>... <config>... 
```
+ `mode` values can be `perf` for perfomance test or `acc` for accuracy test.
+ `kernel_type` is one of
    + [sparse_matmul](#sparse_matmul)
    + [eltwiseop](#eltwiseop)
    + [layernorm_ba](#layernorm_ba)
+ `kernel_specification` specifies information like what algorithm is used for [sparse_matmul](#sparse_matmul).
+ `config` contains information of the test case, for example tensor shapes.

### Environment variables
`BENCHMARK_ITER`: how many iterations to run to calculate kernel execution time. The default value is `100`.

`BENCHMARK_NO_REFRESH`: by default, we refresh data for src tensor in every iteration before executing the kernel. If the value of the variable is set to 1, we use the same src tensor for every iteration.


### sparse_matmul
Currently we are just doing 2D matrix multiplication: A(MxK) x B(KxN) = C(MxN).

Current algorithms for sparse_matmul:

+ [spmm_avx512f](#spmmavx512f)
+ [spmm_vnni](#spmmvnni)
+ [spmm_amx_bf16_x16](#spmmamxbf16x16)

#### spmm_avx512f
```shell
[<environment_variable>...] ./benchmark <mode> sparse_matmul avx512f <M> <K> <N> <sparse_ratio> [<post-op>...]
```
There can be more than one post-op. Please refer to [eltwiseop](#eltwiseop) to see supported `algorithm`.
##### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf sparse_matmul avx512f 1024 1024 1024 0.7 gelu exp
```

#### spmm_vnni
```shell
[<environment_variable>...] ./benchmark <mode> sparse_matmul vnni <M> <K> <N> <sparse_ratio> <micro_bs> <is_fp32_out> <has_append_sum> <micro_oc> <sub_func_level> [<post-op>...]
```

You can use `-1` to use default config for `micro_bs`, `micro_oc`,`sub_func_level`.

`sub_func_level` can be positive integer up to `ssd::subfunc_level::subfunc_level_MAX`. Higher value means more code folding.

##### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf sparse_matmul vnni 1024 1024 1024 0.7 -1 0 0 -1 -1 gelu
```

#### spmm_amx_bf16_x16
##### Build
Please make sure amx instructions are supported on your machine.
```shell
mkdir build
cd build
cmake .. -DSPARSE_LIB_USE_AMX=True
make -j
```

```shell
[<environment_variable>...] ./benchmark <mode> sparse_matmul amx_bf16_x16 <M> <K> <N> <sparse_ratio> <micro_bs> <micro_oc> <is_bf16_out> [<post-op>...]
```

##### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf sparse_matmul amx_bf16_x16 1024 1024 1024 0.9 64 -1 1 gelu
```


### eltwiseop
```shell
[<environment_variable>...] ./benchmark <mode> eltwiseop <M> <N> <data_type>_<algorithm>[+<data_type>_<algorithm>[+...]] <ranges>
```
Though eltwiseop is element-wise, we mostly append it to sparse GEMM in SparseLib. Thus we are using 2D shape specification here.

There can be more than one postop and they should be concatenated by `+`.

`ranges` specifies the interval where values of src tensor are located. It has the form of `<lower_bound>,<upper_bound>`.

Current supported `data_type` :
+ `bf16`
+ `fp32`

Please note in each test case, there should be **only one** `data_type`, for example, `fp32_gelu+bf16_exp` is invalid.

Here `algorithm` just means what numerical calculation [eltwiseop](#eltwiseop) is doing. Current supported `algorithm` :
+ `exp`
+ `gelu`
+ `tanh`
+ `relu`
+ `quantize` : converts `fp32` to `u8`. This `algorithm` doesn't require a `data_type` prefix.
+ `dequantize`: converts `u8` to `fp32`. This `algorithm` doesn't require a `data_type` prefix.

#### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf eltwiseop 1024 1024 dequantize+fp32_relu+quantize -10.0,10.0
```

### layernorm_ba
```shell
[<environment_variable>...] ./benchmark <mode> layernorm_ba <M> <N> <src_dt> <dst_dt>
```
Please note that the src_dt only support fp32,dst_dt support fp32 and s8/u8.
#### Examples
```shell
BENCHMARK_ITER=100 BENCHMARK_NO_REFRESH=0 ./benchmark perf layernorm_ba 1024 1024 fp32 fp32
```

## For developers
To add benchmark support for a newly-added kernel, you may need to follow several steps:

+ Create a subdir for the kernel under `<benchmark_dir>` and make sure you add files `bench_<kernel_name>.hpp` and `bench_<kernel_name>.cpp`. Implement the `test_<kernel_name>` function as the entrance of benchmark procedure of the kernel. Then Include `bench_<kernel_name>.hpp` in `<benchmark_dir>/benchmark.cpp` and add a branch for the function `test_<kernel_name>` in `main` function.
+ You may want to implement several utility functions in other source files under the subdir for the kernel, for example:
    + `get_true_data_<kernel_name>` : to calculate reference output for the kernel.
    + `check_result_<kernel_name>` : to compare reference output and kernel output.
    + `gen_case_<kernel_name>` : to use the config input by user to generate test case, i.e. operator descriptors and runtime data.
    + `run_bench_<kernel_name>` : benchmark procedure of the kernel.
    
    Feel free to add other utility functions if you want.
+ Add a case for the kernel in functions `calc_flop` and `get_refresh_data_idx` in `<benchmark_dir>/benchmark_utils.cpp`.
