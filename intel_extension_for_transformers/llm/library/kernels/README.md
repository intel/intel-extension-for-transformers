# Transformers-Accelerated Libraries
===========================================

## Introduction

Transformers-accelerated Libraries (formerly known as **SparseLib**) is a high-performance operator computing library implemented by assembly. Transformers-accelerated Libraries contains a JIT domain, a kernel domain, and a scheduling proxy framework.

## Installation
### Build
```shell
cd %{workdir}/intel_extension_for_transformers/backends/neural_engine
mkdir build
cd build
cmake .. -DNE_WITH_TESTS=ON                   # if UT needed
         -DNE_WITH_SPARSELIB_ONLY=ON          # if kernels only
         -DNE_WITH_SPARSELIB_BENCHMARK=ON     # if benchmark needed
make -j
```

### Test
```shell
cd build
cmake .. -DNE_WITH_TESTS=ON
make -j
./test_spmm_vnni_kernel
```

### Performance
We provide a benchmark tool to measure the performance out of box, please refer to [benchmark](../test/kernels/benchmark/benchmark.md) for more details.
For advanced users, please refer to [profling section](docs/profiling.md).

## API reference for users
### sparse_matmul kernel:
```cpp
#include "interface.hpp"
  ...
  operator_desc op_desc(ker_kind, ker_prop, eng_kind, ts_descs, op_attrs);
  sparse_matmul_desc spmm_desc(op_desc);
  sparse_matmul spmm_kern(spmm_desc);

  std::vector<const void*> rt_data = {data0, data1, data2, data3, data4};
  spmm_kern.execute(rt_data);
```
Refer to corresponding [unit test](../test/gtest/kernels/) for examples.


## Supported Matrix

| Operator                    | Type          | ISA               |
| --------------------------- | ------------- | ----------------- |
| [InnerProduct](docs/kernel_desc/kernel_vnni.md)                | Dense/Sparse  | VNNI, AMX         |
| InnerProduct + [element-wise](docs/kernel_desc/eltwise_injector.md) | Sparse        | VNNI, AMX         |
| InnerProduct + [residual-add](docs/kernel_desc/binaryop_injector.md) | Sparse        | VNNI, AMX         |
| [InnerProduct + layernorm](docs/kernel_desc/kernel_layernormalized_spmm.md)    | Sparse        | VNNI, AMX         |
| SoftMax/GeLU LUT            | NA            | VNNI, AVX512-BF16 |
| MHA                         | Dense         | AMX-INT8          |
| [Transposed MHA](docs/kernel_desc/kernel_transpose_mha.md)              | Sparse        | AMX-INT8, VNNI    |
| Transposed Layernorm        | Sparse        | AVX512F           |
| [Dynamic Quant Matmul](docs/kernel_desc/kernel_dynamic_quant_matmul.md) | Dense | AMX-INT8 |