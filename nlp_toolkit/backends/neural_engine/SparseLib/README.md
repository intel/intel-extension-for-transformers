Just-in-time Deep Neural Network Library (SparseLib)
===========================================

## Abstract

SparseLib is a high-performance operator computing library implemented by assembly. SparseLib contains a JIT domain, a kernel domain, and a scheduling proxy framework.

## Installation
### Build
```shell
cd SparseLib/
mkdir build
cd build
cmake ..
make -j
```

### Test
```shell
cd test/gtest/SparseLib/
mkdir build
cd build
cmake ..
# cmake .. -DSPARSE_LIB_USE_AMX=True // if enabling AMX support
make -j
./test_spmm_vnni_kernel
```

### Performance
We provide a benchmark tool to measure the performance out of box, please refer to [benchmark](../test/SparseLib/benchmark/README.md) for more details.
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
See test_spmm_vnni_kernel.cpp for details.

## Developer guide for developers
* The jit_domain/ directory, containing different JIT assemblies (Derived class of Xbyak::CodeGenerator).
* The kernels/ directory, containing derived classes of different kernels.
* For different kernels: by convention,
  1. xxxx_kd_t is the descriptor of a specific derived primitive/kernel.
  2. xxxx_k_t is a specific derived primitive/kernel.
  3. jit_xxxx_t is JIT assembly implementation of a specific derived primitive/kernel.
  where, "xxxx" represents an algorithm, such as brgemm, GEMM and so on.
* The kernel is determined by the kernel main-kind and kernel isomer. After determining the kernel, it can be implemented by different algorithms. This is the design logic.
