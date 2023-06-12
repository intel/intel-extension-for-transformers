[README](/README.md#documentation) > **Programming Guidelines**
# Programming Guidelines

The fundamental concept of Intel® XeTLA revolves around micro-kernels, which are used to compute submatrices (also known as tiles) of output, using advanced GPU instructions like 2D block load/store and DPAS. This approach allows developers to solely focus on their algorithm design, including task division, fusion, and memory heirarchial usage, while offloading the complexity of GEMM computation into template-based building blocks.

There are two groups of API to imeplement GEMM, brgemm (mirco-kernels) in group level and GEMM in kernel level for developers. 

| API level | API name                       |
| :-------- | :----------------------------- |
| kernel    | `gpu::xetla::kernel::gemm_t`   |
| group     | `gpu::xetla::group::brgemm_t`  |


## How To Implement A GEMM With Building Block 

To create a customized GEMM kernel, the following steps should be considered:

1. Define a mirco-kernel, `brgemm`, including the work-group and sub-group division, which is the core of your GEMM
2. Define `epilogue` that specifies what you want to fuse in register level after GEMM computation, such as relu,  and how to write out GEMM results
2. Combine micro-kernel with epilogue together to create a functinal `gemm` implementation

For a runnable code example, you can refer to the code in the [01_basic_gemm](/examples/01_basic_gemm), which also includes explanations of the idea behind the implementation.

### Task Mapping 
Before launching the GPU kernel, it should be decided how to map entire GEMM computation into GPU by work-group and sub-group. To efficient utilize the GPU resource, it's improtant to consider factors such as the shape of the operation, data type, and hardware specifications of the GPU.
```c++
  constexpr uint32_t wg_tile_m = 256;
  constexpr uint32_t wg_tile_n = 256;
  constexpr uint32_t sg_tile_m = 32;
  constexpr uint32_t sg_tile_n = 64;
```
In this example, the input for GEMM is a matrix with dimensions (4096, 4096), and the output matrix has the same dimensions. With the specified work-group and sub-group sizes, we can map the GEMM operation into (16, 16) work-groups, where each work-group has (8, 4) sub-groups respectively. Each sub-group will be executed by a hardware thread. And this logic is defined as below code example, these number is used for `NDRange`.

```c++
//Workload mapping, linear mapping will be used in the code
uint32_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
uint32_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;

//Each subgroup will be executed in one hardware thread
//Calculate how many threads in a work-group
uint32_t local_range_m = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
uint32_t local_range_n = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;

//Ndrange and work-group shape
cl::sycl::range<3> GroupRange {1, group_range_m, group_range_n};
cl::sycl::range<3> LocalRange {1, local_range_m, local_range_n};

cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

//Recommended that you use the helper function to caculate NDRange, it is convenient.
cl::sycl::nd_range<3> get_nd_range(uint32_t matrix_m, uint32_t matrix_n);
```
Now, the GPU kernel is starting from `parallel_for` with specific work-groups and sub-groups.

```c++
cl::sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(matrix_m, matrix_n);
cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
    .....
}
```

### Construct Micro-kernel
The micro-kernel is a crucial component of GEMM, and correctly setting it is essential to its implementation. 
To help developers customize their micro-kernels, the `brgemm_select_t` class provides a simple interface as below.
In this template, the memory layout, computation engine and work-group/sub-gourp shape will be provided and the developer can
decide the location of input and output matrix which is either from global or shared local memory.

```c++
template <typename dtype_a,
          typename dtype_b,
          mem_layout mem_layout_a,
          mem_layout mem_layout_b,
          mem_space mem_space_a,
          mem_space mem_space_b,
          int alignment_a,
          int alignment_b,
          typename dtype_acc,
          typename tile_shape,
          int accum_step,
          mma_engine engine,
          gpu_arch arch>
class brgemm_selector_t {};
```

- `dtype_a` and `dtype_b` are the memory data type of matrix A and B
- `mem_layout_a` and `mem_layout_b` are the memory layout of matrix A and B, can be either `mem_layout::row_major` or `mem_layout::col_major`.
- `mem_space_a` and `mem_space_b` are the memory space of matrix A and B, can be either `mem_space::global` or `mem_layout::local`.
- `alignment_a` and `alignment_b` are the memory alignment of matrix A and B, in unit of element count.
- `dtype_acc` is the accumulate data type of mma compute.
- `tile_shape` is the problem size of each group and subgroup.
- `accum_step` is the size of how many elements will be compuated in the inner loop.
- `engine` is the computing engine: xmx, fpu..
- `arch` is the intel hardware architecture: Xe, Xe2...

### Define Epilogue

The fusion of post-operations, such as `bias add`, `relu`, `gelu`,  after GEMM computation can significantly reduce unnecessary memory transitions and greatly improve performance. In Intel® XeTLA, the `epilogue` is specifically designed to seamlessly integrate post-operations into the GEMM computation at the register level.Beside the fusion, the `epilogue` is also used to update the buffer `c` or data conversion and fusing with some post-processing ops, such as `bias add`, `relu`, `gelu`,.etc.

```c++
template <typename epilogue_policy,
          typename tile_shape,
          typename mem_desc_c>
class epilogue_t {};
```

- `epilogue_policy` tells the epilogue behavior, as well as the related configurations, such as `tile_op_t`, `update_method`, ...
  - `tile_op_t` is the post-processing ops that can be fused together with `brgemm`. When there are multiple post-processing ops, Intel® XeTLA provides `chained_tile_op_t<tile_op_0, tile_op_1, ...>` to fuse all the tile ops first, then feed into `epilogue_t`.
  - `update_method` is the method to update buffer `c`, can be either `result_overwrite` or `result_reduce_sum`.
- `tile_shape` is the problem size of each group and subgroup.
- `mem_desc_c` is the description of buffer `c`, which includes `memory data type`, `memory space` and `memory layout`...


In example [03_gemm_fusion](/examples/03_gemm_fusion), a chain of operations is effectively fused into the GEMM computation. 
First, using pre-defined post-operations `bias_add` and `relu`, and then pass it to `epilogue_policy::tile_op_t`.

```c++
     using tile_op_t = chained_tile_op_t<
                       relu_op_t, // apply elementwise ReLU
                       bias_op_t // apply elementwise BiasAdd
                       >;

```

### Construct GEMM 

After configuration of BRGEMM and epilogue, it's simple to build entire GEMM with:
- assigning tasks to each group, setting working boundaries and starting position accordingly.
- ordering the execution of workgroup within the kernel
- performing any synchronization in between that may be necessary
- performing any necessary group remapping logic to maximize data locality

As below interface, GEMM is constructd by `dispatch_policy`, `brgemm` and `epilogue`.

```c++
template <typename dispatch_policy,
          typename brgemm_t,
          typename epilogue_t>
class gemm_t {};

using gemm_op_t = gpu::xetla::kernel::gemm_t<
            gpu::xetla::kernel::dispatch_policy_default<gpu_arch::Xe>, brgemm_t,
            epilogue_t>;
```

- `dispatch_policy` is the kernel launch attribute, which includes the hardware architecture tag, group remapping information, and special parameters for task splitting, e.g., `l3_kslicing` can be used to split the group-level problem along the `K` dimension in order to get higher occupancy.
- `brgemm_t` is the brgemm operation as describe above.
- `epilogue_t` is the epilogue operation as describe above.

Finally, the actual data will be passed using gemm_op_t::arguments_t, and all of these configurations will be instantiated during the compilation stage for the actual kernel.

```c++
 typename gemm_op_t::arguments_t arg(matrix_n, matrix_k,
                        matrix_m, A, matrix_k, B, matrix_n, C, matrix_n);
```
```c++ 
 gemm_op_t gemm_op;
 xetla_exec_item<3> ei(item);
 gemm_op(ei, arg);
```
## Copyright
Copyright (c) 2022-2023 Intel Corporation Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
