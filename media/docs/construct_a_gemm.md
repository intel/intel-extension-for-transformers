# Construct a High Performance GEMM

In this document, we will demonstrate how to construct a General Matrix Multiply (GEMM) operation using the XeTLA API, both at the kernel and workgroup levels. Additionally, we will explore the relationship between the GEMM shape and other relevant parameters, as well as when to employ the `splitK` or `streamK` algorithms.

As shown in the diagram below, each workgroup will calculate a sub-matrix, represented by the blue box in output C. Subsequently, this sub-matrix will be continuously divided into multiple tiles, with dimensions `sg_tile_n` by `sg_tile_m`. These tiles will then be assigned to subgroups. Finally, these tile operations will be mapped to the actual hardware instructions, such as `2d load` and `mma`.

![ALT](/media/docs/dom.jpg "GEMM decomposition by workgroup and subgroup")

## Basic Components  

1. Select a `GEMM building block`, considering the division of work-group and sub-group
2. Decide if `splitK` or `steamK` is needed in specific shape 
3. Define `epilogue` that specifies what you want to fuse after the GEMM computation based on accumulator
4. Instantiate a `gemm` implementation by the selections from 1)-3).

For a runnable code example, you can refer to the code in the [02_basic_gemm](/examples/02_basic_gemm).

### Task Mapping 
Before launching the GPU kernel, it is crucial to determine how to map the entire GEMM computation onto the GPU, considering work-group and sub-group configurations. Efficiently utilizing GPU resources requires careful consideration of factors such as the operation's shape, data type, and the hardware specifications of the GPU. A typical configuration for workgroups and subgroups may resemble the example below, especially when the input shape is sufficient to fully utilize the GPU.

```c++
constexpr uint32_t wg_tile_m = 256;
constexpr uint32_t wg_tile_n = 256;
constexpr uint32_t sg_tile_m = 32;
constexpr uint32_t sg_tile_n = 64;
```
In this example, the input for the GEMM operation is a matrix with dimensions (4096, 4096), and the output matrix has the same dimensions. With the specified work-group and sub-group sizes, we can organize the GEMM operation into (16, 16) work-groups, each containing (8, 4) sub-groups, with each sub-group being executed by a hardware thread.

However, if we consider a scenario where the input dimensions are (32, 1024), the current workgroup and subgroup sizes would result in work-groups that are too large to create a sufficient number of them. In this case, it becomes necessary to adjust the size of the workgroup and subgroup to achieve efficient computation.

### SplitK
This situation is quite common in AI workloads, where the matrix is rectangular, meaning that the M and N dimensions are relatively small, while the K dimension is . For instance, consider a workload with dimensions (256, 256, 8192), resulting in an output shape of C as (256, 256). If we were to use a workgroup shape of (256, 256), only one workgroup would be created, which is far from sufficient for efficient GPU utilization.

Even if we reduce the workgroup size to (64, 64), we would still have only 16 workgroups on the GPU. Further reducing the workgroup size introduces other challenges, including poor memory locality and difficulty in hiding latency. The example code below demonstrates this mapping alogrithm:

```c++
//Workload mapping, linear mapping will be used in the code
uint32_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
uint32_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;

//Each subgroup will be executed in one hardware thread
//Calculate how many threads in a work-group
uint32_t local_range_m = (wg_tile_m + sg_tile_m - 1) / sg_tile_m;
uint32_t local_range_n = (wg_tile_n + sg_tile_n - 1) / sg_tile_n;

//nd_range and work-group shape
cl::sycl::range<3> group_range {1, group_range_m, group_range_n};
cl::sycl::range<3> local_range {1, local_range_m, local_range_n};

cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

//Recommended that you use the helper function to caculate nd_range, it is convenient.
cl::sycl::nd_range<3> get_nd_range(uint32_t matrix_m, uint32_t matrix_n);
```
In this algorithm, the number of workgroups is primarily determined by the workgroup tile size. However, this can become problematic if the output shape is not sufficiently large. On the other hand, considering the large K dimension, we can split it to create more workgroups. As illustrated in the code snippet above, the first parameter of `group_range` is set to 1. By splitting the K dimension into 4, the total number of workgroups increases fourfold, as shown in the accompanying diagram. This approach is commonly referred to as splitK.

In this diagram, the K splitting occurs at the workgroup level, meaning that each workgroup calculates only a portion of the final GEMM results. Subsequently, these partial results must be accumulated together. Notably, there is no explicit synchronization mechanism between workgroups, necessitating the use of `atomic_add` to perform the accumulation, as indicated by the `Cross-Workgroup Reduction` section in the diagram.

It's important to note that this method has limitations. Since `atomic_add` supports only float addition, the output datatype must be float, rather than float16 or bfloat16.

![ALT](/media/docs/workgroup_splitK.jpg "split K in workgroup level")

Alternatively, the subgroup-level splitK is also available i which can accumulate the result during shared local memory inside a workgroup so the half percesion data type is still supported.

![ALT](/media/docs/subgroup_splitK.jpg "split K in subgroup level")

For kernel level API, we can set two parameters in dispatch policy of `gemm_universal` API. Definitely, you can set both value to large than 1 for mixing workgroup and subgroup level split K together. 

```c++
 using dispatch_policy
            = gpu::xetla::kernel::dispatch_policy_kslicing<num_global_splitk, num_local_splitk, gpu_arch::Xe>;
```
For group level API, the developer can leverage `group::cooperative_reduce_t` to add the final results by themselves.

### Configuraiton for GEMM building block
The building block is a crucial component of GEMM, the `gemm_selector_t` class provides a simple interface as below.
In this template, the memory layout, computation engine and work-group/sub-gourp shape will be provided and the developer can
decide the location of input and output matrix which is either from global or shared local memory.

```c++
    using gemm_t = typename xetla::group::gemm_selector_t<
            data_type_a, // input datatype for A
            data_type_b, // input datatype for B
            mem_layout::row_major, // memory layout for A
            mem_layout::row_major, // memory layout for B
            mem_space::global, // memory reading from global mem for A
            mem_space::global, // memory reading from global mem for B
            8, // buffer alignment for A, in unit of element
            8, // buffer alignment for B, in unit of element
            data_type_acc, // accumulator data type for intermediate resutls
            tile_shape, // computation tile shape
            sg_tile_k, // elements in each iteration
            mma_engine::xmx, // compute engine
            gpu_arch::Xe, // GPU arch
            stages, // number of prefetch pipe stage
            sync_freq> // frequency of periodic sync, in unit of inner loop
            ::gemm;
```

- `dtype_a` and `dtype_b` are the memory data type of matrix A and B
- `mem_layout_a` and `mem_layout_b` are the memory layout of matrix A and B, can be either `mem_layout::row_major` or `mem_layout::col_major`.
- `mem_space_a` and `mem_space_b` are the memory space of matrix A and B, can be either `mem_space::global` or `mem_layout::local`.
- `alignment_a` and `alignment_b` are the memory alignment of matrix A and B, in unit of element count.
- `dtype_acc` is the accumulate data type of mma compute.
- `tile_shape` is the problem size of each group and subgroup.
- `k_stride` is the size of how many elements will be compuated in the inner loop.
- `engine` is the computing engine: xmx, fpu..
- `arch` is the intel hardware architecture: Xe, Xe2...

### Define Epilogue

The fusion of post-operations, such as `bias add`, `relu`, `gelu`,  after GEMM computation can significantly reduce unnecessary memory transitions and greatly improve performance. In Intel® XeTLA, the `epilogue` is specifically designed to seamlessly integrate post-operations into the GEMM computation at the register level. Beside the fusion, the `epilogue` is also used to update the buffer `c` or data conversion and fusing with some post-processing ops, such as `bias add`, `relu`, `gelu`,.etc.

```c++
  using epilogue_t = xetla::group::epilogue_t<
            xetla::group::epilogue_policy_default<gpu_arch::Xe>, tile_shape,
            mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>>;
class epilogue_t {};
```

- `epilogue_policy_default` tells the epilogue behavior, as well as the related configurations, such as `tile_op_t`, `update_method`, ...
- `tile_shape` is the problem size of each group and subgroup.
- `mem_desc_c` is the description of buffer `c`, which includes `memory data type`, `memory space` and `memory layout`...

In example [03_gemm_relu_bias](/examples/03_gemm_relu_bias), a chain of operations is effectively fused into the GEMM computation. 
First, using pre-defined post-operations `relu` and `bias_add`, and then pass it to `epilogue_policy::tile_op_t`.

```c++
using tile_op_t = chained_tile_op_t<
                  relu_op_t, // apply elementwise ReLU
                  bias_op_t // apply elementwise BiasAdd
                  >;
```

### GEMM Instantiate 

After configuration of BRGEMM and epilogue, it's simple to build entire GEMM with:
- assigning tasks to each group, setting working boundaries and starting position accordingly.
- ordering the execution of work-group within the kernel
- performing any synchronization in between that may be necessary
- performing any necessary group remapping logic to maximize data locality

As below interface, GEMM is constructd by `dispatch_policy`, `gemm` and `epilogue`.

```c++
using gemm_op_t = xetla::kernel::gemm_universal_t<dispatch_policy, gemm_t,
            epilogue_t>;
```

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
