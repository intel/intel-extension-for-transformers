[README](/README.md#documentation) > **Programming Guidelines**

![ALT](/media/docs/workflow.png "Step by step GEMM decomposition")

# Basic Concepts

The central idea behind Intel® XeTLA revolves around the concept of `building blocks`, which used to create larger and more complex kernels. These building blocks consist of highly performant device code that harnesses advanced GPU instructions such as 2D block load/store and DPAS. Furthermore, it means the most intricacies of computation and data is offloaded into these essential building blocks. XeTLA empowers developers to concentrate exclusively on their algorithm design, encompassing task allocation, fusion, and memory hierarchy utilization. 

There are there groups of API for user, each serving with different purposes. 
- [kernel-level API](https://github.com/pengzhao-intel/xetla/tree/main/include/kernel) is designed for the easiest user experience by combining various `group-level API`. For instance, `gemm_universal` is specifically tailored for GEMM (General Matrix Multiply), where users only need to set the input shapes of A, B, C, a few basic parameters and post functions,  without delving into the intricacies of computation. Of course, developers have the option to customize their own GEMM implementations using the `group-level API`, potentially achieving better performance for their specific input shapes.
- [group-level API](https://github.com/pengzhao-intel/xetla/tree/main/include/group) serves as the primary component for building your own kernels. These group functions are mapped to `workgroup` and executed in the Xe core on the GPU. Therefore, it's crucial to understand how to divide the workload into smaller pieces and allocate it to the workgroups. One major performance concern is having too few workgroups to fully utilize all available DSS resources on the GPU.
- [subgroup-level API](https://github.com/pengzhao-intel/xetla/tree/main/include/subgroup) represents the next lower level of group API. In most cases, creating high-performance kernels can be achieved using the `group-level API`. However, for developers who seek finer control over algorithm details, such as when to perform data prefetch or manage data reuse within a workgroup, the `subgroup-level API` offers the utmost flexibility.

| API level | Example                                  |
| :-------- | :----------------------------------------|
| kernel    | `gpu::xetla::kernel::gemm_universal`     |
| group     | `gpu::xetla::group::gemm`                |
| subgroup  | `gpu::xetla::subgroup::tile_prefetch`    |  


## Kernel-level API 
The `kernel-level API` operates at the GPU-wide scale, where both input and output rely on global memory. Local shared memory and synchronization are handled internally within workgroups, transparent to the developer. Consequently, developers remain unaware of these low-level details.

For instance, consider the `gemm_universal` function. When using this API, developers are required to make choices regarding dispatch policies, select the appropriate GEMM building block, and specify any post-processing operators. The API example is outlined below:

```c++
using gemm_op_t = xetla::kernel::gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>;
```
And then this GEMM can be executed inside `parallel_for`.
```c++
auto gpu_event = queue.submit([&](handler &cgh) {
    // GPU kernel
    cgh.parallel_for(nd_range, [=](nd_item<3> item) KERNEL_MAIN {
        
        // allocate slm and nbarrier resource
        slm_barrier_init<gemm_op_t>();
        gemm_op_t gemm_op;
        gemm_op(item, gemm_arg);
    });
});
```
For a runnable code example, you can refer to the code in [01_gemm_universal](/examples/01_gemm_universal), which also includes explanations of the idea behind the implementation.

## Group-level API 
The use of a `group-level API` in parallel computing provides several notable advantages. Firstly, it offers developers greater flexibility in constructing custom kernels tailored to their specific needs. This flexibility extends to workload distribution across GPU workgroups. In this context, the allocation of workgroups is based on the output matrix C, with each workgroup handling a distinct sub-matrix sized `wg_tile_m` * `wg_tile_n`. Within each workgroup, intricate computations related to the `K` dimension are encapsulated within the GEMM building block, sparing developers from delving into these details at the group level

![ALT](/media/docs/code_map.jpg "Code Example to show workload mapping")

Moreover, a key benefit of the group-level API is the empowerment it grants developers over accumulator variables (`matAcc` in below example). This control enables developers to implement more sophisticated and innovative operations, seamlessly fused with the GEMM computation. This level of customization proves invaluable when striving for optimized performance tailored to specific computational tasks, such as example in [02_basic_gemm](/examples/02_basic_gemm)

```c++
gemm_t::matAcc_t matAcc;
matAcc.init(0);

gemm_t::arguments_t gemm_args(md_a, md_b, inner_loop_count);

// the results is in the matAcc rather than real output C
gemm_t::work_group_t g(item.get_local_linear_id());
gemm(g, matAcc, gemm_args);

// any customized operation here based on matACC

// write the results from matACC to real output C
epilogue_t epilogue;
epilogue(g, matAcc, md_c);
```

### Subgroup-level API
The subgroup represents the lowest level within the entire XeTLA architecture, offering developers a high degree of proximity to the hardware and consequently, finer control over various aspects. Developers have the capability to finely tune operations at this level, including decisions on data loading and storing methods, specifying load and store types, among others.

At the subgroup level, developers are responsible for tasks such as allocating shared local memory, managing synchronization mechanisms, and designing the utilization of registers, among other responsibilities. An illustrative example can be found in the [08_scaled_dot_product_attention](/examples/08_scaled_dot_product_attention). 

The subgroup-level API includes: 
- Tile load/store
- Data Prefetch
- Computation MMA and FMA
- Pre/Post Operators (Epilogue)
- Data transform


## The Key Things for Better Performance
Intel® XeTLA provides the basic building block of GEMM unit; however, it still needs to implement the kernel carefully for the better perforamnce in both algorithm and hardware level.
1. Number of group / subgroup
2. K slicing algorithm
3. Reuse register for post operations
4. Data sharing through shared local memory
5. Reduction

## Copyright
Copyright (c) 2022-2023 Intel Corporation Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
