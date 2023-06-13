[README](/README.md#documentation) > **Terminology**
# Terminology
Intel® XeTLA based on [SYCL](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#glossary) and [ESIMD](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/explicit-simd-sycl-extension.html), some concepts inherited from them.

## Intel® XeTLA:
**boundary check**: Related to surface data access. If the address of the load exceeds the surface boundary, it will return 0; If the address of the store exceeds the surface boundary, it will not update the data in memory.

**brgemm**: The 'br' stands for 'batch-reduce'. For a given `M x K x N` problem, if we split it along `K` dimension with stride `k_stride`, we will have smaller GEMMs, whose `batch_count` is `K / k_stride` . And what brgemm does is looping over all the `batchs`, computing one such smaller GEMM each time and accumulating the intermediate data.

**epilogue**: Some finalizing works after brgemm, it defines how brgemm fused with post-processing and update the final result to memory.

**group API**: A group is a collection of threads that cooperatively work on the same problem. Within this collection, threads can be synchronized and exchang data through dedicated hardware features at a low cost.

**kernel API**: The kernel is a collection of groups together to solve a specific problem.

**SIMD**: Single Instruction Multiple Data

**surface**: The entire data buffer in global memory or local memory.

**SPMD**: Single Program Multiple Data

**thread**: Alias to sub-group, hardware concept.

**tile**: Sub-group level problem. It also involves underneath register storage and operations.

**xetla_exec_item**: The item structure to explict identify the group / local id information. Wrapped SYCL item related operations.


## [SYCL](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#glossary):

**barrier**: A barrier is either a command queue barrier, or a kernel execution group barrier depending on whether it is a synchronization point on the command queue or on a group of work-items in a kernel execution.

**command**: A request to execute work that is submitted to a queue such as the invocation of a SYCL kernel function, the invocation of a host task or an asynchronous copy.

**context**: A context represents the runtime data structures and state required by a SYCL backend API to interact with a group of devices associated with a platform. The context is defined as the sycl::context class.

**device**: A SYCL device is an abstraction of a piece of hardware that can execute SYCL kernels.

**[ESIMD]( https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/explicit-simd-sycl-extension.html)**: Explicit SIMD SYCL Extension, ESIMD provides APIs that are similar to Intel's GPU Instruction Set Architecture (ISA), but it enables you to write explicitly vectorized device code. This explicit enabling gives you more control over the generated code and allows you to depend less on compiler optimizations.

**event**: A SYCL object that represents the status of an operation that is being executed by the SYCL runtime.

**generic memory**: Generic memory is a virtual memory region which can represent global memory, local memory and private memory region.

**global id**: As in OpenCL, a global ID is used to uniquely identify a work-item and is derived from the number of global work items specified when executing a kernel. A global ID is a one, two or three-dimensional value that starts at 0 per dimension.

**group**: A group of work-items within the index space of a SYCL kernel execution, such as a work-group or sub-group.

**host**: Host is the system that executes the C++ application including the SYCL API.

**index space classes**: Like in OpenCL, the kernel execution model defines an nd-range index space. The SYCL runtime class that defines an nd-range is the sycl::nd_range, which takes as input the sizes of global and local work-items, represented using the sycl::range class. The kernel library classes for indexing in the defined nd-range are the following classes:

        • sycl::id : The basic index class representing an id;
        • sycl::item : The item index class that contains the global id and local id;
        • sycl::nd_item : The nd-item index class that contains the global id, local id and the work-group id;
        • sycl::group : The group class that contains the work-group id and the member functions on a work-group.

**kernel**: A kernel represents a SYCL kernel function that has been compiled for a device, including all of the device functions it calls. A kernel is implicitly created when a SYCL kernel function is submitted to a device via a kernel invocation command. However, a kernel can also be created manually by pre-compiling a kernel bundle .

**local id**: A unique identifier of a work-item among other work-items of a work-group.

**local memory**: Local memory is a memory region associated with a work-group and accessible only by work-items in that work-group.

**nd-item**: A unique identifier representing a single work-item and work-group within the index space of a SYCL kernel execution. Can be one, two or three dimensional. In the SYCL interface a nd-item is represented by the nd_item class.

**nd-range**: A representation of the index space of a SYCL kernel execution, the distribution of work items within into work groups. Contains a range specifying the number of global work items, a range specifying the number of local work items and a id specifying the global offset. Can be one, two or three dimensional. The minimum size of range within the nd-range is 0 per dimension; where any dimension is set to zero, the index space in all dimensions will be zero. In the SYCL interface an nd-range is represented by the nd_range class.

**queue**: A SYCL command queue is an object that holds command groups to be executed on a SYCL device. SYCL provides a heterogeneous platform integration using device queue, which is the minimum requirement for a SYCL application to run on a SYCL device.

**range**: A representation of a number of work items or work-group within the index space of a SYCL kernel execution. Can be one, two or three dimensional. In the SYCL interface a work-group is represented by the group.

**sub-group**: The SYCL sub-group (sycl::sub_group class) is a representation of a collection of related work-items within a work-group.

**work-group**: The SYCL work-group (sycl::group class) is a representation of a collection of related work items that execute on a single compute unit. The work items in the group execute the same kernel-instance and share local memory and work-group functions.

**work-item**:The SYCL work-item is a representation of a work-item among a collection of parallel executions of a kernel invoked on a device by a command. A work-item is executed by one or more processing elements as part of a work-group executing on a compute unit. A work-item is distinguished from other work items by its global id or the combination of its work-group id and its local id within a work-group.

# Copyright
Copyright (c) 2022-2023 Intel Corporation Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
