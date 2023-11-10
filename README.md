# Intel® Xe Templates for Linear Algebra

_Intel® XeTLA [v0.3.6](/CHANGELOG.md) - November 2023_

Intel® Xe Templates for Linear Algebra (Intel® XeTLA) is a collection of SYCL/ESIMD templates that enable high-performance General Matrix Multiply (GEMM), Convolution (CONV), and related computations on Intel Xe GPU architecture. Intel® XeTLA offers reusable C++ templates for kernel, group and subgroup levels, allowing developers to optimize and specialize kernels based on data types, tiling policies, algorithms, fusion policies, and more.

One of the key features of Intel® XeTLA is its ability to abstract and hide details of Xe hardware implementations, particularly those related to matrix computations, such as the systolic array and other low level instructions. This ensures that SYCL/DPC++ developers can focus on leveraging the performance benefits of Intel® XeTLA without being burdened by hardware-specific instructions.

<!-- @cond -->
## Compatibility

  |Category|Requirement|Installation|
  |-|-|-|
  |OS|Ubuntu [22.04](http://releases.ubuntu.com/22.04/)| [Install Ubuntu](https://ubuntu.com/tutorials)|
  |GPU Card | Intel® Data Center GPU Max Series |N/A|
  |GPU Driver | [Stable 647](https://dgpu-docs.intel.com/releases/stable_647_21_20230714.html) or later|[Install Intel GPU driver](https://dgpu-docs.intel.com/installation-guides/index.html#intel-data-center-gpu-max-series)|
  |Toolchain |Intel® oneAPI Base Toolkit [2023.2](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) or later|[Install Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)|

<!-- @endcond -->

## Features

- GEMM
  - Data Type
    - Vector-engine-based: `fp32`
    - Matrix-engine-based: `tf32`, `fp16`, `bf16`, `int8`
  - Memory Layout
    - Matrix A: `row-major`, `col-major`
    - Matrix B: `row-major`, `col-major`
    - Matrix C: `row-major`
- Epilogue
  - Bias Add
  - GELU Forward
  - GELU Backward
  - RELU
  - Residual Add

<!-- @cond -->

## Documentation

- [Quick Start](/media/docs/quick_start.md) introduces how to build and run tests/examples.
- [Functionality](/media/docs/functionality.md) describes kernel-level API feature list.
- [API Reference](https://intel.github.io/xetla) provides a comprehensive reference of the library APIs.
- [Programming Guidelines](/media/docs/programming_guidelines.md) explains programming model, functionalities, implementation details, and annotated examples.
- [Construct a High Performance GEMM](/media/docs/construct_a_gemm.md) describes how to construct a high performance GEMM.
- [Terminology](/media/docs/terminology.md) describes terms used in the project.
- [Changelog](/CHANGELOG.md) detailed listing of releases and updates.
 
## Project Structure

```
include/                       # Definitions of Intel® XeTLA APIs
    common/                    #    - Low level APIs that wrap the same functionality APIs from ESIMD
    experimental/              #    - Experimental features
    group/                     #    - Group level APIs 
    kernel/                    #    - Kernel level APIs
    subgroup/                  #    - Subgroup level APIs
    xetla.hpp                  #    - Unified and unique external head file

tests/                         # Tests to verify correctness of Intel® XeTLA APIs
    integration/               #    - Integration testes
    unit/                      #    - Unit tests
    utils/                     #    - Utils implement of unit and integration tests

examples/                      # Examples of Intel® XeTLA basic/fused kernels

tools/                         # Tools for code format, build environment...

media/                         # Documents
```

## Contributing

Refer to [Contributing Guidelines](/CONTRIBUTING.md).


## Limitations

Refer to [Limitations](/media/docs/limitations.md).

<!-- @endcond -->

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](/SECURITY.md)

## Copyright

Copyright (c) 2022-2023 Intel Corporation
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

