[README](/README.md) > **Changelog**
# Changelog

## [v0.3.6](https://github.com/intel/xetla/releases/tag/v0.3.6) (2023-11-10)
- New Features
  * Added GEMM new feature for any shapes support (odd shapes).
  * Provided default configurations for GEMM API (users could get good performance by default configurations, only advanced users need to tune optimization options).
  * Supported converting register layout between tiled and linear.
  * Provided flexible large shape's APIs for other policy (e.g. splitk, improved mat_A & mat_B cache hit ratio).
  * Refined **mem_desc_t** and **payload_t** to expose alignment parameter.
  * Enabled epilogue to support **D = alpha * A * B + beta * C**.
  * Replaced **xetla_exec_item** with **sycl::nd_item**.
  * Refined some examples to invoke kernel level APIs, added fence and barrier to MLP example.
  * Fixed some known issues, enhanced tests, and updated documents.

- Known Issues
    - Refer to [Limitations](/media/docs/limitations.md).

## [v0.3.5](https://github.com/intel/xetla/releases/tag/v0.3.5) (2023-09-28)
- New Features
  * Enhanced limitation checking.
  * Refined GEMM APIs’ name.
  * Supported GEMM APIs load B from SLM.
  * Supported GEMM of any (odd) shapes.
  * Supported Streaming-K.
  * Enhanced L3 K-slicing support.
  * Improved GEMM's performance for large-N (M, N, K) shapes.
  * Fixed tile load/store bugs.
  * Enhanced examples, tests, and updated documents.

- Known Issues
    - Refer to [Limitations](/media/docs/limitations.md).

## [v0.3.4](https://github.com/intel/xetla/releases/tag/v0.3.4) (2023-08-18)
- New Features
  * Enabled limitation checking.
  * Provided "XETLA_PRINTF" and "XETLA_ASSERT" for debugging.
  * Refined fpu based GEMM.
  * Refined tile reduce APIs, deprecated API "tile_row_reduce".
  * Supported new data type int4 (experimental feature).
  * Fixed tile load/store bugs.
  * Enhanced examples, tests, and updated documents.

- Known Issues
    - Refer to [Limitations](/media/docs/limitations.md).

 ## [v0.3.3](https://github.com/intel/xetla/releases/tag/v0.3.3) (2023-06-30)
- New Features
  * Enabled debug build support.
  * Updated documents, added some diagrams and details.
  * Fixed some customer reported issues.
  * Improved the project's quality.

- Known Issues
    - Refer to [Limitations](/media/docs/limitations.md).

## [v0.3.2](https://github.com/intel/xetla/releases/tag/v0.3.2) (2023-06-16)
- New Features
  * Added some kernel-level APIs' parameters check functions, users need to explicit call them before launch the kernel; will return fail and print error messages when detect unsupported scenarios, continue launching the kernel for unspported scenarios may lead to unpredictable result.
  * Removed reduce_sum + tile_op epilogue policy.
  * Added some unit test cases.
  * Refined some examples code.
  * Updated documents, added some diagrams and details.
  * Fixed some customer reported issues.
  * Improved the project's quality.

- Known Issues
    - Refer to [Limitations](/media/docs/limitations.md).

## [v0.3.1](https://github.com/intel/xetla/releases/tag/v0.3.1) (2023-05-22)
- New Features
  * Initial open source release.
  * Improved subgroup level fundamental features: core, tile, utils.
  * Enhanced basic BRGEMM micro-kernels for different data types as well as different epilogue flavours.
  * Unified the examples: basic BRGEMM, GEMM; fusion BRGEMM, GEMM; batched GEMM; MLP, GRU, MHA.
  * Added some unit test cases.

- Known Issues
    - Refer to [Limitations](/media/docs/limitations.md).

## v0.3.0 (2023-04-23)
- New Features
  * Initial internal release.
  * Subgroup level fundamental features: core, tile, utils.
  * Basic BRGEMM micro-kernels for different data types as well as different epilogue flavours.
  * Examples
    * Basic BRGEMM, GEMM; fusion BRGEMM, GEMM; batched GEMM; MLP, GRU, MHA.  
  * Unit tests.
  
- Known Issues
    - Refer to [Limitations](/media/docs/limitations.md).

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

