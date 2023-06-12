[README](/README.md) > **Changelog**
# Changelog

## [v0.3.2](https://github.com/intel/xetla/releases/tag/v0.3.2) (2023-06-16)
- New Features
  * Added some kernel-level APIs' parameters check functions, users need to explicit call them before launch the kernel; will return fail and print error messages when detect unsupported scenarios, continue launching the kernel for unspported scenarios may lead to unpredictable result.
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
  * Improved sub-group level fundamental features: core, tile, utils.
  * Enhanced basic BRGEMM micro-kernels for different data types as well as different epilogue flavours.
  * Unified the examples: basic BRGEMM, GEMM; fusion BRGEMM, GEMM; batched GEMM; MLP, GRU, MHA.
  * Added some unit test cases.

- Known Issues
    - Refer to [Limitations](/media/docs/limitations.md).

## v0.3.0 (2023-04-23)
- New Features
  * Initial internal release.
  * Sub-group level fundamental features: core, tile, utils.
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

