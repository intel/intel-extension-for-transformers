[README](/README.md#Limitations) > **Limitations**
# Limitations

## GEMM
GEMM kernel provides a function `can_implement` to filter the limitations, call it to check the parameters before run GEMM kernel.

- Matrix Mutilple Accumulation (tile_mma)
    - Sub-group tile size on column, `tile_shape::sg_tile_size_x` must be a multiple of 16.
    - Sub-group tile size on row, `tile_shape::sg_tile_size_y` must be a multiple of 8 in matrix-engine-based GEMM.
    - The tile size consumed by each step on the reduction dimension, `k_stride * sizeof(dtype)` must be 32B in vector-engine-based GEMM, multiply of 32B in matrix-engine-based GEMM.
- Global Memory Access
    - Base-address of the matrix must be 64B aligned.
    - Leading-dimension size of the matrix must be multiple of 8B aligned and must be equal or greater than 64B.
    - Width size of the matrix must be 4B aligned and must be equal or greater than 64B.
- Local Memory Access
    - Base-address of the matrix must be 4B aligned.
    - The matrix layout in local memory must be row-major.
    - Loading Matrix A and B works only when the  `mma_engine` is `mma_engine::xmx`.
    - Work-group tile size must be divisible by the sub-group tile size on both row and column dimension.

## Copyright
Copyright (c) 2022-2023 Intel Corporation Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
