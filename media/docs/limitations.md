[README](/README.md#Limitations) > **Limitations**
# Limitations

## GEMM
GEMM kernel provides a function `can_implement` to filter the limitations, call it to check the parameters before run GEMM kernel.

- Matrix Mutilple Accumulation (tile_mma)
    - `tile_shape::sg_tile_size_x` should be a multiple of 16.
    - `tile_shape::sg_tile_size_y` should be a multiple of 8 in matrix-engine-based GEMM.
    - `k_stride * sizeof(dtype)` only can be 32B in vector-engine-based GEMM, 32B or 64B in matrix-engine-based GEMM.
- Global Memory Access
    - Base-address should be 64B aligned.
    - Leading-dimension size of the matrix should be 8B aligned and should be equal or greater than 64B.
    - Width size of the matrix should be 4B aligned and should be equal or greater than 64B.
- Local Memory Access
    - Intel® XeTLA always assumes the matrix layout in local memory is row-major.
    - MatA and MatB can not load from local memory if the mma_engine is `mma_engine::fpu`.
    - No out of boundary check supported. Can not handle group size and local surface size not align.
    - The data access should be 4B aligned (this will be automatically met when following the GEMM MMA limitation).


## Copyright
Copyright (c) 2022-2023 Intel Corporation Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
