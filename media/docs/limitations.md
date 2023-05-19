## Limitations

- GEMM MMA:
    - `sg_tile_n` should be a multiple of 16.
    - `sg_tile_m` should be a multiple of 8 in matrix-engine-based GEMM.
    - `accum_step * sizeof(dtype)` only can be 32B in vector-engine-based GEMM, 32B or 64B in matrix-engine-based GEMM.
- GEMM Global Memory Access
    - Base-address should be QW (8B) aligned.
    - Leading-dimension size of the matrix should be QW (8B) aligned and should be equal or greater than 64B.
    - Width size of the matrix should be DW (4B) aligned and should be equal or greater than 64B.
    - The above hardware limitation can be found in this [link](https://gfxspecs.intel.com/Predator/Home/Index/53567).
    - No out of boundary check supported if the `update_method` is `result_reduce_sum`.
- GEMM Local Memory Access
    - Intel® XeTLA always assumes the matrix layout in local memory is row-major.
    - MatA and MatB can not load from local memory if the mma_engine is `mma_engine::fpu`.
    - No out of boundary check supported.
    - The data access should be DW (4B) aligned (this will be automatically met when following the GEMM MMA limitation).
- LayerNorm
    - `sg_tile_n * sizeof(dtype)` should be DW (4B) aligned.
    - No out of boundary check supported.
