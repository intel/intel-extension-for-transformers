#include "jblas/jit_blas_utils.h"

bool check_amx() { return jblas::utils::parallel::CpuDevice::getInstance()->AMX_BF16(); }
bool check_vnni() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX_VNNI(); }
bool check_avx512f() { return jblas::utils::parallel::CpuDevice::getInstance()->AVX512F(); }