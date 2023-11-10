/*
* Copyright (c) 2020, Intel Corporation
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
* OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/

#include KERNEL_FUNC_FILE

_GENX_MAIN_ void gemm_kernel(DTYPEA *isurface1 [[type("svmptr_t")]],
        DTYPEB *isurface2 [[type("svmptr_t")]],
        DTYPEC *osurface [[type("svmptr_t")]], uint64_t mat_m, uint64_t mat_n,
        uint64_t mat_k, DTYPEACC *acc_surface [[type("svmptr_t")]],
        uint32_t *cnt_surface [[type("svmptr_t")]]) {
    sycl::nd_item<3> item;
    cm_slm_init(SLMSIZE);
    cm_nbarrier_init(BARNUM);
    KERNEL_FUNC_NAME::run(item, isurface1, isurface2, osurface, mat_m, mat_n,
            mat_k, acc_surface, cnt_surface);
}
