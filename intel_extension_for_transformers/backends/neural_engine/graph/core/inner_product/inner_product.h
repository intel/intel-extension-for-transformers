#ifndef NE_GRAPH_INNER_PRODUCT_H
#define NE_GRAPH_INNER_PRODUCT_H

#ifdef __cplusplus
extern "C" {
#endif
void jblas_weights4block_f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda,
                                     int ldo);

void jblas_timer(bool _init);

int jblas_set_threads(int _nth);

#ifdef __cplusplus
}
#endif
#endif
