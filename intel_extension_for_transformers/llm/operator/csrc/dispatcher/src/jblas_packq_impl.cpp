#include "jblas/jit_blas_prologue_b.h"
#include "../include/jblas_weightonly_dispatcher.hpp"

namespace woq {
template <class GemmCore, JBLAS_ISA ISA>
void execute_qpack(woq_packq_param* p, woq_packq_ctx* ctx) {
  using proB = jblas::prologue_b::gemm::WeightKBlockNInteger<GemmCore, ISA>;
  static proB ker;
  auto qpackw = ker.createStorage(ctx->n, ctx->k, p->blocksize, wei2jblasdt_map[p->weight_type],
                                  scale2jblasdt_map[p->scale_type], JBLAS_DTYPE::BF16, p->asym);
  if (p->enable_act_shuffle) ker.enableShuffle(&qpackw);
  *(ctx->output) = torch::empty(qpackw.mSize, torch::kInt8);
  qpackw.assign(ctx->output->data_ptr<int8_t>());
  if (p->enable_act_shuffle)
    ker.setShuffleIndices(ctx->g_idx->data_ptr<int>(), &qpackw, &dispatcher_utils::DefaultThreading);
  ker.packQWeight(ctx->n, ctx->k, ctx->qweight->data_ptr<int8_t>(), ctx->n, ctx->scale->data_ptr<float>(),
                  p->asym ? ctx->zp->data_ptr<int8_t>() : nullptr, &qpackw, &dispatcher_utils::DefaultThreading);
}

void jblas_packq(woq_packq_param* p, woq_packq_ctx* ctx) {
  TORCH_CHECK(p->weight_type == "int8" || p->weight_type == "int4_clip" || p->weight_type == "int4_fullrange",
              "Qbits: only support Integer WOQ in PACKQ");

  if (p->compute_type == "int8") {
    if (dispatcher_utils::check_amx() && p->blocksize % jblas::gemm::ICoreRowNAmxint8KBlock<48, 16>::KTILE == 0) {
      return execute_qpack<jblas::gemm::ICoreRowNAmxint8KBlock<48, 16>, JblasAMX_INT8>(p, ctx);
    }
    if (dispatcher_utils::check_avx512_vnni() &&
        p->blocksize % jblas::gemm::ICoreRowNAvx512vnniKBlock<48, 4>::KTILE == 0) {
      return execute_qpack<jblas::gemm::ICoreRowNAvx512vnniKBlock<48, 4>, JblasAVX512_VNNI>(p, ctx);
    }
    if (dispatcher_utils::check_avx_vnni() && p->blocksize % jblas::gemm::ICoreRowNAvxvnniKBlock<48, 2>::KTILE == 0) {
      return execute_qpack<jblas::gemm::ICoreRowNAvxvnniKBlock<48, 2>, JblasAVX_VNNI>(p, ctx);
    }
    TORCH_CHECK(false, "Qbits: Illegal config in int8 compute_type, blocksize:", p->blocksize,
                ", ISA support vnni:", dispatcher_utils::check_avx_vnni());
  }
  if (p->compute_type == "fp32") {
    if (dispatcher_utils::check_avx512f()) {
      return execute_qpack<jblas::gemm::SCoreRowNAvx512f<48, 8>, JblasAVX512F>(p, ctx);
    }
    if (dispatcher_utils::check_avx2()) {
      return execute_qpack<jblas::gemm::SCoreRowNAvx2<48, 2>, JblasAVX2>(p, ctx);
    }
    TORCH_CHECK(false, "Qbits: device ISA must support AVX2 when compute_type==fp32");
  }
  if (p->compute_type == "bf16") {
    if (dispatcher_utils::check_amx()) {
      return execute_qpack<jblas::gemm::HCoreRowNAmxbf16<64, 16>, JblasAMX_BF16>(p, ctx);
    }
    TORCH_CHECK(false, "Qbits: device ISA must support AMX-BF16 when compute_type==bf16");
  }
  TORCH_CHECK(false, "Qbits: unsupported jblas_config, compute_type:", p->compute_type,
              ", weight_type:", p->weight_type + ", blocksize:", p->blocksize);
}
}  // namespace woq
