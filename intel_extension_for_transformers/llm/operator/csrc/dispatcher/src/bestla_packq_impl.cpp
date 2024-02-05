#include "bestla/bestla_prologue_b.h"
#include "../include/bestla_packq_impl.hpp"

namespace woq {
template <class GemmCore, BTLA_ISA ISA>
void execute_qpack(woq_packq_param* p, woq_packq_ctx* ctx) {
  using proB = bestla::prologue_b::gemm::WeightKBlockNInteger<GemmCore, ISA>;
  static proB ker;
  auto qpackw = ker.createStorage(ctx->n, ctx->k, p->blocksize, wei2bestladt_map[p->weight_type],
                                  scale2bestladt_map[p->scale_type], BTLA_DTYPE::BF16, p->asym);
  if (p->enable_act_shuffle) ker.enableShuffle(&qpackw);
  *(ctx->output) = torch::empty(qpackw.mSize, torch::kInt8);
  qpackw.assign(ctx->output->data_ptr<int8_t>());
  if (p->enable_act_shuffle)
    ker.setShuffleIndices(ctx->g_idx->data_ptr<int>(), &qpackw, &dispatcher_utils::DefaultThreading);
  ker.packQWeight(ctx->n, ctx->k, ctx->qweight->data_ptr<int8_t>(), ctx->n, ctx->scale->data_ptr<float>(),
                  p->asym ? ctx->zp->data_ptr<int8_t>() : nullptr, &qpackw, &dispatcher_utils::DefaultThreading);
}

std::string get_dtype_str(BTLA_DTYPE dtype) {
  switch (dtype) {
    case BTLA_DTYPE::F32:
      return "fp32";
    case BTLA_DTYPE::BF16:
      return "bf16";
    case BTLA_DTYPE::S4_CLIP:
      return "int4_clip";
    case BTLA_DTYPE::S4_FULLRANGE:
      return "int4_fullrange";
    case BTLA_DTYPE::F4_NF4:
      return "nf4";
    case BTLA_DTYPE::F4_E2M1:
      return "fp4_e2m1";
    case BTLA_DTYPE::F4_BNB:
      return "fp4_e2m1_bnb";
    case BTLA_DTYPE::S8:
      return "int8";
    case BTLA_DTYPE::F8_E5M2:
      return "fp8_e5m2";
    case BTLA_DTYPE::F8_E4M3:
      return "fp8_e4m3";
    case BTLA_DTYPE::F8_E8M0:
      return "fp8_e8m0";
    default:
      TORCH_CHECK(false, "QBits: unrecognized data type.")
      break;
  }
}

std::string get_cmpt_str(bestla::gemm::CompType cmpt) {
  using bestla::gemm::CompType;
  switch (cmpt) {
    case CompType::COMP_INT8_US_INT32:
    case CompType::COMP_INT8_US_FP32:
      return "int8";
    case CompType::COMP_FP32:
      return "fp32";
    case CompType::COMP_BF16_FP32:
      return "bf16";
    default:
      TORCH_CHECK(false, "QBits: unrecognized compute type.");
      break;
  }
}

std::vector<int> get_ascii_vec(std::string str) {
  std::vector<int32_t> ret;
  for (char c : str) ret.push_back(static_cast<int32_t>(c));
  return ret;
}

torch::Tensor get_packw_info(torch::Tensor& packw, PACKW_ACQUIRE_TYPE ACQ_T) {
  torch::Tensor output;
  auto packw_ptr = dynamic_cast<bestla::storage::gemm::StorageWeightKBlockNInteger*>(
      bestla::storage::gemm::PackedWeightParser::deserialBuffer(packw.data_ptr()));
  output = torch::empty(1, torch::kInt64);
  switch (ACQ_T) {
    case SIZE:
      return output.index_put_({0}, static_cast<int64_t>(packw_ptr->mSize));
    case BLOCKSIZE:
      return output.index_put_({0}, static_cast<int64_t>(packw_ptr->mBlockSize));
    case K:
      return output.index_put_({0}, static_cast<int64_t>(packw_ptr->mK));
    case N:
      return output.index_put_({0}, static_cast<int64_t>(packw_ptr->mN));
    case ACT_SHUFFLE:
      return output.index_put_({0}, static_cast<int64_t>(packw_ptr->ShfIndice() != nullptr ? 1 : 0));
    case G_IDX: {
      auto tensor_size = packw_ptr->mShuffleIndices.size<int>();
      TORCH_CHECK(packw_ptr->ShfIndice() != nullptr, "QBits: not pack g_idx tensor.");
      output = torch::empty(tensor_size, torch::kInt32);
      memcpy(output.data_ptr(), packw_ptr->ShfIndice(), tensor_size * sizeof(int));
    } break;
    case WEI_TYPE:
    case SCALE_TYPE: {
      BTLA_DTYPE acquire_dt = ACQ_T == WEI_TYPE ? packw_ptr->mDType : packw_ptr->SDtype();
      auto ascii_vec = get_ascii_vec(get_dtype_str(acquire_dt));
      output = torch::empty(ascii_vec.size(), torch::kInt32);
      memcpy(output.data_ptr(), ascii_vec.data(), ascii_vec.size() * sizeof(int));
    } break;
    case CMPT_TYPE: {
      auto CType = bestla::gemm::CoreAttr::get_mask_val(packw_ptr->mCoreId, bestla::gemm::CoreAttr::COMP_MASK,
                                                        bestla::gemm::CoreAttr::COMP_SHIFT);
      auto ascii_vec = get_ascii_vec(get_cmpt_str(static_cast<bestla::gemm::CompType>(CType)));
      output = torch::empty(ascii_vec.size(), torch::kInt32);
      memcpy(output.data_ptr(), ascii_vec.data(), ascii_vec.size() * sizeof(int));
    } break;
    default:
      TORCH_CHECK(false, "QBits: unsupported acquire_type");
      break;
  }
  return output;
}

void bestla_packq(woq_packq_param* p, woq_packq_ctx* ctx) {
  TORCH_CHECK(p->weight_type == "int8" || p->weight_type == "int4_clip" || p->weight_type == "int4_fullrange",
              "Qbits: only support Integer WOQ in PACKQ");

  if (p->compute_type == "int8") {
    if (dispatcher_utils::check_amx() && p->blocksize % bestla::gemm::ICoreRowNAmxint8KBlock<48, 16>::KTILE == 0) {
      return execute_qpack<bestla::gemm::ICoreRowNAmxint8KBlock<48, 16>, BTLA_ISA::AMX_INT8>(p, ctx);
    }
    if (dispatcher_utils::check_avx512_vnni() &&
        p->blocksize % bestla::gemm::ICoreRowNAvx512vnniKBlock<48, 4>::KTILE == 0) {
      return execute_qpack<bestla::gemm::ICoreRowNAvx512vnniKBlock<48, 4>, BTLA_ISA::AVX512_VNNI>(p, ctx);
    }
    if (dispatcher_utils::check_avx_vnni() && p->blocksize % bestla::gemm::ICoreRowNAvxvnniKBlock<48, 2>::KTILE == 0) {
      return execute_qpack<bestla::gemm::ICoreRowNAvxvnniKBlock<48, 2>, BTLA_ISA::AVX_VNNI>(p, ctx);
    }
    TORCH_CHECK(false, "Qbits: Illegal config in int8 compute_type, blocksize:", p->blocksize,
                ", ISA support vnni:", dispatcher_utils::check_avx_vnni());
  }
  if (p->compute_type == "fp32") {
    if (dispatcher_utils::check_avx512f()) {
      return execute_qpack<bestla::gemm::SCoreRowNAvx512f<48, 8>, BTLA_ISA::AVX512F>(p, ctx);
    }
    if (dispatcher_utils::check_avx2()) {
      return execute_qpack<bestla::gemm::SCoreRowNAvx2<48, 2>, BTLA_ISA::AVX2>(p, ctx);
    }
    TORCH_CHECK(false, "Qbits: device ISA must support BTLA_ISA::AVX2 when compute_type==fp32");
  }
  if (p->compute_type == "bf16") {
    if (dispatcher_utils::check_amx()) {
      return execute_qpack<bestla::gemm::HCoreRowNAmxbf16<64, 16>, BTLA_ISA::AMX_BF16>(p, ctx);
    }
    TORCH_CHECK(false, "Qbits: device ISA must support AMX-BF16 when compute_type==bf16");
  }
  TORCH_CHECK(false, "Qbits: unsupported bestla_config, compute_type:", p->compute_type,
              ", weight_type:", p->weight_type + ", blocksize:", p->blocksize);
}
}  // namespace woq
