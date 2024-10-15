// Copyright (c) 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bestla/bestla_prologue_b.h"
#include "../include/bestla_packq_impl.hpp"

namespace woq {

template <class proB>
void execute_qpack(repack_quantized_weight_param* p, repack_quantized_weight_ctx* ctx, WOQ_TASK task) {
  static proB ker;
  using WType = typename proB::StorageWeight;
  WType qpackw(0);
  if constexpr (std::is_same_v<WType, bestla::storage::gemm::StorageWeightKBlockNInteger>) {
    qpackw = ker.createStorage(ctx->n, ctx->k, p->blocksize, wei2bestladt_map.at(p->weight_type),
                               scale2bestladt_map.at(p->scale_type), BTLA_DTYPE::BF16, p->asym);
  } else {
    qpackw = ker.createStorage(ctx->n, ctx->k, p->blocksize, wei2bestladt_map.at(p->weight_type),
                               scale2bestladt_map.at(p->scale_type));
  }
  if (p->enable_act_shuffle) ker.enableShuffle(&qpackw);
  ctx->packw_size = qpackw.mSize;
  if (task == WOQ_GET_PACKW_SIZE) return;
  *(ctx->output) = torch::empty(qpackw.mSize, torch::kInt8);
  qpackw.assign(ctx->output->data_ptr<int8_t>());
  if (p->enable_act_shuffle)
    ker.setShuffleIndices(ctx->g_idx->data_ptr<int>(), &qpackw, dispatcher_utils::qbits_threading::get());
  ker.packQWeight(ctx->n, ctx->k, ctx->qweight->data_ptr<int8_t>(), ctx->n, ctx->scale->data_ptr<float>(),
                  p->asym ? ctx->zp->data_ptr<int8_t>() : nullptr, &qpackw, dispatcher_utils::qbits_threading::get());
}

template <class GemmCore, BTLA_ISA ISA>
void parse_prob(repack_quantized_weight_param* p, repack_quantized_weight_ctx* ctx, WOQ_TASK task) {
  if (p->weight_type == "int8" || p->weight_type == "int4" || p->weight_type == "int3" || p->weight_type == "int2") {
    return execute_qpack<bestla::prologue_b::gemm::WeightKBlockNInteger<GemmCore, ISA>>(p, ctx, task);
  }
  if (p->weight_type == "nf4" || p->weight_type == "fp4_e2m1_bnb" || p->weight_type == "fp4_e2m1") {
    TORCH_CHECK(!p->asym, "Qbits: float-weight unsupports asym quantization.");
    return execute_qpack<bestla::prologue_b::gemm::WeightKBlockNFloat<GemmCore, ISA>>(p, ctx, task);
  }
  TORCH_CHECK(false, "Qbits: unsupported bestla packq config, compute_type: " + p->compute_type +
                         " weight_type: " + p->weight_type);
}

std::string get_dtype_str(BTLA_DTYPE dtype) {
  switch (dtype) {
    case BTLA_DTYPE::F32:
      return "fp32";
    case BTLA_DTYPE::BF16:
      return "bf16";
    case BTLA_DTYPE::S4_CLIP:
      return "int4";
    case BTLA_DTYPE::S3_CLIP:
      return "int3";
    case BTLA_DTYPE::S2_CLIP:
      return "int2";
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

auto get_torch_dtype(BTLA_DTYPE dtype) {
  switch (dtype) {
    case BTLA_DTYPE::F32:
      return torch::kF32;
    case BTLA_DTYPE::BF16:
      return torch::kBFloat16;
    case BTLA_DTYPE::S8:
      return torch::kInt8;
    default:
      TORCH_CHECK(false, "QBits: unsupported unpack dtype, only support fp32/bf16/int8 now.");
      break;
  }
}

int get_sizeof_bestla_dtype(BTLA_DTYPE dtype) {
  switch (dtype) {
    case BTLA_DTYPE::F32:
    case BTLA_DTYPE::S32:
    case BTLA_DTYPE::U32:
      return 4;
    case BTLA_DTYPE::BF16:
    case BTLA_DTYPE::F16:
      return 2;
    case BTLA_DTYPE::S8:
    case BTLA_DTYPE::U8:
      return 1;
    default:
      assert(0);
      break;
  }
  return -1;
}

void bestla_2dcpy_tensor(int row, int col, int ld_src, torch::Tensor& dst, void* src, BTLA_DTYPE dtype) {
  dst = torch::empty({row, col}, get_torch_dtype(dtype));
  auto dt_size = get_sizeof_bestla_dtype(dtype);
  for (int i = 0; i < row; i++) {
    memcpy(reinterpret_cast<char*>(dst.data_ptr()) + i * col * dt_size,
           reinterpret_cast<char*>(src) + i * ld_src * dt_size, col * dt_size);
  }
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
    case IS_ASYM:
      return output.index_put_({0}, static_cast<int64_t>(packw_ptr->IsAsym() ? 1 : 0));
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
    case ZP_TENSOR: {
      TORCH_CHECK(packw_ptr->ZPtr<void>() != nullptr, "QBits: not pack zero-point tensor.");
      bestla_2dcpy_tensor((packw_ptr->mK + packw_ptr->mBlockSize - 1) / packw_ptr->mBlockSize, packw_ptr->mN,
                          packw_ptr->mNPad, output, packw_ptr->ZPtr<void>(), packw_ptr->ZDtype());
    } break;
    case SCALE_TENSOR: {
      bestla_2dcpy_tensor((packw_ptr->mK + packw_ptr->mBlockSize - 1) / packw_ptr->mBlockSize, packw_ptr->mN,
                          packw_ptr->mNPad, output, packw_ptr->SPtr<void>(), packw_ptr->SDtype());
    } break;
    default:
      TORCH_CHECK(false, "QBits: unsupported acquire_type");
      break;
  }
  return output;
}

void bestla_packq(repack_quantized_weight_param* p, repack_quantized_weight_ctx* ctx, WOQ_TASK task) {
  if (p->compute_type == "int8") {
    TORCH_CHECK(
        p->weight_type == "int8" || p->weight_type == "int4" || p->weight_type == "int3" || p->weight_type == "int2",
        "Qbits: only support Integer weight-type with int8 compute-type");
    if (dispatcher_utils::check_amx() && p->blocksize % bestla::gemm::ICoreRowNAmxint8KBlock<64, 16>::KTILE == 0) {
      return parse_prob<bestla::gemm::ICoreRowNAmxint8KBlock<64, 16>, BTLA_ISA::AMX_INT8>(p, ctx, task);
    }
    if (dispatcher_utils::check_avx512_vnni() &&
        p->blocksize % bestla::gemm::ICoreRowNAvx512vnniKBlock<48, 4>::KTILE == 0) {
      return parse_prob<bestla::gemm::ICoreRowNAvx512vnniKBlock<48, 4>, BTLA_ISA::AVX512_VNNI>(p, ctx, task);
    }
    if (dispatcher_utils::check_avx_vnni() && p->blocksize % bestla::gemm::ICoreRowNAvxvnniKBlock<24, 2>::KTILE == 0) {
      return parse_prob<bestla::gemm::ICoreRowNAvxvnniKBlock<24, 2>, BTLA_ISA::AVX_VNNI>(p, ctx, task);
    }
    if (dispatcher_utils::check_avx2() && p->blocksize % bestla::gemm::ICoreRowNAvx2vnniKBlock<24, 2>::KTILE == 0) {
      return parse_prob<bestla::gemm::ICoreRowNAvx2vnniKBlock<24, 2>, BTLA_ISA::AVX2>(p, ctx, task);
    }
    TORCH_CHECK(false, "Qbits: Illegal config in int8 compute_type, blocksize:", p->blocksize,
                ", ISA support avx2:", dispatcher_utils::check_avx2());
  }
  if (p->compute_type == "fp32") {
    if (dispatcher_utils::check_avx512f()) {
      return parse_prob<bestla::gemm::SCoreRowNAvx512f<48, 8>, BTLA_ISA::AVX512F>(p, ctx, task);
    }
    if (dispatcher_utils::check_avx2()) {
      return parse_prob<bestla::gemm::SCoreRowNAvx2<24, 4>, BTLA_ISA::AVX2>(p, ctx, task);
    }
    TORCH_CHECK(false, "Qbits: device ISA must support BTLA_ISA::AVX2 when compute_type==fp32");
  }
  if (p->compute_type == "bf16") {
    if (dispatcher_utils::check_amx()) {
      return parse_prob<bestla::gemm::HCoreRowNAmxbf16<64, 16>, BTLA_ISA::AMX_BF16>(p, ctx, task);
    }
    TORCH_CHECK(false, "Qbits: device ISA must support AMX-BF16 when compute_type==bf16");
  }
  TORCH_CHECK(false, "Qbits: unsupported bestla_config, compute_type:", p->compute_type,
              ", weight_type:", p->weight_type + ", blocksize:", p->blocksize);
}
}  // namespace woq
