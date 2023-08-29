#include "jblas_common.hpp"

using namespace jblas;
using namespace ne_jblas;

bool jblas_fusion_FFN_SiLu_f32f32_support(void* w1ptr, void* w2ptr, void* w3ptr, int seq, int fin, int fmid, int fout) {
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  auto w3tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w3ptr, 0);
  bool support = false;
  if (w1tmp != nullptr && w2tmp != nullptr && w3tmp != nullptr) {
    prologue::PackedWeight* tmps[3] = {w1tmp, w2tmp, w3tmp};
    auto sameKernel = samePackedWeight(tmps, 3);
    if (sameKernel) {
      if (sameKernel) {
        if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32) ||
            w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
          constexpr size_t EleNum = sizeof(GcCompInt8KBlockSet) / sizeof(GcCompInt8KBlockSet[0]);
          support = contains(w1tmp->mCoreType, GcCompInt8KBlockSet, EleNum);
          support &= hasISA(GcCompInt8KBlockSet, EleNum);
        }
      }
    }
  }
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  safe_delete(w3tmp);
  return support;
}

JBLAS_CODE jblas_fusion_FFN_SiLu_s4fp32_f32f32_forward(float* activation, SS4Fp32* w1ptr, SS4Fp32* w2ptr,
                                                       SS4Fp32* w3ptr, float* tmp1, float* tmp2, float* output, int seq,
                                                       int fin, int fmid, int fout) {
  if (w1ptr->mCoreType == GcCompInt8KBlock::TYPE) {
    using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS4KBlock;
    using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmSKernelDynamicS4KBlock;
    using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
    static FusedInter finter;
    int lda = fin;
    int ldtmp1 = fmid;
    int ldtmp2 = fmid;
    int ldo = fout;

    auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize, NULL);
    auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize, NULL);
    auto ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, quanA1, tmp1, ldtmp1, quanA2, w1ptr,
                               w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,   tmp2, ldtmp2, NULL});
    delete quanA1;
    delete quanA2;
    return ret;
  }
  return JblasNotSupport;
}
JBLAS_CODE jblas_fusion_FFN_SiLu_s8fp32_f32f32_forward(float* activation, SS8Fp32* w1ptr, SS8Fp32* w2ptr,
                                                       SS8Fp32* w3ptr, float* tmp1, float* tmp2, float* output, int seq,
                                                       int fin, int fmid, int fout) {
  if (w1ptr->mCoreType == GcCompInt8KBlock::TYPE) {
    using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS8KBlock;
    using SiluGemmKernel = custom::wrapper::kblock::avx512_vnni::SiluGemmSKernelDynamicS8KBlock;
    using FusedInter = custom::wrapper::transformer::FFNFusedInterface<SiluGemmKernel, GemmKernel>;
    static FusedInter finter;
    int lda = fin;
    int ldtmp1 = fmid;
    int ldtmp2 = fmid;
    int ldo = fout;
    auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1ptr->mBlockSize, NULL);
    auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w2ptr->mBlockSize, NULL);
    auto ret = finter.compute({seq,   fin,   fmid, fout,   activation, lda, quanA1, tmp1, ldtmp1, quanA2, w1ptr,
                               w2ptr, w3ptr, tmp1, ldtmp1, output,     ldo, NULL,   tmp2, ldtmp2, NULL});
    delete quanA1;
    delete quanA2;
    return ret;
  }
  return JblasNotSupport;
}

void jblas_fusion_FFN_SiLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                          float* tmp2, float* output, int seq, int fin, int fmid, int fout) {
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  auto w3tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w3ptr, 0);
  auto ret = JblasRuntimeError;

  // must check support before forward, there is no need to check support twice.
  if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret = jblas_fusion_FFN_SiLu_s4fp32_f32f32_forward(activation, dynamic_cast<SS4Fp32*>(w1tmp),
                                                      dynamic_cast<SS4Fp32*>(w2tmp), dynamic_cast<SS4Fp32*>(w3tmp),
                                                      tmp1, tmp2, output, seq, fin, fmid, fout);
  } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
    ret = jblas_fusion_FFN_SiLu_s8fp32_f32f32_forward(activation, dynamic_cast<SS8Fp32*>(w1tmp),
                                                      dynamic_cast<SS8Fp32*>(w2tmp), dynamic_cast<SS8Fp32*>(w3tmp),
                                                      tmp1, tmp2, output, seq, fin, fmid, fout);
  }
  assert(ret == JblasSuccess);
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  safe_delete(w3tmp);
}

bool jblas_fusion_FFN_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout) {
  return false;
}

void jblas_fusion_FFN_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* tmp1, float* output,
                                          int seq, int fin, int fmid, int fout) {
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  if (w1tmp->mCoreType == GcCompInt8KBlock::TYPE) {
    using GemmKernel = custom::wrapper::kblock::avx512_vnni::GemmSKernelDynamicS4KBlock;
    using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::GeluGemmSKernelDynamicS4KBlock;
    using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
    static FusedInter finter;
    int lda = fin;
    int ldtmp1 = fmid;
    int ldo = fout;
    /*auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, NULL);
    auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w1tmp->mBlockSize, NULL);
    finter.compute({seq, fin, fmid, fout, activation, lda, w1tmp, w2tmp, tmp1, ldtmp1, output, ldo});*/
  }
  safe_delete(w1tmp);
  safe_delete(w2tmp);
}

bool jblas_fusion_FFN_Add_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout) {
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  bool support = false;
  if (w1tmp != nullptr && w2tmp != nullptr) {
    prologue::PackedWeight* tmps[2] = {w1tmp, w2tmp};
    auto sameKernel = samePackedWeight(tmps, 2);
    if (sameKernel) {
      if (sameKernel) {
        if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32) ||
            w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
          constexpr size_t EleNum = sizeof(GcCompInt8KBlockSet) / sizeof(GcCompInt8KBlockSet[0]);
          support = contains(w1tmp->mCoreType, GcCompInt8KBlockSet, EleNum);
          support &= hasISA(GcCompInt8KBlockSet, EleNum);
        } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
          constexpr size_t EleNum = sizeof(GcCompInt8Set) / sizeof(GcCompInt8Set[0]);
          support = contains(w1tmp->mCoreType, GcCompInt8Set, EleNum);
          support &= hasISA(GcCompInt8Set, EleNum);
        }
      }
    }
  }
  safe_delete(w1tmp);
  safe_delete(w2tmp);
  return support;
}

JBLAS_CODE jblas_fusion_FFN_Add_GeLu_s4fp32_f32f32_forward(float* activation, SS4Fp32* w1tmp, SS4Fp32* w2tmp,
                                                           float* b1ptr, float* b2ptr, float* tmp1, float* output,
                                                           int seq, int fin, int fmid, int fout, bool broadcast_bias) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (w1tmp->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && w1tmp->mBlockSize % 128 == 0) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::AddGemmSKernelDynamicS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::amx_int8::AddGeluGemmSKernelDynamicS4KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, NULL);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w1tmp->mBlockSize, NULL);
      ret = finter.compute({seq,        fin,    fmid,   fout,
                            activation, lda,    quanA1, tmp1,
                            ldtmp1,     quanA2, w1tmp,  w2tmp,
                            tmp1,       b1ptr,  ldtmp1, broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,  ldo,    broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::AddGemmSKernelDynamicS4KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::AddGeluGemmSKernelDynamicS4KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, NULL);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w1tmp->mBlockSize, NULL);
      ret = finter.compute({seq,        fin,    fmid,   fout,
                            activation, lda,    quanA1, tmp1,
                            ldtmp1,     quanA2, w1tmp,  w2tmp,
                            tmp1,       b1ptr,  ldtmp1, broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,  ldo,    broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_FFN_Add_GeLu_s8fp32_f32f32_forward(float* activation, SS8Fp32* w1tmp, SS8Fp32* w2tmp,
                                                           float* b1ptr, float* b2ptr, float* tmp1, float* output,
                                                           int seq, int fin, int fmid, int fout, bool broadcast_bias) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (w1tmp->mCoreType == GcCompInt8KBlock::TYPE) {
    if (_cd->AMX_INT8() && w1tmp->mBlockSize % 128 == 0) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::AddGemmSKernelDynamicS8KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::amx_int8::AddGeluGemmSKernelDynamicS8KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, NULL);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w1tmp->mBlockSize, NULL);
      ret = finter.compute({seq,        fin,    fmid,   fout,
                            activation, lda,    quanA1, tmp1,
                            ldtmp1,     quanA2, w1tmp,  w2tmp,
                            tmp1,       b1ptr,  ldtmp1, broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,  ldo,    broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::AddGemmSKernelDynamicS8KBlock;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::AddGeluGemmSKernelDynamicS8KBlock;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterface<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, w1tmp->mBlockSize, NULL);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, w1tmp->mBlockSize, NULL);
      ret = finter.compute({seq,        fin,    fmid,   fout,
                            activation, lda,    quanA1, tmp1,
                            ldtmp1,     quanA2, w1tmp,  w2tmp,
                            tmp1,       b1ptr,  ldtmp1, broadcast_bias ? 0 : ldtmp1,
                            output,     b2ptr,  ldo,    broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    }
  }
  return ret;
}

JBLAS_CODE jblas_fusion_FFN_Add_GeLu_s8fp32pern_f32f32_forward(float* activation, SS8Fp32PerN* w1tmp,
                                                               SS8Fp32PerN* w2tmp, float* b1ptr, float* b2ptr,
                                                               float* tmp1, float* output, int seq, int fin, int fmid,
                                                               int fout, bool broadcast_bias) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  if (w1tmp->mCoreType == GcCompInt8::TYPE) {
    if (_cd->AMX_INT8()) {
      using GemmKernel = custom::wrapper::kblock::amx_int8::AddGemmDynamicS8PerN;
      using GeluGemmKernel = custom::wrapper::kblock::amx_int8::AddGeluGemmDynamicSPerN;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterfacePerN<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, NULL);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, NULL);
      ret = finter.compute({seq,
                            fin,
                            fmid,
                            fout,
                            activation,
                            lda,
                            quanA1,
                            tmp1,
                            ldtmp1,
                            quanA2,
                            w1tmp,
                            w2tmp,
                            tmp1,
                            ldtmp1,
                            quanA1->mSPtr,
                            quanA1->lds,
                            w1tmp->mSPtr,
                            b1ptr,
                            broadcast_bias ? 0 : ldtmp1,
                            output,
                            ldo,
                            quanA2->mSPtr,
                            quanA2->lds,
                            w2tmp->mSPtr,
                            b2ptr,
                            broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    } else if (_cd->AVX512_VNNI()) {
      using GemmKernel = custom::wrapper::kblock::avx512_vnni::AddGemmDynamicS8PerN;
      using GeluGemmKernel = custom::wrapper::kblock::avx512_vnni::AddGeluGemmDynamicSPerN;
      using FusedInter = custom::wrapper::transformer::GeluFusedInterfacePerN<GeluGemmKernel, GemmKernel>;
      static FusedInter finter;
      int lda = fin;
      int ldtmp1 = fmid;
      int ldo = fout;
      // FusedInter::Arguments::paramA paramA={activation, lda};
      // FusedInter::Arguments::paramW1 paramW1={w1tmp};
      // FusedInter::Arguments::paramW2 paramW2={w2tmp};
      // FusedInter::Arguments::param1 param1={tmp1, b1ptr, ldtmp1, ldtmp1};
      auto quanA1 = finter.getActivationPtr()->createStorage(seq, fin, NULL);
      auto quanA2 = finter.getActivationPtr()->createStorage(seq, fmid, NULL);
      ret = finter.compute({seq,           fin,         fmid,
                            fout,          activation,  lda,
                            quanA1,        tmp1,        ldtmp1,
                            quanA2,        w1tmp,       w2tmp,
                            tmp1,          ldtmp1,      quanA1->mZPtr,
                            quanA1->mSPtr, quanA1->lds, w1tmp->mRPtr,
                            w1tmp->mSPtr,  b1ptr,       broadcast_bias ? 0 : ldtmp1,
                            output,        ldo,         quanA2->mZPtr,
                            quanA2->mSPtr, quanA2->lds, w2tmp->mRPtr,
                            w2tmp->mSPtr,  b2ptr,       broadcast_bias ? 0 : ldo});
      delete quanA1;
      delete quanA2;
    }
  }
  return ret;
}

void jblas_fusion_FFN_Add_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* b1ptr, float* b2ptr,
                                              float* tmp1, float* output, int seq, int fin, int fmid, int fout,
                                              bool broadcast_bias) {
  GetCPUDevice();
  auto ret = JblasRuntimeError;
  auto w1tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w1ptr, 0);
  auto w2tmp = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(w2ptr, 0);
  if (w1tmp->mType == int(WeightCompType::WeightS4ClipScaleFp32)) {
    ret = jblas_fusion_FFN_Add_GeLu_s4fp32_f32f32_forward(activation, (SS4Fp32*)w1tmp, (SS4Fp32*)w2tmp, b1ptr, b2ptr,
                                                          tmp1, output, seq, fin, fmid, fout, broadcast_bias);
  } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32)) {
    ret = jblas_fusion_FFN_Add_GeLu_s8fp32_f32f32_forward(activation, (SS8Fp32*)w1tmp, (SS8Fp32*)w2tmp, b1ptr, b2ptr,
                                                          tmp1, output, seq, fin, fmid, fout, broadcast_bias);
  } else if (w1tmp->mType == int(WeightCompType::WeightS8ScaleFp32PerChannelN)) {
    ret =
        jblas_fusion_FFN_Add_GeLu_s8fp32pern_f32f32_forward(activation, (SS8Fp32PerN*)w1tmp, (SS8Fp32PerN*)w2tmp, b1ptr,
                                                            b2ptr, tmp1, output, seq, fin, fmid, fout, broadcast_bias);
  }
  assert(ret == JblasSuccess);
  safe_delete(w1tmp);
  safe_delete(w2tmp);
}
