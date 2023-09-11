/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#include "common/core/base_ops.hpp"
#include "common/core/base_types.hpp"
#include "common/core/common.hpp"
#include "common/utils/limitation.hpp"

namespace gpu::xetla {

namespace detail {

/// @brief lookup table for cache hint.
///
///
constexpr __ESIMD_ENS::cache_hint get_cache_hint(gpu::xetla::cache_hint ch) {
    switch (ch) {
        case gpu::xetla::cache_hint::none: return __ESIMD_ENS::cache_hint::none;
        case gpu::xetla::cache_hint::uncached:
            return __ESIMD_ENS::cache_hint::uncached;
        case gpu::xetla::cache_hint::cached:
            return __ESIMD_ENS::cache_hint::cached;
        case gpu::xetla::cache_hint::write_back:
            return __ESIMD_ENS::cache_hint::write_back;
        case gpu::xetla::cache_hint::write_through:
            return __ESIMD_ENS::cache_hint::write_through;
        case gpu::xetla::cache_hint::streaming:
            return __ESIMD_ENS::cache_hint::streaming;
        case gpu::xetla::cache_hint::read_invalidate:
            return __ESIMD_ENS::cache_hint::read_invalidate;
    }
}

/// @brief lookup table for data size.
///
///
constexpr __ESIMD_ENS::lsc_data_size get_data_size(gpu::xetla::data_size ds) {
    switch (ds) {
        case gpu::xetla::data_size::default_size:
            return __ESIMD_ENS::lsc_data_size::default_size;
        case gpu::xetla::data_size::u8: return __ESIMD_ENS::lsc_data_size::u8;
        case gpu::xetla::data_size::u16: return __ESIMD_ENS::lsc_data_size::u16;
        case gpu::xetla::data_size::u32: return __ESIMD_ENS::lsc_data_size::u32;
        case gpu::xetla::data_size::u64: return __ESIMD_ENS::lsc_data_size::u64;
        case gpu::xetla::data_size::u8u32:
            return __ESIMD_ENS::lsc_data_size::u8u32;
        case gpu::xetla::data_size::u16u32:
            return __ESIMD_ENS::lsc_data_size::u16u32;
        case gpu::xetla::data_size::u16u32h:
            return __ESIMD_ENS::lsc_data_size::u16u32h;
    }
}

/// @brief lookup table for memory kind.
///
///
constexpr __ESIMD_ENS::lsc_memory_kind get_memory_kind(
        gpu::xetla::memory_kind mk) {
    switch (mk) {
        case gpu::xetla::memory_kind::untyped_global:
            return __ESIMD_ENS::lsc_memory_kind::untyped_global;
        case gpu::xetla::memory_kind::untyped_global_low_pri:
            return __ESIMD_ENS::lsc_memory_kind::untyped_global_low_pri;
        case gpu::xetla::memory_kind::typed_global:
            return __ESIMD_ENS::lsc_memory_kind::typed_global;
        case gpu::xetla::memory_kind::shared_local:
            return __ESIMD_ENS::lsc_memory_kind::shared_local;
    }
}

/// @brief lookup table for fence op.
///
///
constexpr __ESIMD_ENS::lsc_fence_op get_fence_op(gpu::xetla::fence_op fo) {
    switch (fo) {
        case gpu::xetla::fence_op::none: return __ESIMD_ENS::lsc_fence_op::none;
        case gpu::xetla::fence_op::evict:
            return __ESIMD_ENS::lsc_fence_op::evict;
        case gpu::xetla::fence_op::invalidate:
            return __ESIMD_ENS::lsc_fence_op::invalidate;
        case gpu::xetla::fence_op::discard:
            return __ESIMD_ENS::lsc_fence_op::discard;
        case gpu::xetla::fence_op::clean:
            return __ESIMD_ENS::lsc_fence_op::clean;
        case gpu::xetla::fence_op::flushl3:
            return __ESIMD_ENS::lsc_fence_op::flushl3;
    }
}

/// @brief lookup table for fence scope.
///
///
constexpr __ESIMD_ENS::lsc_scope get_fence_scope(gpu::xetla::fence_scope fs) {
    switch (fs) {
        case gpu::xetla::fence_scope::group:
            return __ESIMD_ENS::lsc_scope::group;
        case gpu::xetla::fence_scope::local:
            return __ESIMD_ENS::lsc_scope::local;
        case gpu::xetla::fence_scope::tile: return __ESIMD_ENS::lsc_scope::tile;
        case gpu::xetla::fence_scope::gpu: return __ESIMD_ENS::lsc_scope::gpu;
        case gpu::xetla::fence_scope::gpus: return __ESIMD_ENS::lsc_scope::gpus;
        case gpu::xetla::fence_scope::system:
            return __ESIMD_ENS::lsc_scope::system;
        case gpu::xetla::fence_scope::sysacq:
            return __ESIMD_ENS::lsc_scope::sysacq;
    }
}

/// @brief lookup table for atomic op.
///
///
constexpr __ESIMD_NS::atomic_op get_atomic_op(gpu::xetla::atomic_op ao) {
    switch (ao) {
        case gpu::xetla::atomic_op::iinc: return __ESIMD_NS::atomic_op::inc;
        case gpu::xetla::atomic_op::idec: return __ESIMD_NS::atomic_op::dec;
        case gpu::xetla::atomic_op::iadd: return __ESIMD_NS::atomic_op::add;
        case gpu::xetla::atomic_op::isub: return __ESIMD_NS::atomic_op::sub;
        case gpu::xetla::atomic_op::smin: return __ESIMD_NS::atomic_op::smin;
        case gpu::xetla::atomic_op::smax: return __ESIMD_NS::atomic_op::smax;
        case gpu::xetla::atomic_op::umin: return __ESIMD_NS::atomic_op::umin;
        case gpu::xetla::atomic_op::umax: return __ESIMD_NS::atomic_op::umax;
        case gpu::xetla::atomic_op::cmpxchg:
            return __ESIMD_NS::atomic_op::cmpxchg;
        case gpu::xetla::atomic_op::fadd: return __ESIMD_NS::atomic_op::fadd;
        case gpu::xetla::atomic_op::fsub: return __ESIMD_NS::atomic_op::fsub;
        case gpu::xetla::atomic_op::fmin: return __ESIMD_NS::atomic_op::fmin;
        case gpu::xetla::atomic_op::fmax: return __ESIMD_NS::atomic_op::fmax;
        case gpu::xetla::atomic_op::fcmpxchg:
            return __ESIMD_NS::atomic_op::fcmpwr;
        case gpu::xetla::atomic_op::bit_and:
            return __ESIMD_NS::atomic_op::bit_and;
        case gpu::xetla::atomic_op::bit_or:
            return __ESIMD_NS::atomic_op::bit_or;
        case gpu::xetla::atomic_op::bit_xor:
            return __ESIMD_NS::atomic_op::bit_xor;
        case gpu::xetla::atomic_op::load: return __ESIMD_NS::atomic_op::load;
        case gpu::xetla::atomic_op::store: return __ESIMD_NS::atomic_op::store;
    }
}
} // namespace detail

/// @addtogroup xetla_core_memory
/// @{

/// @brief Stateless scattered prefetch.
/// Prefetches elements located at specified address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to prefetch per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets in bytes.
/// @param pred    [in] is predicates.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::cached,
        cache_hint L3H = cache_hint::cached, int N>
__XETLA_API void xetla_prefetch_global(
        Ty *p, xetla_vector<uint32_t, N> offsets, xetla_mask<N> pred = 1) {
    using T = native_type_t<Ty>;
    __ESIMD_ENS::lsc_prefetch<T, NElts, gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L3H), N>((T *)p, offsets, pred);
}

/// @brief Stateless block prefetch (transposed gather with 1 channel).
/// Prefetches elements located at specified address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to prefetch per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p      [in] is the base pointer.
/// @param offset [in] is the zero-based offset in bytes.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::cached,
        cache_hint L3H = cache_hint::cached>
__XETLA_API void xetla_prefetch_global(Ty *p, uint64_t offset = 0) {
    using T = native_type_t<Ty>;
    __ESIMD_ENS::lsc_prefetch<T, NElts, gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L3H)>(
            (T *)p + (offset / sizeof(T)));
}

/// @brief Stateless scattered load.
/// Collects elements located at specified address and returns them
/// to a single \ref xetla_vector object.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to load per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets in bytes.
/// @param pred    [in] is predicates.
/// @return  is a xetla_vector of type T and size N * NElts.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
        int N, typename Toffset = uint32_t>
__XETLA_API xetla_vector<Ty, N * NElts> xetla_load_global(
        Ty *p, xetla_vector<Toffset, N> offsets, xetla_mask<N> pred = 1) {
    using T = native_type_t<Ty>;
    DEBUG_INVOKE(dbg_level::core,
            core::general_1d<gpu_arch::Xe,
                    Ty>::template check_restriction<NElts, N>(offsets,
                    (uint64_t)p));

    return __ESIMD_ENS::lsc_gather<T, NElts,
            gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L3H), N>((T *)p, offsets, pred);
}

/// @brief Stateless block load (transposed gather with 1 channel).
/// Collects elements located at specified address and returns them
/// to a single \ref xetla_vector object.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to load per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p      [in] is the base pointer.
/// @param offset [in] is the zero-based offset in bytes.
/// @return is a xetla_vector of type T and size NElts.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__XETLA_API xetla_vector<Ty, NElts> xetla_load_global(
        Ty *p, uint64_t offset = 0) {
    using T = native_type_t<Ty>;
    DEBUG_INVOKE(dbg_level::core,
            core::general_1d<gpu_arch::Xe,
                    Ty>::template check_restriction<NElts>(offset,
                    (uint64_t)p));

    return __ESIMD_ENS::lsc_block_load<T, NElts,
            gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L3H)>(
            (T *)p + (offset / sizeof(T)));
}

/// @brief Stateless scattered store.
/// Writes elements to specific address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_store.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to store per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets in bytes.
/// @param vals    [in] is values to store.
/// @param pred    [in] is predicates.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
        int N, typename Toffset = uint32_t>
__XETLA_API void xetla_store_global(Ty *p, xetla_vector<Toffset, N> offsets,
        xetla_vector<Ty, N * NElts> vals, xetla_mask<N> pred = 1) {
    using T = native_type_t<Ty>;
    __ESIMD_ENS::lsc_scatter<T, NElts, gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L3H), N>(
            (T *)p, offsets, vals, pred);
}

/// @brief Stateless block store (transposed scatter with 1 channel).
/// Writes elements to specific address.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_store.ugm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to store per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p      [in] is the base pointer.
/// @param offset [in] is the zero-based offset in bytes.
/// @param vals   [in] is values to store.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__XETLA_API void xetla_store_global(
        Ty *p, uint64_t offset, xetla_vector<Ty, NElts> vals) {
    using T = native_type_t<Ty>;
    __ESIMD_ENS::lsc_block_store<T, NElts,
            gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L3H)>(
            (T *)p + (offset / sizeof(T)), vals);
}

/// @brief Stateless scattered atomic (0 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__XETLA_API xetla_vector<T, N> xetla_atomic_global(
        T *p, xetla_vector<uint32_t, N> offsets, xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return __ESIMD_ENS::lsc_atomic_update<gpu::xetla::detail::get_atomic_op(Op),
            T, N, gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L3H)>(p, offsets, pred);
}

/// @brief Stateless scattered atomic (1 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets.
/// @param src0    [in] is the first atomic operand.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__XETLA_API xetla_vector<T, N> xetla_atomic_global(T *p,
        xetla_vector<uint32_t, N> offsets, xetla_vector<T, N> src0,
        xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return __ESIMD_ENS::lsc_atomic_update<gpu::xetla::detail::get_atomic_op(Op),
            T, N, gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L3H)>(p, offsets, src0, pred);
}

/// @brief Stateless scattered atomic (2 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p       [in] is the base pointer.
/// @param offsets [in] is the zero-based offsets.
/// @param src0    [in] is the first atomic operand.
/// @param src1    [in] is the second atomic operand.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size,
        cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__XETLA_API xetla_vector<T, N> xetla_atomic_global(T *p,
        xetla_vector<uint32_t, N> offsets, xetla_vector<T, N> src0,
        xetla_vector<T, N> src1, xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return __ESIMD_ENS::lsc_atomic_update<gpu::xetla::detail::get_atomic_op(Op),
            T, N, gpu::xetla::detail::get_data_size(DS),
            gpu::xetla::detail::get_cache_hint(L1H),
            gpu::xetla::detail::get_cache_hint(L3H)>(
            p, offsets, src0, src1, pred);
}
/// @brief Declare per-work-group slm size.
/// @tparam SLMSize  Shared Local Memory (SLM) size (in Bytes).
template <uint32_t SLMSize>
__XETLA_API void xetla_local_init() {
    if constexpr (SLMSize != 0) { __ESIMD_NS::slm_init(SLMSize); }
}

/// @brief SLM scattered load.
/// Collects elements located at slm and returns them as a single \ref xetla_vector object.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.slm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to load per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param offsets [in] is the zero-based offsets for SLM buffer in bytes.
/// @param pred    [in] is predicates.
/// @return is a xetla_vector of type T and size N * NElts.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size, int N>
__XETLA_API xetla_vector<Ty, N * NElts> xetla_load_local(
        xetla_vector<uint32_t, N> offsets, xetla_mask<N> pred = 1) {
    using T = native_type_t<Ty>;
    DEBUG_INVOKE(dbg_level::core,
            core::general_1d<gpu_arch::Xe,
                    Ty>::template check_restriction<NElts, N>(offsets));

    return __ESIMD_ENS::lsc_slm_gather<T, NElts,
            gpu::xetla::detail::get_data_size(DS), N>(
            xetla_cvt<uint64_t, uint32_t>(offsets), pred);
}

/// @brief SLM block load. (transposed gather with 1 channel).
/// Collects elements located at slm and returns them as a single \ref xetla_vector object.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_load.slm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to load per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @param offset [in] is the zero-based offset for SLM buffer in bytes.
/// @return is a xetla_vector of type T and size NElts.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size>
__XETLA_API xetla_vector<Ty, NElts> xetla_load_local(uint32_t offset) {
    using T = native_type_t<Ty>;
    DEBUG_INVOKE(dbg_level::core,
            core::general_1d<gpu_arch::Xe,
                    Ty>::template check_restriction<NElts>((uint64_t)offset));

    return __ESIMD_ENS::lsc_slm_block_load<T, NElts,
            gpu::xetla::detail::get_data_size(DS)>(offset);
}

/// @brief SLM scattered store.
/// Scatters elements located to slm.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_store.slm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to store per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param offsets [in] is the zero-based offsets for SLM buffer in bytes.
/// @param vals    [in] is values to store.
/// @param pred    [in] is predicates.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size, int N>
__XETLA_API void xetla_store_local(xetla_vector<uint32_t, N> offsets,
        xetla_vector<Ty, N * NElts> vals, xetla_mask<N> pred = 1) {
    using T = native_type_t<Ty>;
    DEBUG_INVOKE(dbg_level::core,
            core::general_1d<gpu_arch::Xe, Ty>::template check_restriction<
                    NElts, N, uint32_t>(offsets));

    __ESIMD_ENS::lsc_slm_scatter<T, NElts,
            gpu::xetla::detail::get_data_size(DS), N>(offsets, vals, pred);
}

/// @brief SLM block store (transposed SLM scatter with 1 channel).
/// Scatters elements located to slm.
///
/// Supported platforms: DG2, PVC
///
/// VISA instruction: lsc_store.slm
///
/// @tparam Ty is element type.
/// @tparam NElts is the number of elements to store per address (i.e. vector_size per SIMD channel).
/// @tparam DS is the data size.
/// @param offset [in] is the zero-based offset for SLM buffer in bytes.
/// @param vals   [in] is values to store.
///
template <typename Ty, uint8_t NElts = 1,
        data_size DS = data_size::default_size>
__XETLA_API void xetla_store_local(
        uint32_t offset, xetla_vector<Ty, NElts> vals) {
    using T = native_type_t<Ty>;
    DEBUG_INVOKE(dbg_level::core,
            core::general_1d<gpu_arch::Xe,
                    Ty>::template check_restriction<NElts>(offset));

    __ESIMD_ENS::lsc_slm_block_store<T, NElts,
            gpu::xetla::detail::get_data_size(DS)>(offset, vals);
}

/// @brief SLM scattered atomic (0 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets [in] is the zero-based offsets.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size>
__XETLA_API xetla_vector<T, N> xetla_atomic_local(
        xetla_vector<uint32_t, N> offsets, xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return __ESIMD_ENS::lsc_slm_atomic_update<gpu::xetla::detail::get_atomic_op(
                                                      Op),
            T, N, gpu::xetla::detail::get_data_size(DS)>(offsets, pred);
}

/// @brief SLM scattered atomic (1 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets [in] is the zero-based offsets.
/// @param src0    [in] is the first atomic operand.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size>
__XETLA_API xetla_vector<T, N> xetla_atomic_local(
        xetla_vector<uint32_t, N> offsets, xetla_vector<T, N> src0,
        xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return __ESIMD_ENS::lsc_slm_atomic_update<gpu::xetla::detail::get_atomic_op(
                                                      Op),
            T, N, gpu::xetla::detail::get_data_size(DS)>(offsets, src0, pred);
}

/// @brief SLM scattered atomic (2 src).
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets [in] is the zero-based offsets.
/// @param src0    [in] is the first atomic operand.
/// @param src1    [in] is the second atomic operand.
/// @param pred    [in] is predicates.
///
template <atomic_op Op, typename T, int N,
        data_size DS = data_size::default_size>
__XETLA_API xetla_vector<T, N> xetla_atomic_local(
        xetla_vector<uint32_t, N> offsets, xetla_vector<T, N> src0,
        xetla_vector<T, N> src1, xetla_mask<N> pred) {
    static_assert(!(is_internal_type<T>::value),
            "The internal types are not yet supported!");
    return __ESIMD_ENS::lsc_slm_atomic_update<gpu::xetla::detail::get_atomic_op(
                                                      Op),
            T, N, gpu::xetla::detail::get_data_size(DS)>(
            offsets, src0, src1, pred);
}

/// @brief Memory fence.
/// Supported platforms: DG2, PVC
///
/// @tparam Kind is the Sfid shaded function.
/// @tparam FenceOp is the fence operation.
/// @tparam Scope is the operation scope.
/// @tparam N is the number of SIMD channels (platform dependent).
/// @param pred is predicates.
template <memory_kind Kind = memory_kind::untyped_global,
        fence_op FenceOp = fence_op::none,
        fence_scope Scope = fence_scope::group, int N = 16>
__XETLA_API void xetla_fence(xetla_mask<N> pred = 1) {
    __ESIMD_ENS::lsc_fence<gpu::xetla::detail::get_memory_kind(Kind),
            gpu::xetla::detail::get_fence_op(FenceOp),
            gpu::xetla::detail::get_fence_scope(Scope), N>(pred);
}

/// @} xetla_core_memory

} // namespace gpu::xetla
