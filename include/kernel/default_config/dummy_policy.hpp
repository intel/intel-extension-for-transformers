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

#pragma once

#include "kernel/default_config/common.hpp"

namespace gpu::xetla {
template <param_optimizer_tag tag_, typename dict_t_, typename... candidates_t>
struct dummy_optimizer : param_optimizer_base {
    struct impl {
        enum class eval_tag { TYPE, DELTA, LINEAR, SQUARE };

        template <auto key_, typename T, typename U, eval_tag eval_tag_,
                typename = void>
        struct param_distance_eval_fcn;

        template <auto key_, typename T, typename U>
        struct param_distance_eval_fcn<key_, T, U, eval_tag::TYPE> {
            static constexpr int value = []() constexpr {
                using T_L = typename T::template find_elem_t<key_>::type;
                using T_R = typename U::template find_elem_t<key_>::type;
                return (std::is_same<T, U>::value) ? 0 : 1;
            }
            ();
        };

        template <auto key_, typename T, typename U>
        struct param_distance_eval_fcn<key_, T, U, eval_tag::DELTA> {
            static constexpr int value = []() constexpr {
                auto l = T::template find_elem_v<key_>;
                auto r = U::template find_elem_v<key_>;
                return (l == r) ? 0 : 1;
            }
            ();
        };

        template <typename T>
        static constexpr T const_max(const T &l, const T &r) {
            return (l > r) ? l : r;
        }

        template <typename T>
        static constexpr T const_min(const T &l, const T &r) {
            return (l > r) ? r : l;
        }

        template <auto key_, typename T, typename U>
        struct param_distance_eval_fcn<key_, T, U, eval_tag::LINEAR> {
            static constexpr int value = []() constexpr {
                auto l = T::template find_elem_v<key_>;
                auto r = U::template find_elem_v<key_>;
                return (const_max(l, r) - const_min(l, r));
            }
            ();
        };

        template <auto key_, typename T, typename U>
        struct param_distance_eval_fcn<key_, T, U, eval_tag::SQUARE> {
            static constexpr int value = []() constexpr {
                auto l = T::template find_elem_v<key_>;
                auto r = U::template find_elem_v<key_>;
                auto ret = (l - r);
                return ret * ret;
            }
            ();
        };

        template <auto key_, typename T, typename U>
        struct param_distance_eval {
            static constexpr int value = []() constexpr {
                using eval_fcn = typename std::conditional<
                        ((key_ == tune_key::DATA_TYPE_A)
                                || (key_ == tune_key::DATA_TYPE_B)
                                || (key_ == tune_key::DATA_TYPE_C)
                                || (key_ == tune_key::DATA_TYPE_ACC)
                                || (key_ == tune_key::EPILOGUE_POLICY)),
                        param_distance_eval_fcn<key_, T, U, eval_tag::TYPE>,
                        typename std::conditional<
                                ((key_ == tune_key::GLOBAL_KSLICING_RATIO)
                                        || (key_
                                                == tune_key::
                                                        LOCAL_KSLICING_RATIO)
                                        || (key_ == tune_key::WG_TILE_K)
                                        || (key_ == tune_key::PREFETCH_DISTANCE)
                                        || (key_
                                                == tune_key::
                                                        PERIODIC_SYNC_INTERVAL)),
                                param_distance_eval_fcn<key_, T, U,
                                        eval_tag::LINEAR>,
                                param_distance_eval_fcn<key_, T, U,
                                        eval_tag::DELTA>>::type>::type;
                switch (key_) {
                    case tune_key::WG_TILE_K: return 10 * eval_fcn::value;
                    case tune_key::GLOBAL_KSLICING_RATIO:
                    case tune_key::LOCAL_KSLICING_RATIO:
                        return 1000 * eval_fcn::value;
                    case tune_key::DATA_TYPE_ACC: return 10 * eval_fcn::value;
                    default: return 10000000 * eval_fcn::value;
                }
                return 0;
            }
            ();
        };

        template <typename T, typename U>
        struct param_distance_eval<tune_key::WG_TILE_SHAPE, T, U> {
            static constexpr int value = []() constexpr {
                using T_L = typename T::template find_elem_t<
                        tune_key::WG_TILE_SHAPE>::type;
                using T_R = typename T::template find_elem_t<
                        tune_key::WG_TILE_SHAPE>::type;

                int l_x = T_L::template dim<0>();
                int l_y = T_L::template dim<1>();

                int r_x = T_R::template dim<0>();
                int r_y = T_R::template dim<1>();

                return (const_max(l_x, r_x) - const_min(l_x, r_x))
                        + (const_max(l_y, r_y) - const_min(l_y, r_y));
            }
            ();
        };

        template <typename T, typename U>
        struct param_distance_eval<tune_key::SG_TILE_SHAPE, T, U> {
            static constexpr int value = []() constexpr {
                using T_L = typename T::template find_elem_t<
                        tune_key::SG_TILE_SHAPE>::type;
                using T_R = typename T::template find_elem_t<
                        tune_key::SG_TILE_SHAPE>::type;

                int l_x = T_L::template dim<0>();
                int l_y = T_L::template dim<1>();

                int r_x = T_R::template dim<0>();
                int r_y = T_R::template dim<1>();

                return (const_max(l_x, r_x) - const_min(l_x, r_x))
                        + (const_max(l_y, r_y) - const_min(l_y, r_y));
            }
            ();
        };

        template <typename T, typename U>
        struct param_distance {
            static constexpr int value = []() constexpr {
                int sum = 0;
                sum += param_distance_eval<tune_key::DATA_TYPE_A, T, U>::value;
                sum += param_distance_eval<tune_key::MEMORY_LAYOUT_A, T,
                        U>::value;
                sum += param_distance_eval<tune_key::MEMORY_ALIGNMENT_A, T,
                        U>::value;
                sum += param_distance_eval<tune_key::DATA_TYPE_B, T, U>::value;
                sum += param_distance_eval<tune_key::MEMORY_LAYOUT_B, T,
                        U>::value;
                sum += param_distance_eval<tune_key::MEMORY_ALIGNMENT_B, T,
                        U>::value;
                sum += param_distance_eval<tune_key::DATA_TYPE_C, T, U>::value;
                sum += param_distance_eval<tune_key::MEMORY_LAYOUT_C, T,
                        U>::value;
                sum += param_distance_eval<tune_key::MEMORY_ALIGNMENT_C, T,
                        U>::value;
                sum += param_distance_eval<tune_key::DATA_TYPE_ACC, T,
                        U>::value;
                if constexpr (tag_ == param_optimizer_tag::WORKGROUP) {
                    sum += param_distance_eval<tune_key::MEMORY_SPACE_A, T,
                            U>::value;
                    sum += param_distance_eval<tune_key::MEMORY_SPACE_B, T,
                            U>::value;
                    sum += param_distance_eval<tune_key::MEMORY_SPACE_C, T,
                            U>::value;
                }
                if constexpr (tag_ == param_optimizer_tag::KERNEL) {
                    sum += param_distance_eval<tune_key::GLOBAL_KSLICING_RATIO,
                            T, U>::value;
                    sum += param_distance_eval<tune_key::LOCAL_KSLICING_RATIO,
                            T, U>::value;
                }
                sum += param_distance_eval<tune_key::WG_TILE_SHAPE, T,
                        U>::value;
                sum += param_distance_eval<tune_key::WG_TILE_K, T, U>::value;
                if constexpr (tag_ == param_optimizer_tag::WORKGROUP) {
                    sum += param_distance_eval<tune_key::SG_TILE_SHAPE, T,
                            U>::value;
                    sum += param_distance_eval<tune_key::PRE_PROCESSING, T,
                            U>::value;
                }
                sum += param_distance_eval<tune_key::PREFETCH_DISTANCE, T,
                        U>::value;
                sum += param_distance_eval<tune_key::PERIODIC_SYNC_INTERVAL, T,
                        U>::value;
                sum += param_distance_eval<tune_key::MMA_ENGINE, T, U>::value;
                sum += param_distance_eval<tune_key::GPU_ARCH, T, U>::value;
                sum += param_distance_eval<tune_key::EPILOGUE_POLICY, T,
                        U>::value;
                if constexpr (tag_ == param_optimizer_tag::KERNEL) {
                    sum += param_distance_eval<tune_key::DISPATCH_POLICY, T,
                            U>::value;
                    sum += param_distance_eval<tune_key::GROUP_SWIZZLE_POLICY,
                            T, U>::value;
                }
                return sum;
            }
            ();
        };

        template <int opt_val_, typename opt_t_, typename... elems>
        struct finder_impl;

        template <int opt_val_, typename opt_t_>
        struct finder_impl<opt_val_, opt_t_> {
            using type = opt_t_;
            static constexpr int value = opt_val_;
        };

        template <int opt_val_, typename opt_t_, typename elem_,
                typename... elems>
        struct finder_impl<opt_val_, opt_t_, elem_, elems...> {
            static constexpr int can_val
                    = param_distance<dict_t_, elem_>::value;
            using cur_opt_t = typename std::conditional<(can_val < opt_val_),
                    elem_, opt_t_>::type;
            static constexpr int cur_opt_val = const_min(opt_val_, can_val);

            using nxt_result = finder_impl<cur_opt_val, cur_opt_t, elems...>;

            using type = typename nxt_result::type;
            static constexpr int value = nxt_result::value;
        };

        template <typename opt_t_, typename... elems>
        struct finder_impl_helper
            : finder_impl<param_distance<dict_t_, opt_t_>::value, opt_t_,
                      elems...> {};

        using type = typename finder_impl_helper<candidates_t...>::type;
        using fallback_type = fallback_optimizer<dict_t_, type>;
    };
    static constexpr bool use_fallback
            = !(param_optimizer_base::template validate_attribute<dict_t_,
                    typename impl::type>::value);
    using type = typename std::conditional<use_fallback,
            typename impl::fallback_type, impl>::type::type;
};

} // namespace gpu::xetla
