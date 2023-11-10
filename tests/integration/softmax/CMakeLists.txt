set(SRC_FWD softmax_fwd.cpp) 
set(SRC_BWD softmax_bwd.cpp)

add_integration_test(softmax_fwd_ut ${SRC_FWD} softmax_fwd_kernel.hpp)
add_integration_test(softmax_bwd_ut ${SRC_BWD} softmax_bwd_kernel.hpp)

