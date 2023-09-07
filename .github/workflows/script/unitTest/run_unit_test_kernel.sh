#!/bin/bash
source /intel-extension-for-transformers/.github/workflows/script/change_color.sh
test_install_backend="true"
LOG_DIR=/intel-extension-for-transformers/log_dir
mkdir -p ${LOG_DIR}
WORKING_DIR="/intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/deprecated"
# -------------------gtest------------------------
function gtest() {
    pip install cmake
    cmake_path=$(which cmake)
    ln -s ${cmake_path} ${cmake_path}3 || true
    cd ${WORKING_DIR}

    mkdir build && cd build && cmake .. -DNE_WITH_SPARSELIB=ON -DNE_WITH_TESTS=ON -DPYTHON_EXECUTABLE=$(which python) && make -j 2>&1 |
        tee -a ${LOG_DIR}/gtest_cmake_build.log
    cp /tf_dataset2/inc-ut/nlptoolkit_ut_model/*.yaml test/gtest/
}

# -------------------engine test-------------------
function engine_test() {
    cd ${WORKING_DIR}/build

    if [[ ${test_install_backend} == "true" ]]; then
        local ut_log_name=${LOG_DIR}/unit_test_engine_gtest_backend_only.log
    else
        local ut_log_name=${LOG_DIR}/unit_test_engine_gtest.log
    fi

    ctest -V -L "engine_test" 2>&1 | tee ${ut_log_name}
    if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "PASSED" ${ut_log_name}) == 0 ] ||
        [ $(grep -c "Segmentation fault" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "core dumped" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "==ERROR:" ${ut_log_name}) != 0 ]; then
        $BOLD_RED && echo "Find errors in engine test, please check the output..." && $RESET
        exit 1
    else
        $BOLD_GREEN && echo "engine test finished successfully!" && $RESET
    fi
}

# ------------------kernel test--------------------
function kernel_test() {
    cd ${WORKING_DIR}/build

    if [[ ${test_install_backend} == "true" ]]; then
        local ut_log_name=${LOG_DIR}/unit_test_kernel_gtest_backend_only.log
    else
        local ut_log_name=${LOG_DIR}/unit_test_kernel_gtest.log
    fi

    ctest -V -L "kernel_test" 2>&1 | tee ${ut_log_name}
    if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "PASSED" ${ut_log_name}) == 0 ] ||
        [ $(grep -c "Segmentation fault" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "core dumped" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "==ERROR:" ${ut_log_name}) != 0 ]; then
        $BOLD_RED && echo "Find errors in kernel test, please check the output..." && $RESET
        exit 1
    else
        $BOLD_GREEN && echo "kernel test finished successfully!" && $RESET
    fi
}

function main() {
    gtest
    kernel_test
    engine_test
}

main
