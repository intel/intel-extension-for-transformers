#!/bin/bash
source /intel-extension-for-transformers/.github/workflows/script/change_color.sh
export COVERAGE_RCFILE="/intel-extension-for-transformers/.github/workflows/script/unitTest/coverage/.engine-coveragerc"
LOG_DIR=/log_dir
mkdir -p ${LOG_DIR}
WORKING_DIR="/intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/deprecated"

# get parameters
PATTERN='[-a-zA-Z0-9_]*='
PERF_STABLE_CHECK=true

for i in "$@"; do
    case $i in
        --test_name=*)
            test_name=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

# -------------------pytest------------------------
function pytest() {
    local coverage_log_dir=$1
    JOB_NAME=unit_test

    cd ${WORKING_DIR}/test/pytest
    engine_path=$(python -c 'import intel_extension_for_transformers; import os; print(os.path.dirname(intel_extension_for_transformers.__file__))')
    engine_path="${engine_path}/llm/runtime"
    echo "engine path is ${engine_path}"
    find . -name "test*.py" | sed 's,\.\/,coverage run --source='"${engine_path}"' --append ,g' | sed 's/$/ --verbose/' >run.sh
    coverage erase

    mkdir -p ${coverage_log_dir}
    ut_log_name=${LOG_DIR}/${JOB_NAME}_pytest.log

    # run UT
    $BOLD_YELLOW && echo "cat run.sh..." && $RESET
    cat run.sh | tee ${ut_log_name}
    $BOLD_YELLOW && echo "------UT start-------" && $RESET
    bash run.sh 2>&1 | tee -a ${ut_log_name}
    $BOLD_YELLOW && echo "------UT end -------" && $RESET

    # run coverage report
    coverage report -m --rcfile=${COVERAGE_RCFILE} | tee ${coverage_log_dir}/coverage.log
    coverage html -d ${coverage_log_dir}/htmlcov --rcfile=${COVERAGE_RCFILE}
    coverage xml -o ${coverage_log_dir}/coverage.xml --rcfile=${COVERAGE_RCFILE}

    # check UT status
    if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ]; then
        $BOLD_RED && echo "Find errors in UT test, please check the output..." && $RESET
        exit 1
    fi
    $BOLD_GREEN && echo "UT finished successfully! " && $RESET
}

# -------------------gtest------------------------
function gtest() {
    pip install cmake
    cmake_path=$(which cmake)
    ln -s ${cmake_path} ${cmake_path}3 || true
    cd ${WORKING_DIR}

    mkdir build && cd build && cmake .. -DNE_WITH_SPARSELIB=ON -DNE_WITH_TESTS=ON -DPYTHON_EXECUTABLE=$(which python) && make -j 2>&1 |
        tee -a ${LOG_DIR}/gtest_cmake_build.log
    cp /tf_dataset2/inc-ut/nlptoolkit_ut_model/*.yaml test/gtest/

    if [[ ${test_install_backend} == "true" ]]; then
        ut_log_name=${LOG_DIR}/unit_test_gtest_backend_only.log
    else
        ut_log_name=${LOG_DIR}/unit_test_gtest.log
    fi

    cd ${WORKING_DIR}/build
    ctest -V -L "engine_test" 2>&1 | tee ${ut_log_name}
    if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "PASSED" ${ut_log_name}) == 0 ] ||
        [ $(grep -c "Segmentation fault" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "core dumped" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "==ERROR:" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "ModuleNotFoundError:" ${ut_log_name}) != 0 ]; then
        $BOLD_RED && echo "Find errors in gtest, please check the output..." && $RESET
        exit 1
    else
        $BOLD_GREEN && echo "gtest finished successfully!" && $RESET
    fi
}

function main() {
    bash /intel-extension-for-transformers/.github/workflows/script/unitTest/env_setup.sh "${WORKING_DIR}/test/pytest"
    cd ${WORKING_DIR}/test/pytest || exit 1
    if [[ $test_name == "PR-test" ]]; then
        pytest "${LOG_DIR}/coverage_pr"
        gtest
    elif [[ $test_name == "baseline" ]]; then
        pytest "${LOG_DIR}/coverage_base"  
    fi
}

main
