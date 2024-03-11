#!/bin/bash
source /intel-extension-for-transformers/.github/workflows/script/change_color.sh
test_install_backend="true"
LOG_DIR=/intel-extension-for-transformers/log_dir
mkdir -p ${LOG_DIR}
WORKING_DIR="/intel-extension-for-transformers/intel_extension_for_transformers/transformers/llm/runtime/neural_speed"

# -------------------Neural Speed Test-------------------
function neuralspeed_test() {
    cd ${WORKING_DIR}/tests
    local ut_log_name=${LOG_DIR}/unit_test_neural_speed.log
    find . -name "test*.py" | sed 's,\.\/,python ,g' | sed 's/$/ --verbose/' >run.sh
    # run UT
    $BOLD_YELLOW && echo "cat run.sh..." && $RESET
    cat run.sh | tee ${ut_log_name}
    $BOLD_YELLOW && echo "------UT start-------" && $RESET
    bash run.sh 2>&1 | tee -a ${ut_log_name}
    $BOLD_YELLOW && echo "------UT end -------" && $RESET

    if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "OK" ${ut_log_name}) == 0 ] ||
        [ $(grep -c "Segmentation fault" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "core dumped" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "==ERROR:" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "ModuleNotFoundError:" ${ut_log_name}) != 0 ] ||
        [ $(grep -c "ImportError:" ${ut_log_name}) != 0 ]; then
        $BOLD_RED && echo "Find errors in engine test, please check the output..." && $RESET
        exit 1
    else
        $BOLD_GREEN && echo "engine test finished successfully!" && $RESET
    fi
}

function main() {
    bash /intel-extension-for-transformers/.github/workflows/script/unitTest/env_setup.sh "${WORKING_DIR}"
    neuralspeed_test
}

main
