#!/bin/bash
source /intel-extension-for-transformers/.github/workflows/script/change_color.sh
test_install_backend="true"
LOG_DIR=/intel-extension-for-transformers/log_dir
mkdir -p ${LOG_DIR}
WORKING_DIR="/intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/graph"

# -------------------LLM Runtime Test-------------------
function llmruntime_test() {
    cd ${WORKING_DIR}
    pip install -r requirements.txt
    cd ${WORKING_DIR}/tests
    local ut_log_name=${LOG_DIR}/unit_test_llm_runtime.log
    find . -name "test*.py" | sed 's,\.\/,python ,g' | sed 's/$/ --verbose/' >run.sh
    # run UT
    $BOLD_YELLOW && echo "cat run.sh..." && $RESET
    cat run.sh | tee ${ut_log_name}
    $BOLD_YELLOW && echo "------UT start-------" && $RESET
    bash run.sh 2>&1 | tee -a ${ut_log_name}
    $BOLD_YELLOW && echo "------UT end -------" && $RESET
}

function main() {
    llmruntime_test
}

main
