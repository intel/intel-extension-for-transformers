#!/bin/bash

source /intel-extension-for-transformers/.github/workflows/script/change_color.sh

pip install cpplint
REPO_DIR=/intel-extension-for-transformers
log_dir=/intel-extension-for-transformers/.github/workflows/script/formatScan
log_path=${log_dir}/cpplint.log
cpplint --extensions cpp,hpp --filter=-build/include_subdir,-build/header_guard --recursive --quiet --linelength=120 ${REPO_DIR}/intel_extension_for_transformers/llm/runtime/deprecated/compile 2>&1 | tee ${log_path}
cpplint --extensions cpp,hpp --filter=-build/include_subdir,-build/header_guard --recursive --quiet --linelength=120 ${REPO_DIR}/intel_extension_for_transformers/llm/runtime/deprecated/executor 2>&1 | tee -a ${log_path}
cpplint --extensions cpp,hpp --filter=-build/include_subdir,-build/header_guard --recursive --quiet --linelength=120 ${REPO_DIR}/intel_extension_for_transformers/llm/runtime/deprecated/test 2>&1 | tee -a ${log_path}
cpplint --extensions cpp,hpp --filter=-build/include_subdir,-build/header_guard --recursive --quiet --linelength=120 ${REPO_DIR}/intel_extension_for_transformers/llm/runtime/graph/application 2>&1 | tee -a ${log_path}
cpplint --extensions cpp,hpp --filter=-build/include_subdir,-build/header_guard --recursive --quiet --linelength=120 ${REPO_DIR}/intel_extension_for_transformers/library/kernels 2>&1 | tee -a ${log_path}
cpplint --extensions cpp,hpp --filter=-build/include_subdir,-build/header_guard --recursive --quiet --linelength=120 ${REPO_DIR}/intel_extension_for_transformers/llm/runtime/graph/models 2>&1 | tee -a ${log_path}
cpplint --extensions cpp,hpp --filter=-build/include_subdir,-build/header_guard --recursive --quiet --linelength=120 ${REPO_DIR}/intel_extension_for_transformers/llm/runtime/graph/vectors 2>&1 | tee -a ${log_path}
if [[ ! -f ${log_path} ]] || [[ $(grep -c "Total errors found:" ${log_path}) != 0 ]]; then
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, check passed!" && $LIGHT_PURPLE && echo "You can click on the artifact button to see the log details." && $RESET
exit 0
