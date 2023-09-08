#!/bin/bash
source /intel-extension-for-transformers/.github/workflows/script/change_color.sh

pip install clang-format==14.0.0
log_dir=/intel-extension-for-transformers/.github/workflows/script/formatScan
log_path=${log_dir}/clangformat.log

cd /intel-extension-for-transformers
git config --global --add safe.directory "*"

cd /intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/graph
python scripts/clang-format.py

echo "run git diff"
git diff 2>&1 | tee -a ${log_path}

if [[ ! -f ${log_path} ]] || [[ $(grep -c "diff" ${log_path}) != 0 ]]; then
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, check passed!" && $LIGHT_PURPLE && echo "You can click on the artifact button to see the log details." && $RESET
exit 0
