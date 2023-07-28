#!/bin/bash
source /intel-extension-for-transformers/.github/workflows/script/change_color.sh
export COVERAGE_RCFILE="/intel-extension-for-transformers/.github/workflows/script/unitTest/coverage/.optimize-coveragerc"
LOG_DIR=/log_dir
mkdir -p ${LOG_DIR}

function pytest() {
    local coverage_log_dir=$1
    mkdir -p ${coverage_log_dir}
    pip install --no-cache-dir protobuf==3.20.0

    cd /intel-extension-for-transformers/tests || exit 1
    JOB_NAME=unit_test
    ut_log_name=${LOG_DIR}/${JOB_NAME}.log
    export GLOG_minloglevel=2

    itrex_path=$(python -c 'import intel_extension_for_transformers; import os; print(os.path.dirname(intel_extension_for_transformers.__file__))')
    find . -name "test*.py" | sed 's,\.\/,coverage run --source='"${itrex_path}"' --append ,g' | sed 's/$/ --verbose/' >run.sh
    coverage erase

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

function re_install_packages() {
    local package_name=$1
    echo "re-install ${package_name} resolve the issue..."
    pip uninstall ${package_name} -y
    pip install --no-cache-dir ${package_name}
}

function install_itrex_base() {
    pip uninstall intel_extension_for_transformers -y

    cd /intel-extension-for-transformers
    git config --global --add safe.directory "*"
    git fetch
    git checkout main

    #bash /intel-extension-for-transformers/.github/workflows/script/install_binary.sh
    #pip install --no-cache-dir protobuf==3.20.0
    $BOLD_YELLOW && echo "---------------- git submodule update --init --recursive -------------" && $RESET
    git config --global --add safe.directory "*"
    git submodule update --init --recursive
    $BOLD_YELLOW && echo "---------------- pip install binary -------------" && $RESET
    pip install .
}

function main() {
    bash /intel-extension-for-transformers/.github/workflows/script/unitTest/env_setup.sh
    pytest "${LOG_DIR}/coverage_pr"
    install_itrex_base
    pytest "${LOG_DIR}/coverage_base"
}

main
