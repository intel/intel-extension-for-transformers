#!/bin/bash

source /intel-extension-for-transformers/.github/workflows/script/change_color.sh
export COVERAGE_RCFILE="/intel-extension-for-transformers/.github/workflows/script/unitTest/coverage/.neural-chat-coveragerc"
LOG_DIR=/log_dir
mkdir -p ${LOG_DIR}
WORKING_DIR="/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tests/ci/"
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

function pytest() {
    local coverage_log_dir=$1
    mkdir -p ${coverage_log_dir}

    cd ${WORKING_DIR} || exit 1
    JOB_NAME=unit_test
    ut_log_name=${LOG_DIR}/${JOB_NAME}.log
    export GLOG_minloglevel=2

    itrex_path=$(python -c 'import intel_extension_for_transformers; import os; print(os.path.dirname(intel_extension_for_transformers.__file__))')
    find . -name "test*.py" | sed 's,\.\/,coverage run --source='"${itrex_path}"' --append ,g' | sed 's/$/ --verbose/' >> run.sh
    sort run.sh -o run.sh
    coverage erase

    # run UT
    sleep 1
    $BOLD_YELLOW && echo "cat run.sh..." && $RESET
    cat run.sh | tee ${ut_log_name}
    sleep 1
    $BOLD_YELLOW && echo "------UT start-------" && $RESET
    bash -x run.sh 2>&1 | tee -a ${ut_log_name}
    sleep 1
    $BOLD_YELLOW && echo "------UT end -------" && $RESET

    # run coverage report
    coverage report -m --rcfile=${COVERAGE_RCFILE} | tee ${coverage_log_dir}/coverage.log
    coverage html -d ${coverage_log_dir}/htmlcov --rcfile=${COVERAGE_RCFILE}
    coverage xml -o ${coverage_log_dir}/coverage.xml --rcfile=${COVERAGE_RCFILE}

    # check UT status
    if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ]; then
        $BOLD_RED && echo "Find errors in UT, please search [FAILED]..." && $RESET
        exit 1
    fi
    if [ $(grep -c "ModuleNotFoundError:" ${ut_log_name}) != 0 ]; then
        $BOLD_RED && echo "Find errors in UT, please search [ModuleNotFoundError:]..." && $RESET
        exit 1
    fi
    if [ $(grep -c "core dumped" ${ut_log_name}) != 0 ]; then
        $BOLD_RED && echo "Find errors in UT, please search [core dumped]..." && $RESET
        exit 1
    fi
    if [ $(grep -c "OK" ${ut_log_name}) == 0 ]; then
        $BOLD_RED && echo "No pass case found, please check the output..." && $RESET
        exit 1
    fi
    if [ $(grep -c "==ERROR:" ${ut_log_name}) != 0 ]; then
       $BOLD_RED && echo "ERROR found in UT, please check the output..." && $RESET
        exit 1
    fi 
    if [ $(grep -c "Segmentation fault" ${ut_log_name}) != 0 ]; then
       $BOLD_RED && echo "Segmentation Fault found in UT, please check the output..." && $RESET
        exit 1
    fi  
    $BOLD_GREEN && echo "UT finished successfully! " && $RESET
}

function main() {
    bash /intel-extension-for-transformers/.github/workflows/script/unitTest/env_setup.sh ${WORKING_DIR}
    apt-get update
    apt-get install ffmpeg -y
    apt-get install lsof
    apt-get install libgl1
    apt-get install -y libgl1-mesa-glx
    apt-get install -y libgl1-mesa-dev
    apt-get install libsm6 libxext6 -y
    wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
    dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
    python -m pip install --upgrade --force-reinstall torch
    pip install paddlepaddle==2.4.2 paddlenlp==2.5.2 paddlespeech==1.4.1 paddle2onnx==1.0.6
    cd ${WORKING_DIR} || exit 1
    echo "test on ${test_name}"
    if [[ $test_name == "PR-test" ]]; then
        pytest "${LOG_DIR}/coverage_pr"
    elif [[ $test_name == "baseline" ]]; then
        pytest "${LOG_DIR}/coverage_base"  
    fi
}
main
