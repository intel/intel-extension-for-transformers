#!/bin/bash
source /intel-extension-for-transformers/.github/workflows/script/change_color.sh
export COVERAGE_RCFILE="/intel-extension-for-transformers/.github/workflows/script/unitTest/coverage/.neural-chat-coveragerc"
LOG_DIR=/log_dir
mkdir -p ${LOG_DIR}
WORKING_DIR="/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/tests"
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
    find . -name "test*.py" | sed 's,\.\/,coverage run --source='"${itrex_path}"' --append ,g' | sed 's/$/ --verbose/' >run.sh
    echo -e '
# Kill the neuralchat server processes
ports="7000 8000 9000"
# Loop through each port and find associated PIDs
for port in $ports; do
    # Use lsof to find the processes associated with the port
    pids=$(lsof -ti :$port)

    if [ -n "$pids" ]; then
        echo "Processes running on port $port: $pids"
        # Terminate the processes gracefully with SIGTERM
        kill $pids
        echo "Terminated processes on port $port."
    else
        echo "No processes found on port $port."
    fi
done
' >> run.sh
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

function main() {
    bash /intel-extension-for-transformers/.github/workflows/script/unitTest/env_setup.sh
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
    pip install git+https://github.com/UKPLab/sentence-transformers.git
    pip install git+https://github.com/Muennighoff/sentence-transformers.git@sgpt_poolings_specb
    pip install --upgrade git+https://github.com/UKPLab/sentence-transformers.git
    pip install -U sentence-transformers
    cd ${WORKING_DIR} || exit 1
    if [ -f "requirements.txt" ]; then
        python -m pip install --default-timeout=100 -r requirements.txt
        pip list
    else
        echo "Not found requirements.txt file."
    fi
    echo "test on ${test_name}"
    if [[ $test_name == "PR-test" ]]; then
        pytest "${LOG_DIR}/coverage_pr"
    elif [[ $test_name == "baseline" ]]; then
        pytest "${LOG_DIR}/coverage_base"  
    fi
}
main
