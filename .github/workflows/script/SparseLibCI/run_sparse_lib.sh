#!/bin/bash
export COVERAGE_RCFILE="/intel-extension-for-transformers/.github/workflows/script/unitTest/coverage/.coveragerc"

output_log_dir="/intel-extension-for-transformers/benchmark_log"
mkdir ${output_log_dir}
mkdir ${output_log_dir}/cur
cur_dir=${output_log_dir}/cur
ref_dir=${output_log_dir}/ref
cp /intel-extension-for-transformers/.github/workflows/script/SparseLibCI/generate_sparse_lib.sh /generate_sparse_lib.sh
cp /intel-extension-for-transformers/.github/workflows/script/SparseLibCI/generate_sparse_lib.py /generate_sparse_lib.py
pip install cmake

rm -rf /intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/build
cd /intel-extension-for-transformers

git config --global --add safe.directory "*"
git fetch
git submodule update --init --recursive
cd /intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime
mkdir build && cd build

CC=gcc CXX=g++ cmake .. -DNE_WITH_SPARSELIB=ON -DNE_WITH_TESTS=ON -DNE_WITH_SPARSELIB_BENCHMARK=ON -DPYTHON_EXECUTABLE=$(which python)
make -j
cd bin
bash /intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/build/bin/ci/run_ci.sh $cur_dir

for caselog in $(find $cur_dir/*); do
    case_name=$(echo $caselog | sed -e 's/\.log$//')
    $BOLD_YELLOW && echo "[VAL INFO] write summary, case_name=$case_name" && $RESET
    bash /intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/test/kernels/benchmark/ci/to_summary.sh $caselog | tee -a "${case_name}_summary.log"
done

cd /intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime
mkdir refer && cd refer
git checkout -b refer origin/main
git pull
CC=gcc CXX=g++ cmake .. -DNE_WITH_SPARSELIB=ON -DNE_WITH_TESTS=ON -DNE_WITH_SPARSELIB_BENCHMARK=ON -DPYTHON_EXECUTABLE=$(which python)
make -j
cd bin

mkdir -p ${output_log_dir}/ref
bash /intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/refer/bin/ci/run_ci.sh $ref_dir

for caselog in $(find $ref_dir/*); do
    case_name=$(echo $caselog | sed -e 's/\.log$//')
    $BOLD_YELLOW && echo "[VAL INFO] write summary, case_name=$case_name" && $RESET
    bash /intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/test/kernels/benchmark/ci/to_summary.sh $caselog | tee -a "${case_name}_summary.log"
done
