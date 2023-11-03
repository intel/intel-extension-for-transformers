#!/bin/bash
source /intel-extension-for-transformers/.github/workflows/script/change_color.sh

cd /intel-extension-for-transformers
export CMAKE_ARGS="-DNE_DNNL_CACHE_DIR=/cache"
$BOLD_YELLOW && echo "---------------- git submodule update --init --recursive -------------" && $RESET
git config --global --add safe.directory "*"
git submodule update --init --recursive


$BOLD_YELLOW && echo "---------------- run python setup.py sdist bdist_wheel -------------" && $RESET
python setup.py bdist_wheel


$BOLD_YELLOW && echo "---------------- pip install binary -------------" && $RESET
pip install dist/intel_extension_for_transformers*.whl
pip list
