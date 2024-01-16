#!/bin/bash

source /intel-extension-for-transformers/.github/workflows/script/change_color.sh
cd /intel-extension-for-transformers
$BOLD_YELLOW && echo "---------------- git submodule update --init --recursive -------------" && $RESET
git config --global --add safe.directory "*"
git submodule update --init --recursive

$BOLD_YELLOW && echo "---------------- install ITREX -------------" && $RESET
export PYTHONPATH=`pwd`
pip list

cd /intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/
if [ -f "requirements.txt" ]; then
    python -m pip install --default-timeout=100 -r requirements.txt
    pip list
else
    echo "Not found requirements.txt file."
fi

cd /intel-extension-for-transformers
log_dir=/intel-extension-for-transformers/.github/workflows/script/formatScan
if [ -f "requirements.txt" ]; then
    python -m pip install --default-timeout=100 -r requirements.txt
    pip list
else
    echo "Not found requirements.txt file."
fi
# install packages
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@83dbfbf6070324f3e5872f63e49d49ff7ef4c9b3
pip install accelerate nlpaug nltk schema optimum-intel==1.11.0 optimum==1.13.3 peft==0.6.2
# install neural-speed
git clone https://github.com/intel/neural-speed.git
cd neural-speed && pip install -r requirements.txt && pip install .
cd ..
pip list | grep neural-speed

echo "[DEBUG] list pipdeptree..."
pip install pipdeptree
pipdeptree

python -m pylint -f json --disable=R,C,W,E1129 \
    --enable=line-too-long \
    --max-line-length=120 \
    --extension-pkg-whitelist=numpy,nltk \
    --ignored-classes=TensorProto,NodeProto \
    --ignored-modules=tensorflow,torch,torch.quantization,torch.tensor,torchvision,mxnet,onnx,onnxruntime,neural_compressor,neural_compressor.benchmark,intel_extension_for_transformers.neural_engine_py,cv2,PIL.Image \
    --ignore-paths=/intel-extension-for-transformers/intel_extension_for_transformers/llm/runtime/graph/ \
    /intel-extension-for-transformers/intel_extension_for_transformers >${log_dir}/pylint.json
exit_code1=$?

python -m pylint -f json --disable=R,C,W,E1129 \
    --enable=line-too-long \
    --max-line-length=120 \
    --disable=no-name-in-module,import-error,no-member,undefined-variable,no-value-for-parameter,unexpected-keyword-arg,not-callable,no-self-argument,too-many-format-args,invalid-unary-operand-type,too-many-function-args \
    --extension-pkg-whitelist=numpy,nltk \
    --ignored-classes=TensorProto,NodeProto \
    --ignored-modules=tensorflow,torch,torch.quantization,torch.tensor,torchvision,mxnet,onnx,onnxruntime,neural_compressor,neural_compressor.benchmark,intel_extension_for_transformers.neural_engine_py,cv2,PIL.Image \
    /intel-extension-for-transformers/intel_extension_for_transformers >> ${log_dir}/pylint.json
exit_code2=$?

$BOLD_YELLOW && echo " -----------------  Current log file output start --------------------------" && $RESET
cat ${log_dir}/pylint.json
$BOLD_YELLOW && echo " -----------------  Current log file output end --------------------------" && $RESET

if [ ${exit_code1} -ne 0 ] || [ ${exit_code2} -ne 0 ]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and view Pylint error details." && $RESET
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, Pylint check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET
exit 0
