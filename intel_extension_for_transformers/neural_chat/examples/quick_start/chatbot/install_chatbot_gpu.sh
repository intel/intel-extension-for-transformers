
pip install intel-extension-for-transformers==1.3.2

git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
git checkout v1.3.2 #

cd intel_extension_for_transformers/neural_chat

# Install neural-chat dependency for Intel GPU
pip install -r requirements_xpu.txt

pip install accelerate==0.28.0

