
#pip install intel-extension-for-transformers==1.3.2

export cwd=`pwd`
echo $cwd

git clone https://github.com/intel/intel-extension-for-transformers.git ~/itrex

cd ~/itrex
git checkout v1.4rc1

patch -p1  < $cwd/requirements-gpu.patch

pip install -r requirements-gpu.txt
pip install -v .

cd intel_extension_for_transformers/neural_chat

# Install neural-chat dependency for Intel GPU
pip install -r requirements_xpu.txt

pip install accelerate==0.28.0


cd ~/itrex/intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval
pip install -r requirements.txt
#pip install -U langchain-community
