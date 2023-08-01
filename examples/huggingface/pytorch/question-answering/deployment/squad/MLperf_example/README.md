Step-by-Step
============
This example used [Neural Engine](https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/backends/neural_engine) to get the MLPerf performance. It can test model `Bert Large Squad Sparse` and `MiniLM L12`.
The benchmark was evaluated using a server with two Intel(R) Xeon(R) Platinum 8480+ (Sapphire Rappids) CPUs with 56 cores or two Intel(R) Xeon(R) Platinum 8380 CPU (IceLake) with 40 cores.
| Benchmark      | F1 Score [%] | Machine info  |  Offline Throughput [samples/sec]  |
|:----------------:|:------:|:-------:|:--------:|
| Bert-Large (Sparse 90% structured) | 90.252 | IceLake | 333 |
| Bert-MiniLM L12 | 90.929 | IceLake | 2236.38 |
| Bert-MiniLM L12 | 91.0745 | Sapphire Rappids | 4237.26 |
## Prerequisite

# prepare C++ environment
`GCC greater than 9.2.1` && `cmake greater than 3.18`
prepare intel oneapi
```
source /PATH_TO_ONEAPI/intel/oneapi/compiler/latest/env/vars.sh
```

# generate dataset and model
>> Note: Require a python with tensorflow, six, numpy
```
pip install tensorflow six numpy
cd ./datasets
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ./dev-v1.1.json
python gen_data.py
cd ../
pip install gdown 
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1nXrJvP1_gVk-eBR2FtJ0Cm7mHglK5K69
```

# make neural engine library
```
git clone --recursive https://github.com/intel/intel-extension-for-transformers.git
cp -r intel-extension-for-transformers/intel_extension_for_transformers/ ./ 
pushd intel_extension_for_transformers/backends/neural_engine/
mkdir build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3) -DNE_WITH_SPARSELIB=True && make -j
popd
bash install_third_party.sh
```

# install mlperf loadgen library
```
git clone --branch master --recursive https://github.com/mlcommons/inference.git
pushd ./inference
mkdir loadgen/build/ && cd loadgen/build/
cmake .. && cmake --build .
popd
```

# make mlperf sut example
```
mkdir build && cd build
CC=icx CXX=icpx cmake ..
make -j
cd ..
```

## Run Command
Modify the `user.conf` when you run different models:

+ Set `bert.Offline.target_qps=1000` when you run bert large on ICX machine.

+ Set `bert.Offline.target_qps=2500` when you run minilm on ICX machine.

+ Set `bert.Offline.target_qps=5000` when you run minilm on SPR machine.

### 1. Performance Mode

+ When you run minilm, please also add `--minilm=true` for both performance and accuracy.

+ When you run benchmark on `SPR` machine, please add `--inter_parallel=28 and set --INST_NUM=28` for both perfomance and accuracy.

+ When you run bert large please keep batch size as 4.

+ Please make sure `--model_conf` and `--model_weight` navigate to real model directory.
```
mkdir mlperf_output
GLOG_minloglevel=2  INST_NUM=20 ./build/inference_sut --model_conf=./minilm_mha_ir/conf.yaml --model_weight=./minilm_mha_ir/model.bin  --sample_file=./datasets/ --output_dir=./mlperf_output --mlperf_config=./mlperf.conf --user_config=user.conf
```

### 2. Accuracy Mode
Require transformers in python
```
mkdir mlperf_output
GLOG_minloglevel=2  INST_NUM=20 ./build/inference_sut --model_conf=./minilm_mha_ir/conf.yaml --model_weight=./minilm_mha_ir/model.bin  --sample_file=./datasets/ --output_dir=./mlperf_output --mlperf_config=./mlperf.conf --user_config=user.conf  --accuracy=true
python accuracy-squad.py | tee mlperf_output/accuracy.txt
```
