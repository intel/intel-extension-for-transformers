# Unit Test

## For CPU

Please follow the letacy rule to run unit test for CPU.

## For GPU

The case is drafted for GPU.

Note: it depend on IPEX 2.1.0 for GPU or newer.

|GPU Case|
|-|
|test_weight_only_gpu.py|

### Run Guide

1. Setup Running Environment
```
./setup_env_gpu.sh
```

2. Execute Unit Test Case
```
conda activate env_itrex_gpu
cd tests
python test_weight_only_gpu.py
```
