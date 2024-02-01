# Step-by-Step

To get better performance of popular large language models, we recommand using [Neural Speed](https://github.com/intel/neural-speed.git), an innovation library designed to provide the efficient inference of LLMs.Here, we provide the inference script `run_example.py`, and `runtime_acc.py` for accuracy evaluation. 


# Prerequisite​
## 1. Create Environment​
Pytorch and Intel-extension-for-pytorch version 2.1 are required, python version requests equal or higher than 3.9 but lower than 3.11 due to [text evaluation library](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) limitation, the dependent packages are listed in requirements, we recommend install [Neural Speed](https://github.com/intel/neural-speed.git) from source code to use its' latest features.

> Note: To build neural-speed from source code, GCC higher than 10 is required. If you can't upgrade system gcc/g++, here is a solution using conda
> ```bash
> compiler_version==13.1
> conda install --update-deps -c conda-forge gxx==${compiler_version} gcc==${compiler_version} gxx_linux-64==${compiler_version} libstdcxx-ng sysroot_linux-64 -y
> ```

```bash
git clone https://github.com/intel/neural-speed.git
cd neural-speed
pip install -r requirements.txt
python setup.py install
# back to current working directory
cd ..
pip install -r requirements.txt
```

> Note: If `ImportError: /lib64/libstdc++.so.6: version ``GLIBCXX_3.4.29`` not found` error raised when import intel-extension-for-pytorch, it is due to the high gcc library request, there is the solution to find the correct version.
> ```bash
> find $CONDA_PREFIX | grep libstdc++.so.6
> export LD_PRELOAD=<the path of libstdc++.so.6>:${LD_PRELOAD}
> ```


# Run


> Note: Please prepare model and save locally before running performance.


## 1. Performance

``` bash
# int4 with group-size=32
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python runtime_example.py \
    --model_path ./Llama2 \
    --prompt "Once upon a time, there existed a little girl," \
    --max_new_tokens 32 \
    --group_size 128
```

## 2. Accuracy

```bash
# fp32
python runtime_acc.py \
    --model_name ./Llama2 \
    --tasks "lambada_openai"
```
