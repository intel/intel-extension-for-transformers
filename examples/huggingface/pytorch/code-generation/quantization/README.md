# Step-by-Step
We provide the inference benchmarking script `run_generation.py` for Starcoder models, [bigcode/starcode](https://huggingface.co/bigcode/starcoder), [bigcode/starcodebase](https://huggingface.co/bigcode/starcoderbase) for code generation tasks, the evaluation part(solution execution) for [MultiPL-E](https://github.com/nuprl/MultiPL-E) requires extra dependencies for some programming languages, we provide a `Dockerfile-multiple` with all dependencies, see [Docker](./Dockerfile-multiple) for more details.


# Prerequisite​
## 1. Create Environment​
Recommend python 3.7 or higher version is recommended. The dependent packages are listed in requirements, please install them as follows,

```shell
git clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers
pip install -r requirements.txt
python setup.py install
```

Required libraries.
```shell
pip install -r requirements.txt
```

We use the gpt_bigcode definition script [modeling_gpt_bigcode.py](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/transformers/modeling/gpt_bigcode/modeling_gpt_bigcode.py) in `run_generation.py`. Here is a little change to success trace.
```diff
# Line 227 in modeling_gpt_bigcode.py on transformers 4.28.1
-      query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
+      query, key, value = self.c_attn(hidden_states).split((self.embed_dim, self.kv_dim, self.kv_dim), dim=2)

# Line 239 in modeling_gpt_bigcode.py on transformers 4.28.1
+      key_value = torch.cat((key, value), dim=-1)


# Line 642 in modeling_gpt_bigcode.py on transformers 4.28.1
-      presents = [] if use_cache else None
+      presents = () if use_cache else None

# Line 682 in modeling_gpt_bigcode.py on transformers 4.28.1
-      presents.append(outputs[1])
+      presents += (outputs[1],)

```


# Run

## 1. Quantization
``` bash
python run_generation.py \
    --model bigcode/starcoder \
    --output_dir "./saved_results" \
    --quantize \
    --sq \
    --alpha 0.7  \
    --ipex \
    --calib_iters 500 \
    --calib_batch_size 1 \
    --dataset "mbpp" \
    --calib_split "test"
```

## 2. Performance

```bash
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
# --int8 is used for int8 model
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_generation.py \
    --model bigcode/starcoder \
    --output_dir "./saved_results" \
    --int8 \
    --ipex \
    --benchmark \
    --batch_size 1
```

## 3. Accuracy
```bash
# --int8 is used for int8 model
python run_generation.py \
    --model bigcode/starcoder \
    --output_dir "./saved_results" \
    --int8 \    
    --ipex \
    --batch_size 20 \
    --accuracy \
    --n_samples 20 \
    --allow_code_execution \
    --temperature 0.2 \
    --do_sample
```
>Note:
please follow the [guide](https://huggingface.co/docs/accelerate/usage_guides/ipex) to set up the configuration if `accelerate launch` is used.

# Docker Run

We provide a Dockerfile based [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/Dockerfile-multiple) to do quantization and evaluation on MultiPL-E inside a docker container.

## Prepare model and datasets
Please ensure that the path of fp32 model, datasets and saved int8 model are accessible by the docker container.

## Building Docker images
Here's how to build a docker image:
```bash
sudo make DOCKERFILE=Dockerfile-multiple all
```
This creates an image called `evaluation-harness-multiple`, and runs a test on it. To skip the test remove `all` form the command.

## Evaluating inside a container
Suppose the fp32 model is `starcoder-3b`, saved quantized model in `saved_results` and do evaluation on `multiple-lua` tasks with:
```
docker run -v $(CURDIR):$(CURDIR) -it /bin/bash
python3 run_generation.py \
    --model $(CURDIR)/starcoder-3b \
    --quantize  \
    --sq \
    --alpha 0.7 \
    --ipex \
    --calib_iters 500 \
    --calib_batch_size 1 \
    --dataset "mbpp" \
    --calib_split "test" \ 
    --output_dir "$(CURDIR)/saved_results" \
    --int8 \
    --accuracy \
    --tasks multiple-py \
    --batch_size 20 \
    --n_samples 20 \
    --allow_code_execution \
    --do_sample \
    --temperature 0.2

```
>Note: "mbpp" is Python programming datasets, please change the calibration dataset to get better results if you want to evaluate on other programming tasks (eg, multiple-lua).

To run the container (here from image `evaluation-harness-multiple`) to quantize and evaluate on `CURDIR`, or another file mount it with -v, specify n_samples and allow code execution with --allow_code_execution (and add the number of problems --limit if it was used during generation):
```bash
docker run -v $(CURDIR):$(CURDIR) \
    -it $(IMAGE_NAME) python3 run_generation.py --model $(CURDIR)/starcoder-3b --quantize   --sq --alpha 0.7 --ipex \
    --calib_iters 5 --calib_batch_size 1 --dataset "mbpp" --calib_split "test" --output_dir "$(CURDIR)/saved_results" \
    --int8 --accuracy --tasks multiple-py  --batch_size 20 --n_samples 20 --allow_code_execution \
    --do_sample --temperature 0.2 --limit 2

```


