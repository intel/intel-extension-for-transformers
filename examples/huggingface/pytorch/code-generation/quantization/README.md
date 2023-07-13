# Step-by-Step
We provide the inference benchmarking script `run_generation.py` for Starcoder models, [bigcode/starcode](https://huggingface.co/bigcode/starcoder), [bigcode/starcodebase](https://huggingface.co/bigcode/starcoderbase) for code generation tasks. 


# Prerequisite​
## 1. Create Environment​
Recommend python 3.7 or higher version is recommended. The dependent packages are listed in requirements, please install them as follows,

```shell
git clone https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit
git checkout maktukmak/starcode_dev
pip install frameworks.ai.nlp-toolkit.intel-nlp-toolkit/
```
Here is how to install intel-extension-for-pytorch from source.
```shell
#  gcc version >= 11
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git submodule sync && git submodule update --init --recursive
python setup.py install
```

Required libraries.
```shell
pip install -r requirements.txt
```

We use the local gpt_bigcode defination script `modeling_gpt_bigcode.py` in `run_generation.py`. Here is a little change to success trace.
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
accelerate launch run_generation.py \
    --model bigcode/starcoderbase \
    --output_dir "./saved_results" \
    --quantize \
    --int8 \
    --sq \
    --alpha 0.5  \
    --ipex \
    --calib_iters 4 \
    --calib_batch_size 8 \
```

## 2. Performance
```bash
accelerate launch run_generation.py \
    --model bigcode/starcoderbase \
    --output_dir "./saved_results" \
    --int8 \
    --ipex \
    --benchmark \
    --batch_size 1 \
```

## 3. Accuracy
Please install [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) before measuring accuracy.
```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness
pip install -e .
```
And then, run the accuracy command.
```bash
accelerate launch run_generation.py \
    --model bigcode/starcoderbase \
    --output_dir "./saved_results" \
    --int8 \
    --ipex \
    --batch_size 10 \
    --accuracy \
    --n_samples 10 \
    --allow_code_execution \
    --do_sample \
```
