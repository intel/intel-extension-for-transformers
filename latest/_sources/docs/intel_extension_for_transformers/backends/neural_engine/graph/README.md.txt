# ITREX Graph 

ITREX Graph is an experimental c++ bare metal LLM inference solution that mainly references and borrows from the llama.cpp project. llama.cpp puts almost all core code and kernels in a single file and use a large number of macros, making it difficult for developers to read and modify. The graph has the following features:

- Simple and hierarchical structure, you can add your own high-performance implementation.
- Applied quantization into high 4 bitsfrom int8, higher performance and accuracy compared with the original one.
- Utilized AVX512F instruction set, more instructions support on the way.
- Currently only supports x86 platforms, and initial Intel GPU support.

In short, ITREX Graph is an experimental feature and may keep changing.

### Compile Graph
```shell
mkdir build
cd build
cmake .. -G Ninja
ninja
```

## How to use
### Convert model
Currently, Graph uses the same models as llama.cpp. You can also convert the model yourself
```bash
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model

# convert the pytorch 7B model to llama.cpp format
python scripts/convert_llama.py --outtype f32 --outfile ${output_path}/ne-f32.bin models/7B/

./build/bin/quant_llama --model_file ${output_path}/ne-f32.bin --out_file ${output_path}/ne-q4_j.bin --bits 4 --block_size 32 # bits=4, block_size=128, gemm_isa=vnni means q4_j_vnni_b128(recommend)  

# convert the pytorch gptneox model to llama.cpp format
python scripts/convert_gptneox.py  ${input_model_name_or_path} --outtype f32 --outfile ${output_path}

./build/bin/quant_gptneox --model_file ${output_path}/ne-f32.bin --out_file ${output_path}/ne-q4_j.bin --bits 4

# convert the pytorch mpt model to llama.cpp format
python scripts/convert_mpt.py ${input_model_name_or_path} --outtype f32 --outfile ${output_path}

./build/bin/quant_mpt --model_file ${output_path}/ne-f32.bin --out_file ${output_path}/ne-q4_j.bin --bits 4
# convert the pytorch falcon model to llama.cpp format (0 for fp32 model type)
python scripts/convert_falcon.py ${input_model_name_or_path} --outtype f32 --outfile ${output_path}

./build/bin/quant_falcon --model_file ${output_path}/ne-f32.bin --out_file ${output_path}/ne-q4_j.bin --bits 4

```

### Run Models
Running LLAMA model, for details please refer to [LLaMA model documentation](./application/ChatLLAMA/README.md).

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 ./build/bin/main_llama -m ~/llama.cpp/models/ne-model-q4_j.bin --seed 12 -c 512 -b 1024 -n 256 --keep 48 -t 56 --repeat-penalty 1.0 --color -p "She opened the door and see"
```

Running GPT-NEOX/ MPT / FALCON / GPT-J model, please use `main_gptneox` / `main_mpt` / `main_falcon` / `main_gptj`.

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 ./build/bin/main_gptneox -m ${output_path}/ne-q8.bin --seed 12 -c 512 -b 1024 -n 256 -t 56 --repeat-penalty 1.0 -p "She opened the door and see"
```

for GPT-J, you can also try python binds which is experimental currently:

```bash
cp scripts/gptj_binding.py build
cd build
python gptj_binding.py
```

### Supported model
Now we supports [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), [LLaMA](https://github.com/facebookresearch/llama), [MPT](https://huggingface.co/mosaicml/mpt-7b), [FALCON](https://huggingface.co/tiiuae/falcon-7b), [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)
