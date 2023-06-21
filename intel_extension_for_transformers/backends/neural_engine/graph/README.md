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

./build/bin/quant_llama ${output_path}/ne-f32.bin ${output_path}/ne-q4_j.bin 10  #10 for our Q4

# convert the pytorch gptneox model to llama.cpp format
python scripts/convert_gptneox.py  ${input_model_name_or_path} ${output_path} 0

./build/bin/quant_gptneox ${output_path}/ne-f32.bin ${output_path}/ne-q8.bin "q8_0"

# convert the pytorch mpt model to llama.cpp format
python scripts/convert_mpt.py ${input_model_name_or_path} ${output_path} 0

./build/bin/quant_mpt ${output_path}/ne-f32.bin ${output_path}/ne-q8.bin "q8_0"
```

### Run Models
Running LLAMA model, for details please refer to [LLaMA model documentation](./application/ChatLLAMA/README.md).

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 ./build/bin/main_llama -m ~/llama.cpp/models/ne-model-q4_j.bin --seed 12 -c 512 -b 1024 -n 256 --keep 48 -t 56 --repeat_penalty 1.0 --color -p "She opened the door and see"
```

Running GPT-NEOX/ MPT model, please use main_gptneox/ main_mpt.

```bash
OMP_NUM_THREADS=56 numactl -m 0 -C 0-55 ./build/bin/main_gptneox -m ${output_path}/ne-q8.bin -p "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."
```

### Supported model
Now we supports [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), [LLaMA](https://github.com/facebookresearch/llama), [MPT](https://huggingface.co/mosaicml/mpt-7b).
