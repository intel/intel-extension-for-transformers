# ChatGLM-2

## How to use

### 1. Build Graph
```shell
mkdir build
cd build
cmake .. 
make -j
```

### 2. Convert Models

```bash

# convert the pytorch 7B model to llama.cpp format
python scripts/convert_chatglm.py --outtype q8_0 --outfile ne-chatglm2-q8_0.bin -i THUDM/chatglm2-6b

```

### 3. Run Models
Running ChatGLM-2 model:

```bash
# For example:
./build/bin/main_chatglm -m ./ne-chatglm2-q8_0.bin -p "你好"

# using -i to check the interactive mode
./build/bin/main_chatglm -m ./ne-chatglm2-q8_0.bin -i
```
