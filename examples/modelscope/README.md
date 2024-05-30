# ModelScope with ITREX

Intel® Extension for Transformers(ITREX) support almost all the LLMs in Pytorch format from ModelScope such as phi, Qwen, ChatGLM, Baichuan, gemma, etc.

## Usage Example

ITREX provides a script that demonstrates the use of modelscope. Use numactl to improve performance and run it with the following command:
```bash
OMP_NUM_THREADS=num_cores numactl -l -C 0-num_cores-1 python run_modelscope_example.py --model=qwen/Qwen-7B --prompt=你好
```

## Supported and Validated Models
We have validated the majority of existing models using modelscope==1.13.1:
* [qwen/Qwen-7B](https://www.modelscope.cn/models/qwen/Qwen-7B/summary)
* [ZhipuAI/ChatGLM-6B](https://www.modelscope.cn/models/ZhipuAI/ChatGLM-6B/summary)(transformers=4.33.1)
* [ZhipuAI/chatglm2-6b](https://www.modelscope.cn/models/ZhipuAI/chatglm2-6b/summary)(transformers=4.33.1)
* [ZhipuAI/chatglm3-6b](https://www.modelscope.cn/models/ZhipuAI/chatglm3-6b/summary)(transformers=4.33.1)
* [baichuan-inc/Baichuan2-7B-Chat](https://www.modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary)(transformers=4.33.1)
* [baichuan-inc/Baichuan2-13B-Chat](https://www.modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat/summary)(transformers=4.33.1)
* [LLM-Research/Phi-3-mini-4k-instruct](https://www.modelscope.cn/models/LLM-Research/Phi-3-mini-4k-instruct/summary)
* [LLM-Research/Phi-3-mini-128k-instruct](https://www.modelscope.cn/models/LLM-Research/Phi-3-mini-128k-instruct/summary)
* [AI-ModelScope/gemma-2b](https://www.modelscope.cn/models/AI-ModelScope/gemma-2b/summary)

If you encounter any problems, please let us know.
