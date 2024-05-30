# ModelScope with ITREX

Intel extension for transformers(ITREX) support almost all the LLMs in Pytorch format from ModelScope such as phi,Qwen,ChatGLM,Baichuan,gemma,etc.

## Usage Example

ITREX provides a script that demonstrates the vLLM inference acceleration. Run it with the following command:
```bash
numactl -m 0 -C 0-55 python run_modelscope_example.py --model_path=qwen/Qwen-7B --prompt=你好
```

## Supported and Validated Models
We have validated the majority of existing models using modelscope==1.13.1:
* [qwen/Qwen-7B](https://www.modelscope.cn/models/qwen/Qwen-7B/summary)
* [ZhipuAI/ChatGLM-6B](https://www.modelscope.cn/models/ZhipuAI/ChatGLM-6B/summary)
* [ZhipuAI/chatglm2-6b](https://www.modelscope.cn/models/ZhipuAI/chatglm2-6b/summary)
* [ZhipuAI/chatglm3-6b](https://www.modelscope.cn/models/ZhipuAI/chatglm3-6b/summary)
* [baichuan-inc/Baichuan2-7B-Chat](https://www.modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary)
* [baichuan-inc/Baichuan2-13B-Chat](https://www.modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat/summary)
* [AI-ModelScope/phi-2](https://www.modelscope.cn/models/AI-ModelScope/phi-2/summary)
* [AI-ModelScope/gemma-2b](https://www.modelscope.cn/models/AI-ModelScope/gemma-2b/summary)

If you encounter any problems, please let us know.
