# Streaming LLM

Streaming LLM is an useful approach in long context generation and multi-round chatting scenario. In this example, we will show how to enable it in Intel Guadi device. More technical details, please refet to [paper](https://arxiv.org/abs/2309.17453).

> Note: Only supports Llama model architecture and one-single HPU card.

## Create Environment
Validate in Habana version 1.15.1 with its Pytorch-2.2.0 docker.
​
```shell
# start docker
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v <workfloder>:<docker workfolder>  --name "streaming_llm" vault.habana.ai/gaudi-docker/1.15.1/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest

# install related packages in docker
pip install git+https://github.com/huggingface/optimum-habana.git@753da20f98ad6f874075701995428072159ba600
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex && git switch yzt/gaudi_streaming_llm
python setup.py install
```

## Run
We provide [01-ai/Yi-34B](https://huggingface.co/01-ai/Yi-34B) as an model example by default for demonstrating streaming outputs.

1. bf16 data type:
```shell
bash run_yi_34b_bf16_streaming.sh
```

2. fp8 data type:
```shell
bash run_yi_34b_fp8_streaming.sh
```

You can change the input args values (like `attention_sink_window_size`, `num_sample` for fp8 calibration, etc.) to implement more experiments.

## Evaluation (ppl)

We follow the one token by one token ppl evaluation way in [streaming llm](https://github.com/mit-han-lab/streaming-llm/blob/main/examples/eval_long_ppl.py#L81-L91).

1. test `llama2-13b` ppl to align paper's result.

```shell
HF_TOKEN=<your HF account token> MODEL=meta-llama/Llama-2-13b-hf bash eval_bf16_streaming.sh
```

2. test another model with bf16 data type

```shell
MODEL=<HF model name or local path> bash eval_bf16_streaming.sh
```

3. test model with fp8 data type

```shell
MODEL=<HF model name or local path> bash eval_fp8_streaming.sh
```

The shell script will plot `ppl_memory` and `ppl_latency` figures with svg format for visualization after ppl evaluation.
