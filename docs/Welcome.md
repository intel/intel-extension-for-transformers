# Intel® Extension for Transformers: Accelerating Transformer-based Models on Intel Platforms
 Intel® Extension for Transformers is an innovative toolkit to accelerate Transformer-based models on Intel platforms, in particular effective on 4th Intel Xeon Scalable processor Sapphire Rapids (codenamed [Sapphire Rapids](https://www.intel.com/content/www/us/en/products/docs/processors/xeon-accelerated/4th-gen-xeon-scalable-processors.html)). The toolkit provides the key features and examples as below:


*  Seamless user experience of model compressions on Transformers-based models by extending [Hugging Face transformers](https://github.com/huggingface/transformers) APIs and leveraging [Intel® Neural Compressor](https://github.com/intel/neural-compressor)


*  Advanced software optimizations and unique compression-aware runtime (released with NeurIPS 2022's paper [Fast Distilbert on CPUs](https://arxiv.org/abs/2211.07715) and [QuaLA-MiniLM: a Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114), and NeurIPS 2021's paper [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754))


*  Accelerated end-to-end Transformer-based applications such as [Stable Diffusion](./examples/optimization/pytorch/huggingface/textual_inversion), [GPT-J-6B](./examples/optimization/pytorch/huggingface/language-modeling/inference/README.md#GPT-J), [BLOOM-176B](./examples/optimization/pytorch/huggingface/language-modeling/inference/README.md#BLOOM-176B), [T5](https://github.com/intel/intel-extension-for-transformers/blob/main/examples/optimization/pytorch/huggingface/summarization/quantization), and [SetFit](./docs/tutorials/pytorch/text-classification/SetFit_model_compression_AGNews.ipynb)       


## Selected Publications/Events
* Blog published on Medium: [MLefficiency — Optimizing transformer models for efficiency](https://medium.com/@kawapanion/mlefficiency-optimizing-transformer-models-for-efficiency-a9e230cff051) (Dec 2022)
* NeurIPS'2022: [Fast Distilbert on CPUs](https://arxiv.org/abs/2211.07715) (Nov 2022)
* NeurIPS'2022: [QuaLA-MiniLM: a Quantized Length Adaptive MiniLM](https://arxiv.org/abs/2210.17114) (Nov 2022)
* Blog published by Cohere: [Top NLP Papers—November 2022](https://txt.cohere.ai/top-nlp-papers-november-2022/) (Nov 2022)
* Blog published by Alibaba: [Deep learning inference optimization for Address Purification](https://zhuanlan.zhihu.com/p/552484413) (Aug 2022)
* NeurIPS'2021: [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754) (Nov 2021)
