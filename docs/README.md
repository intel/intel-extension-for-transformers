# Docs of overview and installation
These documents provide a detailed description of the functionalities of Intel Extension for Transformers.
[architecture.md](architecture.md) offers an overview of Intel Extension for Transformers, encompassing various architectural levels.

You can find install guidelines in [installation.md](installation.md).

# Docs of optimization and inference

Optimization docs: [quantization.md](quantization.md), [pruning.md](pruning.md), [autodistillation.md](autodistillation.md), [distillation.md](distillation.md), 

[export.md](export.md) provides PyTorch to onnx including int8 model.

[get_started.md](get_started.md) guides you to use optimization API and sparse inference. Comprehensive API guidance is available in [API doc](https://intel.github.io/intel-extension-for-transformers/latest/docs/Welcome.html), generated using [api_doc](api_doc) and [build_docs](build_docs). You may disregard the two mentioned folders. 

We also provide examples in [examples.md](examples.md). P.S., smoothquant, weight-only quantization, and cpp inference are not in these documents. SmoothQuant: [smoothquant.md](smoothquant.md), weight-only quantization: [weightonlyquant.md](weightonlyquant.md).

[tutorials/pytorch/](tutorials/pytorch) provides tutorials of many tasks.

# Docs of other functionalities

These utils may help you:
[data_augmentation.md](data_augmentation.md) does NLP datasets augmentation.
[benchmark.md](benchmark.md) is used to measure the model performance. 
[metrics.md](metrics.md) defines which metric will be used to measure the model.
To measure the status of the tuning model, you can refer to [objectives.md](objectives.md).
[pipeline.md](pipeline.md) makes it simple to use any model from the Hub for inference on tasks.

# Docs of contribution and legal

You are invited to commit your code; kindly adhere to the guidelines specified in [contributions.md](contributions.md) and maintain a positive demeanor as outlined in [code_of_conduct.md](code_of_conduct.md). The approvers for each component can be found in [component_owner.md](component_owner.md).


[publication.md](publication.md) lists the publications related to this project, some of them can be found in [pubs/](pubs/).
[release.md](release.md) provides links to all releases. License and legal information can be found in [legal.md](legal.md)
