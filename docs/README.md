# Documentation Overview and Installation


You can find install guidelines in [installation.md](installation.md).

These documents offer a comprehensive description of Intel Extension for Transformers:


- [architecture.md](architecture.md): Provides an overview of Intel Extension for Transformers, covering various architectural levels.
- [installation.md](installation.md): Contains installation guidelines.

# Optimization and Inference Documentation

**Optimization Documentation**:

- [quantization.md](quantization.md): Covers quantization techniques.
- [pruning.md](pruning.md): Details the pruning process.
- [autodistillation.md](autodistillation.md) and [distillation.md](distillation.md): Discuss distillation methods.
- [export.md](export.md): Explains PyTorch to ONNX, including the int8 model.
- [get_started.md](get_started.md): Guides you through using the optimization API and sparse inference.


We also provide examples in [examples.md](examples.md). P.S., smoothquant, weight-only quantization, and cpp inference are not in these documents. SmoothQuant: [smoothquant.md](smoothquant.md), weight-only quantization: [weightonlyquant.md](weightonlyquant.md).

**API Documentation**:


For comprehensive API guidance, visit the [API doc](https://intel.github.io/intel-extension-for-transformers/latest/docs/Welcome.html), generated using [api_doc](api_doc) and [build_docs](build_docs). You can disregard the two mentioned folders.

**Examples**:

Practical examples can be found in [examples.md](examples.md). Note that `smoothquant`, `weight-only quantization`, and `cpp inference` are not covered in these documents. For specific information on these topics:

- [smoothquant.md](smoothquant.md): SmoothQuant
- [weightonlyquant.md](weightonlyquant.md): Weight-Only Quantization

**Tutorials**:

Explore various tutorials for different tasks in [tutorials/pytorch/](tutorials/pytorch).


[publication.md](publication.md) lists the publications related to this project, some of them can be found in [pubs/](pubs/).
[release.md](release.md) provides links to all releases. License and legal information can be found in [legal.md](legal.md)

# Other Functionalities Documentation

**Utilities**:

- [data_augmentation.md](data_augmentation.md): Describes NLP dataset augmentation.
- [benchmark.md](benchmark.md): Explains how to measure model performance.
- [metrics.md](metrics.md): Defines the metrics used for model measurement.
- To measure the status of the tuning model, refer to [objectives.md](objectives.md).
- [pipeline.md](pipeline.md): Simplifies the process of using any model from the Hub for inference on tasks.

# Contribution and Legal Documentation

You are invited to contribute your code. Kindly adhere to the guidelines specified in [contributions.md](contributions.md) and maintain a positive demeanor as outlined in [code_of_conduct.md](code_of_conduct.md). For component-specific approvers, refer to [component_owner.md](component_owner.md).

**Publication**:
[publication.md](publication.md) lists publications related to this project, some of which can be found in [pubs/](pubs/).

**Release Information**:
Access links to all releases in [release.md](release.md).

**Legal Information**:
For license and legal details, please consult [legal.md](legal.md).
