# Documentation Overview and Installation

These documents offer a comprehensive description of Intel Extension for Transformers:

- [architecture.html](architecture.html): Provides an overview of Intel Extension for Transformers, covering various architectural levels.
- [installation.html](installation.html): Contains installation guidelines.

# Optimization and Inference Documentation

**Optimization Documentation**:

- [quantization.html](quantization.html): Covers quantization techniques.
- [pruning.html](pruning.html): Details the pruning process.
- [autodistillation.html](autodistillation.html) and [distillation.html](distillation.html): Discuss distillation methods.
- [export.html](export.html): Explains PyTorch to ONNX, including the int8 model.
- [get_started.html](get_started.html): Guides you through using the optimization API and sparse inference.

**API Documentation**:

For comprehensive API guidance, visit the [API doc](https://intel.github.io/intel-extension-for-transformers/latest/docs/Welcome.html), generated using [api_doc](api_doc) and [build_docs](build_docs). You can disregard the two mentioned folders.

**Examples**:

Practical examples can be found in [examples.html](examples.html). Note that `smoothquant`, `weight-only quantization`, and `cpp inference` are not covered in these documents. For specific information on these topics:

- [smoothquant.html](smoothquant.html): SmoothQuant
- [weightonlyquant.html](weightonlyquant.html): Weight-Only Quantization

**Tutorials**:

Explore various tutorials for different tasks in [tutorials/pytorch/](tutorials/pytorch).

# Other Functionalities Documentation

**Utilities**:

- [data_augmentation.html](data_augmentation.html): Describes NLP dataset augmentation.
- [benchmark.html](benchmark.html): Explains how to measure model performance.
- [metrics.html](metrics.html): Defines the metrics used for model measurement.
- To measure the status of the tuning model, refer to [objectives.html](objectives.html).
- [pipeline.html](pipeline.html): Simplifies the process of using any model from the Hub for inference on tasks.

# Contribution and Legal Documentation

You are invited to contribute your code. Kindly adhere to the guidelines specified in [contributions.html](contributions.html) and maintain a positive demeanor as outlined in [code_of_conduct.html](code_of_conduct.html). For component-specific approvers, refer to [component_owner.html](component_owner.html).

**Publication**:
[publication.html](publication.html) lists publications related to this project, some of which can be found in [pubs/](pubs/).

**Release Information**:
Access links to all releases in [release.html](release.html).

**Legal Information**:
For license and legal details, please consult [legal.html](legal.html).
