# Distillation

1. [Introduction](#introduction)

    1.1 [Distillation](#distillation)

    1.2 [Knowledge Distillation](#knowledge-distillation)

    1.3 [Intermediate Layer Knowledge Distillation](#intermediate-layer-knowledge-distillation)

2. [usage](#usage)

    2.1 [Pytorch Script](#pytorch-script)

    2.2 [Tensorflow Script](#tensorflow-script)

    2.3 [Create an Instance of Metric](#create-an-instance-of-metric)

    2.4 [Create an Instance of Criterion(Optional)](#create-an-instance-of-criterionoptional)

    2.5 [Create an Instance of DistillationConfig](#create-an-instance-of-distillationconfig)

    2.6 [Distill with Trainer](#distill-with-trainer)

## Introduction
### Distillation
Distillation is a widely-used approach to perform network compression, which transfers knowledge from a large model to a smaller one without significant loss of validity. As smaller models are less expensive to evaluate, they can be deployed on less powerful hardware (such as a mobile device). Graph shown below is the workflow of the distillation, the teacher model will take the same input that feed into the student model to produce the output that contains knowledge of the teacher model to instruct the student model.
<br>

<img src="./imgs/Distillation_workflow.png" alt="Architecture" width=700 height=300>
<br>

### Knowledge Distillation
Knowledge distillation is proposed in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). It leverages the logits (the input of softmax in the classification tasks) of teacher and student model to minimize the the difference between their predicted class distributions, this can be done by minimizing the below loss function. 

$$L_{KD} = D(z_t, z_s)$$

Where $D$ is a distance measurement, e.g. Euclidean distance and Kullback–Leibler divergence, $z_t$ and $z_s$ are the logits of teacher and student model, or predicted distributions from softmax of the logits in case the distance is measured in terms of distribution.

### Intermediate Layer Knowledge Distillation

There are more information contained in the teacher model beside its logits, for example, the output features of the teacher model's intermediate layers often been used to guide the student model, as in [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/pdf/1908.09355) and [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984). The general loss function for this approach can be summarized as follow.

$$L_{KD} = \sum\limits_i D(T_t^{n_i}(F_t^{n_i}), T_s^{m_i}(F_s^{m_i}))$$

Where $D$ is a distance measurement as before, $F_t^{n_i}$ the output feature of the $n_i$'s layer of the teacher model, $F_s^{m_i}$ the output feature of the $m_i$'s layer of the student model. Since the dimensions of $F_t^{n_i}$ and $F_s^{m_i}$ are usually different, the transformations $T_t^{n_i}$ and $T_s^{m_i}$ are needed to match dimensions of the two features. Specifically, the transformation can take the forms like identity, linear transformation, 1X1 convolution etc.


## usage
### Pytorch Script:
```python
from intel_extension_for_transformers.optimization import metric, objectives, DistillationConfig, Criterion
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
# Replace transformers.Trainer with NLPTrainer
# trainer = transformers.Trainer(......)
trainer = NLPTrainer(......)
metric = metrics.Metric(name="eval_accuracy")
d_conf = DistillationConfig(metrics=tune_metric)
model = trainer.distill(
    distillation_config=d_conf, teacher_model=teacher_model
)
```

Please refer to [example](../examples/huggingface/pytorch/text-classification/distillation/run_glue.py) for the details.

### Tensorflow Script:
```python
from intel_extension_for_transformers.optimization import (DistillationConfig, metrics)
from intel_extension_for_transformers.optimization.distillation import Criterion

optimizer = TFOptimization(...)
metric_ = metrics.Metric(name="eval_accuracy")
criterion = Criterion(name='KnowledgeLoss',
                    layer_mappings=[['classifier', 'classifier']],
                    loss_types=['CE', 'CE'],
                    loss_weight_ratio=[0.5, 0.5],
                    add_origin_loss=False)
distillation_conf = DistillationConfig(metrics=metric_,
                                        criterion=criterion)
distilled_model = optimizer.distill(
            distillation_config=distillation_conf,
            teacher_model=teacher_model)
```
Please refer to [example](../examples/huggingface/tensorflow/text-classification/distillation/run_glue.py) for the details.
### Create an Instance of Metric
The Metric defines which metric will be used to measure the performance of tuned models.
- example:
    ```python
    metric = metrics.Metric(name="eval_accuracy")
    ```

    Please refer to [metrics document](metrics.md) for the details.

### Create an Instance of Criterion(Optional)
The criterion used in training phase.

- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |name       |String|Name of criterion, like:"KnowledgeLoss", "IntermediateLayersLoss"  |"KnowledgeLoss"|
    |temperature|Float |parameter for KnowledgeDistillationLoss               |1.0             |
    |loss_types|List of string|Type of loss                               |['CE', 'CE']        |
    |loss_weight_ratio|List of float|weight ratio of loss                 |[0.5, 0.5]     |
    |layer_mappings|List|parameter for IntermediateLayersLoss             |[] |
    |add_origin_loss|bool|parameter for IntermediateLayersLoss            |False |

- example:
    ```python
    criterion = Criterion(name='KnowledgeLoss')
    ```

### Create an Instance of DistillationConfig
The DistillationConfig contains all the information related to the model distillation behavior. If you created Metric and Criterion instance, then you can create an instance of DistillationConfig. Metric and pruner_config is optional.

- arguments:
    |Argument   |Type       |Description                                        |Default value    |
    |:----------|:----------|:-----------------------------------------------|:----------------|
    |framework  |string     |which framework you used                        |"pytorch"        |
    |criterion|Criterion |criterion of training                              |"KnowledgeLoss"|
    |metrics    |Metric    |Used to evaluate accuracy of tuning model, no need for NoTrainerOptimizer|None    |

- example:
    ```python
    d_conf = DistillationConfig(metrics=metric, criterion=criterion)
    ```

### Distill with Trainer
- Distill with Trainer
    NLPTrainer inherits from transformers.Trainer, so you can create a trainer as in examples of Transformers. Then you can distill model with trainer.distill function.
    ```python
    model = trainer.distill(
        distillation_config=d_conf, teacher_model=teacher_model
    )
    ```
