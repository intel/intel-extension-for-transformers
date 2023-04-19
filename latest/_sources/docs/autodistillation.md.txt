# AutoDistillation Design

1. [AutoDistillation Pipeline](#autodistillation-design)

2. [Framework Class Design](#framework-class-design)

3. [Criterion Class Design](#criterion-class-design)

4. [Usage](#usage)

## AutoDistillation Pipeline

AutoDistillation is composed of three major stages, i.e. Model Exploration, Flash Distillation, and Evaluation.
<br>

In Model Exploration, a search engine will search for a better compressed model from the architecture design space in each iteration.
<br>

Flash Distillation is the stage for training the searched model to discover its potential.
<br>

In Evaluation stage, the trained model will be evaluated to measure its performances (e.g. the prediction accuracy, the hardware performance etc.) in order to select the best model architecture.
<br>

For implementing AutoDistillation, a framework class ***'AutoDistillation'*** is designed for executing the total pipeline, and a criterion class ***'IntermediateLayersKnowledgeDistillationLoss'*** is designed for handling Flash Distillation with the existing Distillation class.
<br>

## Framework Class Design

The framework class is designed for handling the whole pipeline of AutoDistillation.
<br>

It contains a ***search_loop*** method for processing the whole pipeline of iterations for searching the best model architecture.
<br>

Within each iteration, ***model_arch_proposition*** method will propose a promising model architecture for assessing, and ***train_evaluate*** method will train and evaluate this model for measuring its potential.

**Class AutoDistillation**

**Attributes**

**1. search_space** (e.g. {'hidden_size':[64, 128], 'layer_num':[4, 8]})
<br>

**2. model_builder** (the function for building model instance based on the specific sample point in the search space, ***need provided by user***)
<br>

**3. advisor** (the search algorithm instance e.g. Bayesian Optimization, Random Search)
<br>

**4. train_func** (the train function to train the model)
<br>

**5. eval_func** (the evaluation function to evaluate the model)
<br>

**6. config** (the configuration, ***need provided by user***)
<br>

**7. search_result** (store results of the search process)
<br>

**8. best_model_arch** (the function to record the best model architecture ever find)

**Methods**

**1. model_arch_proposition** (propose architecture of the model based on search algorithm for next search iteration)
<br>

**2. search_loop** (begin search iterations)
<br>

**3. train_evaluate** (the process of one search iteration to train and evaluate the model proposed by search algorithm)

## Criterion Class Design
### KnowledgeDistillationLoss

Knowledge distillation is proposed in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). It leverages the logits (the input of softmax in the classification tasks) of teacher and student model to minimize the the difference between their predicted class distributions, this can be done by minimizing the below loss function. 

$$L_{KD} = D(z_t, z_s)$$

Where $D$ is a distance measurement, e.g. Euclidean distance and Kullback–Leibler divergence, $z_t$ and $z_s$ are the logits of teacher and student model, or predicted distributions from softmax of the logits in case the distance is measured in terms of distribution.

**Class KnowledgeDistillationLoss**

**Attributes**

**1. student_model**
<br>

**2. teacher_model**
<br>

**3. temperature** (Hyperparameters that control the entropy of probability distributions. Defaults to 1.0.)
<br>

**4. loss_weights** (weights assigned to each loss term)
<br>

**5. loss_types** (types of each loss term)
<br>

**Methods**

**1. teacher_student_loss_cal** (Define parameters for teacher_student_loss_cal function)
<br>

**2. student_targets_loss_cal** (Define parameters for student_targets_loss_cal function)
<br>

**3. teacher_model_forward** (run forward for the teacher_model)
<br>

**4. loss_cal** (calculate loss)
<br>

**4. loss_cal_sloss** (Calculate all losses between student model and teacher model)
<br>

### IntermediateLayersKnowledgeDistillationLoss
IntermediateLayersKnowledgeDistillationLoss is designed for calculating the knowledge distillation loss of the intermediate layer features.
<br>

To deal the issue of dimension mismatch between the intermediate layer features of the teacher model and the student model, feature_matchers is provided for matching the features dimension.
<br>

For example, the shape of a feature from the teacher model is (8, 512), and the shape of a corresponding feature from the student model is (8, 128), then the feature_matcher will be a linear transformation layer whose weight has a shape of (128, 512).

**Class IntermediateLayersKnowledgeDistillationLoss**

**Attributes**

**1. student_model**
<br>

**2. teacher_model**
<br>

**3. student_features** (store features of intermediate layers of student_model)
<br>

**4. teacher_features** (store features of intermediate layers of teacher_model)
<br>

**5. layer_mappings** (intermediate layers mapping info between student_model and teacher_model)
<br>

**6. layer_output_process** (info for processing layer's output to desired data format)
<br>

**7. loss_weights** (weights assigned to each loss term)
<br>

**8. loss_types** (types of each loss term)
<br>

**9. feature_matchers** (linear transform modules for unmatched features between student_model and teacher_model)

**Methods**

**1. init_loss_funcs** (initialize the loss functions)
<br>

**2. init_feature_matcher** (initialize the feature_matcher instance)
<br>

**3. teacher_model_forward** (run forward for the teacher_model)
<br>

**4. loss_cal** (calculate loss)

## Usage

### Pytorch

#### AutoDistillation API in Trainer

```python
class Trainer:
...
    def autodistillation(self, teacher_model, \
                         model_builder=None, model_cls=None, \
                         train_func=None, eval_func=None):
        agent = AutoDistillation(model_builder, self.autodistillation_config)
        def train_func_builtin(model):
            # initialize flash_distiller 
            flash_distiller ...
            # initialize regular_distiller 
            regular_distiller ...
            scheduler = Scheduler()
            scheduler.combine(flash_distiller, regular_distiller)
            return scheduler()
        def eval_func_builtin(model):
            ...
        agent.train_func = train_func \
            if train_func else train_func_builtin
        agent.eval_func = eval_func \
            if eval_func else eval_func_builtin
        return agent.search_loop()
    
    def model_builder_builtin(self, model_arch_paras, model_cls):
        ...
        return model

### Usage for Trainer.autodistillation
# OOB mode
trainer = Trainer(...)
teacher_model = ...
trainer.autodistillation_config = {
  'search': {
    'search_space': {
        'hidden_size': [128, 246, 384, 512],
        'intra_bottleneck_size': [64, 96, 128, 160],
        'num_attention_heads': [1, 2, 4, 8],
        'intermediate_size': [384, 512, 640],
        'num_feedforward_networks': [2, 4, 6]
      }
    },
  'flash_distillation': {
    'knowledge_transfer': {
      'block_names': 
        ['mobilebert.encoder.layer.{}'.format(i) for i in range(24)],
      'layer_mappings_for_knowledge_transfer': 
        [
          [
            (
              'mobilebert.encoder.layer.{}.attention.self'.format(i), '1',
              'bert.encoder.layer.{}.attention.self'.format(i), '1'
            ),
            (
              'mobilebert.encoder.layer.{}.output'.format(i), 
              'bert.encoder.layer.{}.output'.format(i)
            )
          ] for i in range(24)
        ],
      'loss_types': [['KL', 'MSE'] for i in range(24)],
      },
    'regular_distillation': {
      'layer_mappings_for_knowledge_transfer': [
          [('cls', '0', 'cls', '0')]
        ],
      'loss_types': [['KL']],
      'add_origin_loss': [True],
      },
    }
  }
best_model_arch = trainer.autodistillation(teacher_model, model_cls=AutoModelForPreTraining)

# Advanced mode
def model_builder(model_arch_paras):
    ...
    return model
def train_func(model):
    ...
    return model
def eval_func(model):
    ...
    return metrics
trainer = Trainer(...)
teacher_model = ...
trainer.autodistillation_config = {
  'search': {
    'search_space': {
        'hidden_size': [128, 246, 384, 512],
        'intra_bottleneck_size': [64, 96, 128, 160],
        'num_attention_heads': [1, 2, 4, 8],
        'intermediate_size': [384, 512, 640],
        'num_feedforward_networks': [2, 4, 6]
      },
    'search_algorithm': 'BO',
    'metrics': ['accuracy', 'latency']
    'higher_is_better': [True, False]
    'max_trials': 10
    'seed': 42
    },
  'flash_distillation': {
    'knowledge_transfer': {
      'block_names': 
        ['mobilebert.encoder.layer.{}'.format(i) for i in range(24)],
      'layer_mappings_for_knowledge_transfer': 
        [
          [
            (
              'mobilebert.encoder.layer.{}.attention.self'.format(i), '1',
              'bert.encoder.layer.{}.attention.self'.format(i), '1'
            ),
            (
              'mobilebert.encoder.layer.{}.output'.format(i), 
              'bert.encoder.layer.{}.output'.format(i)
            )
          ] for i in range(24)
        ],
      'loss_types': [['KL', 'MSE'] for i in range(24)],
      'loss_weights': [[0.5, 0.5] for i in range(24)],
      'train_steps': [500 for i in range(24)]
      },
    'regular_distillation': {
      'layer_mappings_for_knowledge_transfer': [
          [('cls', '0', 'cls', '0')]
        ],
      'loss_types': [['KL']],
      'add_origin_loss': [True],
      'train_steps': [25000]
      },
    }
  }
best_model_arch = trainer.autodistillation(teacher_model, model_builder, train_func=train_func, eval_func=eval_func)
```

#### Flash distillation config example

```yaml
model:
  name: mobilebert_distillation
  framework: pytorch

distillation:
  train:
    optimizer:
      SGD:
        learning_rate: 0.001
    criterion:
      IntermediateLayersKnowledgeDistillationLoss:
        layer_mappings_for_knowledge_transfer: [
          ['mobilebert.encoder.layer.0.attention.self', '1', 'bert.encoder.layer.0.attention.self', '1'],
          ['mobilebert.encoder.layer.0.output', 'bert.encoder.layer.0.output'],
                        ]
        loss_types: ['KL', 'MSE']
        loss_weights: [0.5, 0.5]
        add_origin_loss: False
```

#### Regular distillation config example

```yaml
model:
  name: mobilebert_distillation
  framework: pytorch

distillation:
  train:
    optimizer:
      SGD:
        learning_rate: 0.001
    criterion:
      IntermediateLayersKnowledgeDistillationLoss:
        layer_mappings_for_knowledge_transfer: [
          ['mobilebert.output', 'bert.output'],
                        ]
        loss_types: ['KL']
        loss_weights: [1]
        add_origin_loss: True
```

#### AutoDistillation config example

```yaml
model:
  name: MobileBERT_NAS
  framework: pytorch

auto_distillation:
  search:
    search_space: {
        'hidden_size': [128, 246, 384, 512],
        'intra_bottleneck_size': [64, 96, 128, 160],
        'num_attention_heads': [1, 2, 4, 8],
        'intermediate_size': [384, 512, 640],
        'num_feedforward_networks': [2, 4, 6]
        }
    search_algorithm: 'BO'
    metrics: ['accuracy', 'latency']
    higher_is_better: [True, False]
    max_trials: 10
    seed: 42
  flash_distillation:
    block_names: ['mobilebert.encoder.layer.0']
    layer_mappings_for_knowledge_transfer: [
        [
          ['mobilebert.encoder.layer.0.attention.self', '1', 'bert.encoder.layer.0.attention.self', '1'],
          ['mobilebert.encoder.layer.0.output', 'bert.encoder.layer.0.output'],
        ]
                        ]
    loss_types: [['KL', 'MSE']]
    train_steps: [5000]
  regular_distillation:
    layer_mappings_for_knowledge_transfer: [
          ['mobilebert.output', 'bert.output'],
                        ]
    loss_types: ['KL']
    add_origin_loss: True
  train:
    # optional. required if user doesn't provide train_func
    ...
  evaluation:
    # optional. required if user doesn't provide eval_func
    ...
```
Please refer to [example](../examples/huggingface/pytorch/language-modeling/distillation/run_mlm_autodistillation.py) for the details

### Tensorflow
#### AutoDistillation API in optimizer_tf
```python
class TFOptimization:
  ...
def autodistill(
        self,
        autodistillation_config,
        teacher_model: PreTrainedModel,
        model_builder: Optional[Callable] = None,
        model_cls: Optional[Callable] = None,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None
        ):
        self.autodistillation_config = autodistillation_config
        if model_builder is None:
            assert model_cls is not None, "Must specify model_cls to use the built-in " + \
                "model_builder, e.g. model_cls=AutoModelForPreTraining, or you can use " + \
                "the customized model_builder."
            model_builder = partial(self.model_builder_builtin, model_cls=model_cls)
        agent = AutoDistillation(model_builder, self.autodistillation_config, framework='tensorflow')

        def train_func_builtin(model):
          ...
        def eval_func_builtin(model):
          ...
        agent.framework = 'tensorflow'
        agent.train_func = train_func \
            if train_func else train_func_builtin
        agent.eval_func = eval_func \
            if eval_func else eval_func_builtin
        # pylint: disable=E1101
        os.makedirs(self.args.output_dir, exist_ok=True)
        return agent.search(self.args.output_dir, model_cls)

### Usage for TFOptimization.autodistill
optimizer = TFOptimization(
  model=model,
  args=args,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
  compute_metrics=compute_metrics)
autodistillation_config = AutoDistillationConfig(
  search_space={
      'hidden_size': [120, 240],
      'intermediate_size': [256, 512]
  },
  search_algorithm=search_algorithm,
  max_trials=max_trials,
  metrics=[
      metrics.Metric(name="eval_loss", greater_is_better=False)
  ],
  knowledge_transfer=TFDistillationConfig(
      train_steps=[3],
      loss_types=['CE', 'CE'],
      loss_weights=[0.5, 0.5],
      temperature=1.0
  ),
  regular_distillation=TFDistillationConfig(
      train_steps=[3],
      loss_types=['CE', 'CE'],
      loss_weights=[0.5, 0.5],
      temperature=1.0
  )
  )
best_model_archs = optimizer.autodistill(
  autodistillation_config,
  teacher_model,
  model_cls=TFAutoModelForSequenceClassification,
  train_func=None,
  eval_func=None
)
```

#### Distillation config example
```yaml
model:
  name: mobilebert_distillation
  framework: pytorch

distillation:
  train:
    optimizer:
      SGD:
        learning_rate: 0.001
    criterion:
      KnowledgeDistillationLoss:
        temperature: 1.0,
        loss_types: ['CE', 'CE']
        loss_weights: [0.5, 0.5]
```

#### AutoDistillation config example
```yaml
model:
  name: distilbert-base-uncased
  framework: tensorflow

auto_distillation:
  search:
    search_space: {
          'hidden_size': [120, 240],
          'intermediate_size': [256, 512]
      }
    search_algorithm: 'BO'
    metrics: ['accuracy', 'latency']
    max_trials: 10
    seed: 42
  flash_distillation:
    temperature: 1.0
    loss_types: ['CE', 'CE']
    loss_weights: [0.5, 0.5]
    train_steps: [5000]
  train:
    # optional. required if user doesn't provide train_func
    ...
  evaluation:
    # optional. required if user doesn't provide eval_func
    ...
```

Please refer to [example](../examples/huggingface/pytorch/text-classification/distillation/run_glue.py) for the details