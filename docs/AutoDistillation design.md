# AutoDistillation Design

## **1. AutoDistillation Pipeline**

AutoDistillation is composed of three major stages, i.e. Model Exploration, Flash Distillation, and Evaluation.
<br>

In Model Exploration, a search engine will search for a better compressed model from the architecture design space in each iteration.
<br>

Flash Distillation is the stage for training the searched model to discover its potential.
<br>

In Evaluation stage, trained model will be evaluated to measure its performances (e.g. the prediction accuracy, the hardware performance etc.) inorder to select the best model architecture.
<br>

For implementing AutoDistillation, a framework class ***'AutoDistillation'*** is designed for excuting the total pipeline, and a criterion class ***'IntermediateLayersKnowledgeDistillationLoss'*** is designed for handling Flash Distillation with existing Distillation class.
<br>

## **2. Framework Class Design**

Framework class is designed for handling the whole pipeline of AutoDistillation.
<br>

It contains a ***search_loop*** method for processing the whole pipeline of iterations for searching the best model architecture.
<br>

Within each iteration, ***model_arch_proposition*** method will propose a promising model architecture for assessing, and ***train_evaluate*** method will train and evaluate this model for measuring its potential.

### **Class AutoDistillation**

#### **Attributes**

**1. search_space** (e.g. {'hidden_size':[64, 128], 'layer_num':[4, 8]})
<br>

**2. model_builder** (function for building model instance based on the specific sample point in the search space, ***need provided by user***)
<br>

**3. advisor** (search algorithm instance e.g. Bayesian Optimization, Random Search)
<br>

**4. train_func** (train function to train the model)
<br>

**5. eval_func** (evaluation function to evaluate the model)
<br>

**6. config** (configuration, ***need provided by user***)
<br>

**7. search_result** (store results of the search process)
<br>

**8. best_model_arch** (record the best model architecture ever find)

#### **Methods**

**1. model_arch_proposition** (propose architecture of the model based on search algorithm for next search iteration)
<br>

**2. search_loop** (begin search iterations)
<br>

**3. train_evaluate** (process of one search iteration to train and evaluate the model proposed by search algorithm)

## **3. Criterion Class Design**
***IntermediateLayersKnowledgeDistillationLoss*** is designed for calculating the knowledge distillation loss of the intermediate layer features.
<br>

To deal the issue of dimension mismatch between the intermediate layer features of the teacher model and the student model, feature_matchers is provided for matching the features dimension.
<br>

For example, shape of a feature from the teacher model is (8, 512), shape of a corresponding feature from the student model is (8, 128), then feature_matcher will be a linear transformation layer whose weight has a shape of (128, 512).

### **Class IntermediateLayersKnowledgeDistillationLoss**

#### **Attributes**

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

#### **Methods**

**1. init_loss_funcs** (initialize the loss functions)
<br>

**2. init_feature_matcher** (initialize the feature_matcher instance)
<br>

**3. teacher_model_forward** (run forward for teacher_model)
<br>

**4. loss_cal** (calculate loss)

## **4. Usage**

### **AutoDistillation API in Trainer**

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

### **flash distillation config example**

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

### **regular distillation config example**

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

### **AutoDistillation config example**

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

## **5. Problems**
- [x] 1. Delete config 
- [x] 2. OOB mode
- [x] 3. Advanced mode