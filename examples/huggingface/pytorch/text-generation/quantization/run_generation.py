import torch
import transformers
import sys
import time
from tqdm import tqdm
import os
from torch.nn.functional import pad
import logging
from dataclasses import field, dataclass
from typing import Optional
from intel_extension_for_transformers.optimization import (NoTrainerOptimizer, QuantizationConfig, OptimizedModel)
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    model_type: Optional[str] = field(
        default=None, metadata={"help": "The model group."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    onnx: bool = field(
        default=True, metadata={"help": "convert PyTorch model to ONNX"}
    )
    no_cache: bool = field(
        default=False, metadata={"help": "Whether or not to cache"}
    )
    bootstrap_iters: int = field(
        default=100000, metadata={"help": "Number of iterations for bootstrap statistics"}
    )
    iters: int = field(
        default=100,
        metadata={
            "help": "The inference iterations to run for benchmark."
        },
    )
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tasks: Optional[str] = field(
        default="lambada", metadata={"help": "The name of the dataset to use (restrict to the task list)."}
    )
    batch_size: Optional[int] = field(default=8, metadata={"help": "The batch size"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )


@dataclass
class OptimizationArguments:
    """
    Arguments pertaining to what type of optimization we are going to apply on the model.
    """

    tune: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply quantization."},
    )
    quantization_approach: Optional[str] = field(
        default="PostTrainingStatic",
        metadata={"help": "Quantization approach. Supported approach are PostTrainingStatic, "
                  "PostTrainingDynamic and QuantizationAwareTraining."},
    )
    is_relative: Optional[bool] = field(
        default=True,
        metadata={"help": "Metric tolerance mode, True is for relative, otherwise for absolute."},
    )
    max_train_sample: Optional[int] = field(
        default=None,
        metadata={"help": "The batch size of the train set."},
    )
    max_cali_sample: Optional[int] = field(
        default=None,
        metadata={"help": "The batch size of the calibration set."},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "run benchmark."})
    mix_precision: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply quantization."},
    )
    int8: bool = field(
        default=False,
        metadata={"help":"run benchmark."})
    show: bool = field(
        default=False,
        metadata={"help":"present the generation example."})
    accuracy_only: bool = field(
        default=False,
        metadata={"help":"Whether to only test accuracy for model tuned by Neural Compressor."})


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, device='cpu'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.dataloader = NLPDataloader(dataset, tokenizer, batch_size, device, 195)
        self.batch_size = batch_size

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        batch_time = AverageMeter('Time', ':6.3f')
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for index, (batched_input, label, batched_index) in enumerate(tqdm(self.dataloader)):
            input_ids = batched_input
            start = time.time()
            outputs = model(input_ids)
            batch_time.update(time.time() - start)
            # print()
            # print('Time usage: ', time.time()-start)
            last_token_logits = outputs[0][:, batched_index, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if (index+1) % 10 == 0:
                accu = hit / total
                print('#############')
                print('Temporary Accu:', accu, flush=True)
                print('#############')
        accu = hit / total
        print('Batch size = {}'.format(self.batch_size))
        print('Latency: %.3f ms' % (batch_time.avg / self.batch_size * 1000))
        print('Throughput: %.3f images/sec' % (self.batch_size / batch_time.avg))
        return accu

class NLPDataloader():
    def __init__(self, dataset, tokenizer, batch_size=8,
                        device='cpu', max_length=195, calibration=False
        ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        import math
        self.length = math.ceil(len(dataset) / self.batch_size)
        # for evaluation, max_length is 195, while for calibration with train dataset, it's 512.
        self.max_length = max_length
        self.calib_flag = calibration

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    def pad_input(self, input):
        input_id = input['input_ids'].unsqueeze(0)
        if self.max_length >= input_id.shape[1]:
            label = input_id[:, -1].to(self.device)
            pad_len = self.max_length - input_id.shape[1]
            label_index = -2 - pad_len
            input_id = pad(input_id, (0, pad_len), value=3)
        else:
            input_id = input_id[:, :self.max_length]
            label_index = -2
            label = input_id[:, -1].to(self.device)
        return (input_id, label, label_index)

    def __iter__(self):
        input_ids = None
        labels = None
        label_indices = None
        for idx, batch in enumerate(self.dataset):
            input_id, label, label_index = self.pad_input(batch)

            if input_ids is None:
                input_ids = input_id
                labels = label
                label_indices = [label_index]
            else:
                input_ids = torch.cat((input_ids, input_id), 0)
                labels = torch.cat((labels, label), 0)
                label_indices.append(label_index)

            if (idx + 1) % self.batch_size == 0:
                if not self.calib_flag:
                    yield (input_ids, labels, label_indices)
                else:
                    yield input_ids
                input_ids = None
                labels = None
                label_indices = None
        if (idx + 1) % self.batch_size != 0:
            if not self.calib_flag:
                yield (input_ids, labels, label_indices)
            else:
                yield input_ids

    def __len__(self):
        return self.length


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, OptimizationArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args, optim_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args, optim_args = parser.parse_args_into_dataclasses()

model_name = model_args.model_name_or_path

config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.config_name:
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
elif model_args.model_name_or_path:
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

if optim_args.int8:
    # Load the model obtained after Intel Neural Compressor (INC) quantization
    model = OptimizedModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

tokenizer_kwargs = {
    "cache_dir": model_args.cache_dir,
    "use_fast": model_args.use_fast_tokenizer,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
elif model_args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )

# dataset = load_dataset(data_args.tasks, split='validation').select(range(1000))
if optim_args.max_train_sample:
    dataset = load_dataset(data_args.tasks, split='validation').select(range(optim_args.max_train_sample))
else:
    dataset = load_dataset(data_args.tasks, split='validation')
dataset = dataset.shuffle(seed=42)
train_dataset = load_dataset(data_args.tasks, split='train')
train_dataset = train_dataset.shuffle(seed=42)

if optim_args.max_cali_sample:
    calib_dataset = train_dataset.select(range(optim_args.max_cali_sample))
else:
    calib_dataset = train_dataset
calib_dataloader = NLPDataloader(
    dataset,
    tokenizer,
    batch_size=data_args.batch_size,
    device='cpu',
    max_length=512,
    calibration=True
)

evaluator = Evaluator(dataset, tokenizer, 16, 'cpu')

def eval_func(model):
    acc = evaluator.evaluate(model)
    return acc

if optim_args.tune:
    tokenizer.save_pretrained(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)
    q_config = QuantizationConfig(approach=optim_args.quantization_approach, max_trials=600)
    quantizer = NoTrainerOptimizer(model, training_args.output_dir)
    model = quantizer.quantize(q_config, eval_func=eval_func, calib_dataloader=calib_dataloader)

if optim_args.benchmark or optim_args.accuracy_only:
    results = eval_func(model)
    print('Accuracy:', results)
    if optim_args.show:
        example = dataset[0]
        encoded_prompt = tokenizer.encode(example["text"], add_special_tokens=False, return_tensors="pt")
        # print("Original text: ", example["text"])
        input_ids = encoded_prompt
        input=input_ids
        input=torch.squeeze(input)[:-1].unsqueeze(dim=0)
        output_sequences = model.generate(input_ids=input,max_length=len(encoded_prompt[0]))
        text = tokenizer.decode(output_sequences[0], clean_up_tokenization_spaces=True)
        print("Generated text: ", text)