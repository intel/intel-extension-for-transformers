import collections
import inspect
import math
import os
import copy
import sys
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

import torch
from packaging import version
import torch.distributed as dist

# from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import Trainer, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_torch_tpu_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
)

# Integrations must be imported before ML frameworks:
from transformers.integrations import hp_params
from transformers.modeling_utils import unwrap_model
from transformers.trainer import TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import (
    HPSearchBackend,
    ShardedDDPOption,
    TrainOutput,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from nlp_toolkit import OptimizeConfig
from nlp_toolkit import CONFIG_NAME as config_name, Provider
from neural_compressor.experimental import Component
from neural_compressor.utils import logger

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    from .trainer_pt_utils import smp_forward_backward

if TYPE_CHECKING:
    import optuna

__version__ = "4.9.2"


class NLPTrainer(OptimizeConfig, Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_training = False
        self._eval_func = None
        self._train_func = None
        self.teacher_model = None
        self._calib_dataloader = None
        self._resuming_checkpoint = None
        self.compression_ctrl = None
        self.inc_int8_flag = False

    @property
    def resuming_checkpoint(self):
        return self._resuming_checkpoint

    @resuming_checkpoint.setter
    def resuming_checkpoint(self, path: str):
        self._resuming_checkpoint = path

    @property
    def eval_func(self):
        return self._eval_func

    @property
    def train_func(self):
        return self._train_func

    @property
    def calib_dataloader(self):
        return self._calib_dataloader

    @eval_func.setter
    def eval_func(self, func: Callable):
        self._eval_func = func

    @train_func.setter
    def train_func(self, func: Callable):
        self._train_func = func

    @calib_dataloader.setter
    def calib_dataloader(self, dataloader):
        self._calib_dataloader = dataloader

    def builtin_eval_func(self, model):
        assert self.metrics is not None, "Please set metrics to NLPtrainer"
        self.model = model
        results = self.evaluate()
        if isinstance(self.metrics["metrics"], list):
            nums = len(self.metrics["metrics"])
            for i in range(nums):
                assert self.metrics["metrics"][i] in results.keys(), \
                    "Please set metric from {}".format(results.keys())
            if nums == 1:
                result = results.get(self.metrics["metrics"][0])
            else:
                assert "weights" in self.metrics.keys(), \
                    "Please set weights for metrics if you want to use more than one metric"
                assert len(self.metrics["metrics"]) == len(self.metrics["weights"]), \
                    "Please set the same length to metrics and weights"
                result = 0
                for i in range(nums):
                    result += results[self.metrics["metrics"][i]] * self.metrics["weights"][i]
            logger.info("metrics: {}".format(result))
        elif isinstance(self.metrics["metrics"], str):
            assert self.metrics["metrics"] in results.keys(), \
                    "Please set metric from {}".format(results.keys())
            result = results.get(self.metrics["metrics"])
            logger.info("{}: {}".format(self.metrics["metrics"], result))
        else:
            assert False, "Please set the correct metrics format from the README"
        logger.info("Throughput: {} samples/sec".format(results.get("eval_samples_per_second")))
        return result

    def builtin_train_func(self, model):
        self.model_wrapped = model
        self.model = model
        train_result = self.train(resume_from_checkpoint=self._resuming_checkpoint)
        metrics = train_result.metrics
        self.save_model()  # Saves the tokenizer too for easy upload
        self.log_metrics("train", metrics)
        self.save_metrics("train", metrics)
        self.save_state()
        return self.model

    def _init_quantizer(self):
        from .quantization import QuantizationMode
        from neural_compressor.experimental import Quantization, common
        if self.quantization.quant_config.usr_cfg.quantization.approach == \
          QuantizationMode.POSTTRAININGDYNAMIC.value:
            self.quantization.quant_config.usr_cfg.model.framework = "pytorch"
        else:
            self.quantization.quant_config.usr_cfg.model.framework = "pytorch_fx"

        quantizer = Quantization(self.quantization.quant_config)
        quantizer.model = common.Model(self.model)

        if self._eval_func is not None:
            quantizer.eval_func = self._eval_func
        else:
            assert self.metrics is not None, "Please pass metrics to trainer.quantization.metrics!"
            quantizer.eval_func = self.builtin_eval_func

        if self.quantization.quant_config.usr_cfg.quantization.approach == \
          QuantizationMode.POSTTRAININGSTATIC.value:
            quantizer.calib_dataloader = self.get_train_dataloader() \
                if self._calib_dataloader is None else self._calib_dataloader
        elif self.quantization.quant_config.usr_cfg.quantization.approach == \
          QuantizationMode.QUANTIZATIONAWARETRAINING.value:
            quantizer.q_func = \
                self.builtin_train_func if self._train_func is None else self._train_func

        self.quantizer = quantizer
        return quantizer

    def _nncf_quantize(self):
        from nncf import create_compressed_model
        self.parse_nncf_arguments()
        compression_state = None
        nncf_compression_state_file = self._provider_arguments.get("compression_state", None)

        if os.path.isfile(nncf_compression_state_file):
            compression_state = torch.load(nncf_compression_state_file)
        else:
            compression_state = None

        compression_algo_controller, model = create_compressed_model(
            self.model, self._provider_arguments.get("nncf_config"), compression_state=compression_state
        )

        self.compression_ctrl = \
            compression_algo_controller.distributed() if self._provider_arguments.get("distributed", None) else compression_algo_controller

        self.model = self._train_func(model)

    def _inc_quantize(self):
        self.parse_inc_arguments()
        quantizer = self._init_quantizer()
        self.opt_model = quantizer.fit()
        self.inc_int8_flag = True
        self._save_inc_int8(self.opt_model, self.args.output_dir)
        logger.info(
            "quantized model and configure file have saved to {}".format(self.args.output_dir)
        )
        return self.opt_model

    def quantize(
        self,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
        calib_dataloader=None,
    ):
        self._eval_func = self.builtin_eval_func if eval_func is None else eval_func
        self._train_func = self.builtin_train_func if train_func is None else train_func
        if calib_dataloader is not None:
            self._calib_dataloader = calib_dataloader

        if self._provider == Provider.NNCF.value:
            return self._nncf_quantize()
        elif self._provider == Provider.INC.value:
            return self._inc_quantize()
        else:
            assert False, "Unsupport provider:{}".format(self._provider)

    def _save_inc_int8(self, opt_model, output_dir):
        weights_file = os.path.join(os.path.abspath(
          os.path.expanduser(output_dir)), WEIGHTS_NAME)
        torch.save(opt_model.quantized_state_dict(), weights_file)
        logger.info(
            "quantized model and configure file have saved to {}".format(weights_file)
        )
        

    def _init_pruner(self):
        from neural_compressor.experimental import Pruning, common

        self.pruning.framework = "pytorch"
        pruning_start_epoch, pruning_end_epoch = self.pruning.epoch_range

        if pruning_start_epoch > self.args.num_train_epochs - 1:
            logger.warning(
                f"Pruning end epoch {pruning_start_epoch} is higher than the total number of training epoch "
                f"{self.args.num_train_epochs}. No pruning will be applied."
            )

        if pruning_end_epoch > self.args.num_train_epochs - 1:
            logger.warning(
                f"Pruning end epoch {pruning_end_epoch} is higher than the total number of training epoch "
                f"{self.args.num_train_epochs}. The target sparsity will not be reached."
            )

        pruner = Pruning(self.pruning.prune_config)
        pruner.model = common.Model(self.model)

        if self._eval_func is not None:
            pruner.eval_func = self._eval_func
        else:
            assert self.metrics is not None, "Please pass metrics to trainer.pruning.metrics!"
            pruner.eval_func = self.builtin_eval_func

        pruner.pruning_func = \
            self.builtin_train_func if self._train_func is None else self._train_func
        self.pruner = pruner
        return pruner

    def prune(
        self,
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
        calib_dataloader=None,
    ):
        self.parse_inc_arguments()
        if self.pruning.metrics is not None:
            self.metrics = self.pruning.metrics
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func

        pruner = self._init_pruner()
        self.opt_model = pruner.fit()

        return self.opt_model

    def _init_distiller(self):
        from neural_compressor.experimental import Distillation, common

        self.distillation.framework = "pytorch"
        distiller = Distillation(self.distillation.distill_config)
        distiller.model = common.Model(self.model)
        distiller.teacher_model = common.Model(self.teacher_model)


        if self._eval_func is not None:
            distiller.eval_func = self._eval_func
        else:
            assert self.metrics is not None, "Please pass metrics to trainer.distillation.metrics!"
            distiller.eval_func = self.builtin_eval_func

        distiller.train_func = \
            self.builtin_train_func if self._train_func is None else self._train_func
        distiller.create_criterion()
        self.distiller = distiller
        return distiller

    def distill(
        self,
        teacher_model: Union[PreTrainedModel, torch.nn.Module],
        eval_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
    ):
        self.parse_inc_arguments()
        if self.distillation.metrics is not None:
            self.metrics = self.distillation.metrics
        if eval_func is not None:
            self._eval_func = eval_func
        if train_func is not None:
            self._train_func = train_func

        self.teacher_model = teacher_model
        distiller = self._init_distiller()
        self.opt_model = distiller.fit()

        return self.opt_model

    def train(
        self,
        component: Optional[Component] = None,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            component (:obj:`Component`, `optional`):
                Component object handling the training process.
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (:obj:`List[str]`, `optional`)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        self.component = component

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
            # If the model is on the GPU, it still works!
            self._load_state_dict_in_model(state_dict)

            # release memory
            del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
        if isinstance(component, Component):
            component.pre_epoch_begin()
            if component.combination is not None and "Quantization" in component.combination:
                model = component.model.model
        for epoch in range(epochs_trained, num_train_epochs):
            if self.compression_ctrl is not None:
                self.compression_ctrl.scheduler.epoch_step()
                print(self.compression_ctrl.statistics().to_str())
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            if isinstance(component, Component):
                component.on_epoch_begin(epoch)

            self.in_training = True
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                    if isinstance(component, Component):
                        component.on_batch_begin(step)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    if isinstance(component, Component):
                        component.on_post_grad()

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    if self.compression_ctrl is not None:
                        self.compression_ctrl.scheduler.step()
                    optimizer_was_run = True
                    self.optimizer.step()

                    if optimizer_was_run:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.state.curr_loss = tr_loss_step.cpu().detach().item()
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    if isinstance(component, Component):
                        component.on_batch_end()
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.in_training = False
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            if isinstance(component, Component):
                # When Distillation is involved, model will be evaluated in "on_epoch_end" hook, while in SQuAD 
                # evaluation, "start_positions" and "end_positions" will be removed from inputs of the fx model,
                # this will damage the training afterward, so use the copied model for evaluation, 
                # and then restore the model.
                component.model.model = copy.deepcopy(model)
                component.on_epoch_end()
                component.model.model = model
                if 'Distillation' in component.__repr__():
                    model.train()
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )

            if self.control.should_training_stop:
                break

        if isinstance(component, Component):
            component.post_epoch_end()
            if component.combination is not None and "Quantization" in component.combination:
                self.model = component.model.model

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            if self.compression_ctrl is not None:
                from nncf.common.utils.tensorboard import prepare_for_tensorboard
                logs["compression_loss"] = self.compression_ctrl.loss().item()
                compression_stats = self.compression_ctrl.statistics()
                for key, value in prepare_for_tensorboard(compression_stats).items():
                    logs["compression/statistics/{0}".format(key)] = value
                print(compression_stats.to_str())

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.compression_ctrl is not None:
            compression_loss = self.compression_ctrl.loss()
            loss += compression_loss

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if "teacher_logits" in inputs:
            teacher_logits = inputs.pop("teacher_logits")
            if "start_positions" in inputs and "end_positions" in inputs: # for SQuAD
                teacher_logits = torch.vstack(list(teacher_logits))
        else:
            teacher_logits = None

        outputs = model(**inputs)
        if self.in_training and hasattr(self, "component") and \
           hasattr(self.component, "criterion"):
            qa_output_merger = lambda outputs : torch.vstack([torch.vstack([sl, el]) for sl, el in \
                                                zip(outputs["start_logits"], outputs["end_logits"])])
            qa_output_spliter = lambda outputs : (outputs[0::2], outputs[1::2])
            def get_logits(outputs):
                if isinstance(outputs, dict):
                    if "logits" in outputs:
                        logits = outputs["logits"]
                    elif "start_logits" in outputs and "end_logits" in outputs:
                        logits = qa_output_merger(outputs)
                    else:
                        raise AssertionError("Logits of outputs not included, can't compute loss")
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs[1]
                return logits

            if labels is None:
                if "labels" in inputs: # for GLUE
                    labels = inputs["labels"]
                elif "start_positions" in inputs and "end_positions" in inputs: # for SQuAD
                    labels = torch.hstack([torch.tensor([sp, ep]) for sp, ep in \
                            zip(inputs["start_positions"], inputs["end_positions"])])
                else:
                    raise AssertionError("Labels of input data not provided, can't compute loss")
            logits = get_logits(outputs)
            if hasattr(self.component, "on_post_forward"):
                self.component.on_post_forward(inputs, teacher_output=teacher_logits)
                self.component.criterion.teacher_outputs = get_logits(self.component.criterion.teacher_outputs)
            loss = self.component.criterion(logits, labels)
            if "start_positions" in inputs and "end_positions" in inputs:
                start_logits, end_logits = qa_output_spliter(logits)
                outputs = {"start_logits":start_logits, "end_logits":end_logits, "loss":loss}
            else:
                outputs = {"logits":logits, "loss":loss}
        else:
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            if self._provider == "nncf":
                signature = inspect.signature(self.model.get_nncf_wrapped_model().forward)
            else:
                signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids", "teacher_logits"]
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            unwrapped_model = unwrap_model(self.model)
            if self._provider == "nncf":
                is_pretrained = isinstance(unwrapped_model.get_nncf_wrapped_model(), PreTrainedModel)
            else:
                is_pretrained = isinstance(unwrapped_model, PreTrainedModel)

            if is_pretrained:
                if state_dict is None:
                    state_dict = unwrapped_model.state_dict()
                unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
            #overwrite `pytorch_model.bin` with inc int8 format.
            if self.inc_int8_flag:
                self._save_inc_int8(self.opt_model, output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
