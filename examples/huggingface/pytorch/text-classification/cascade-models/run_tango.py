import argparse
import logging
from time import perf_counter

import evaluate
import numpy as np
from argparse_range import range_action
from datasets import load_dataset
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="TangoBERT")
    parser.add_argument(
        "--task_name",
        type=str,
        help="Name of the GLUE task.",
        choices=list(task_to_keys.keys()),
        required=True
    )

    parser.add_argument(
        "--small_model_name_or_path",
        type=str,
        help="Path to the small pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--big_model_name_or_path",
        type=str,
        help="Path to the big pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--device_small",
        default=0,
        help="Defines the device (e.g., \"cpu\", \"cuda:1\", \"mps\", or a GPU ordinal rank like 1) on which  "
             "small model pipeline will be allocated. "
    )
    parser.add_argument(
        "--device_big",
        default=0,
        help="Defines the device (e.g., \"cpu\", \"cuda:1\", \"mps\", or a GPU ordinal rank like 1) on which this "
             "big model pipeline will be allocated. "
    )
    parser.add_argument(
        "--per_device_eval_batch_size-small",
        type=int,
        default=32,
        help="Batch size (per device) for small model inference.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size-big",
        type=int,
        default=32,
        help="Batch size (per device) for big model inference.",
    )
    parser.add_argument("--confidence_threshold",
                        type=float,
                        default=0.85,
                        action=range_action(0.5, 1.0),
                        help="Confidence threshold for small model prediction (must be in range 0.5..1.0)."
                        )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(args)

    logger.info("Prepare dataset")
    eval_dataset = load_dataset("glue", args.task_name)[
        "validation_matched" if args.task_name == "mnli" else "validation"]

    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    texts = (
        (eval_dataset[sentence1_key],) if sentence2_key is None else (
            eval_dataset[sentence1_key], eval_dataset[sentence2_key])
    )

    logger.info("Loading small model")
    small_model = AutoModelForSequenceClassification.from_pretrained(args.small_model_name_or_path)
    small_tokenizer = AutoTokenizer.from_pretrained(args.small_model_name_or_path)
    pipe_small = pipeline("text-classification", model=small_model, tokenizer=small_tokenizer, device=args.device_small)

    logger.info("Loading big model")
    big_model = AutoModelForSequenceClassification.from_pretrained(args.big_model_name_or_path)
    big_tokenizer = AutoTokenizer.from_pretrained(args.big_model_name_or_path)
    pipe_big = pipeline("text-classification", model=big_model, tokenizer=big_tokenizer, device=args.device_big)

    logger.info("Inference")
    start_time = perf_counter()
    output_small = pipe_small(*texts, batch_size=args.per_device_eval_batch_size_small)
    low_confidence_indices = [idx for idx, pred in enumerate(output_small) if
                              np.max(pred['score']) < args.confidence_threshold]
    if len(low_confidence_indices) > 0:
        low_confidence_subset = np.asarray(*texts)[low_confidence_indices].tolist()
        output_big = pipe_big(low_confidence_subset, batch_size=args.per_device_eval_batch_size_big)
    end_time = perf_counter()

    logger.info("Evaluation")
    high_confidence_indices = [idx for idx, pred in enumerate(output_small) if
                               np.max(pred['score']) >= args.confidence_threshold]
    predictions = np.empty([len(output_small)])
    predictions[high_confidence_indices] = [small_model.config.label2id[o['label']] for o in
                                            np.asarray(output_small)[high_confidence_indices]]
    if len(low_confidence_indices) > 0:
        predictions[low_confidence_indices] = [big_model.config.label2id[o['label']] for o in output_big]

    metric = evaluate.load("glue", args.task_name)
    metric.add_batch(predictions=predictions, references=eval_dataset['label'])
    eval_metric = metric.compute()
    logger.info(f"eval_metric: {eval_metric}")
    inference_time = round((end_time - start_time) * 1000, 0)
    logger.info(f"inference time:  {inference_time} ms")


if __name__ == "__main__":
    main()
