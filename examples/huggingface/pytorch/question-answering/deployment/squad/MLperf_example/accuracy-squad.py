# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import six
from transformers import BertConfig, BertTokenizer, AutoTokenizer
import tokenization 
from create_squad_data import read_squad_examples, convert_examples_to_features

# To support feature cache.
import pickle

max_seq_length = 384
max_query_length = 64
doc_stride = 128

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #     pred_text = steve smith
    #     orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file):
    """Write final predictions to the json file and log-odds of null if needed."""
    print("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")


def load_loadgen_log(log_path, eval_features, model_name, output_transposed=False):
    with open(log_path,'r+') as f:
        predictions = json.load(f)

    results = []
    for prediction in predictions:
        qsl_idx = prediction["qsl_idx"]
        if output_transposed:
            logits = np.frombuffer(bytes.fromhex(prediction["data"]), np.float32).reshape(2, -1)
            logits = np.transpose(logits)
        else:
            logits = np.frombuffer(bytes.fromhex(prediction["data"]), np.float32).reshape(-1, 2)
        # Pad logits to max_seq_length
        seq_length = logits.shape[0]
        start_logits = np.ones(max_seq_length) * -10000.0
        end_logits = np.ones(max_seq_length) * -10000.0
        start_logits[:seq_length] = logits[:, 0]
        end_logits[:seq_length] = logits[:, 1]
        if model_name == "Minilm_l12":
            results.append(RawResult(
                unique_id=eval_features[qsl_idx].unique_id,
                start_logits=start_logits.tolist(),
                end_logits=end_logits.tolist()
            ))
        elif model_name == "Minilm_l8":
            results.append(RawResult(
                unique_id=qsl_idx,
                start_logits=start_logits.tolist(),
                end_logits=end_logits.tolist()
                ))

    return results
def evaluate_minilm_l12():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Minilm_l12", type=str, help="Model name for data")
    parser.add_argument("--vocab_file", default="datasets/vocab.txt", help="Path to vocab.txt")
    parser.add_argument("--val_data", default="datasets/dev-v1.1.json", help="Path to validation data")
    parser.add_argument("--log_file", default="./mlperf_output/mlperf_log_accuracy.json", help="Path to LoadGen accuracy log")
    parser.add_argument("--out_file", default="predictions.json", help="Path to output predictions file")
    parser.add_argument("--output_transposed", action="store_true", help="Transpose the output")
    args = parser.parse_args()
    if args.model_name != "Minilm_l12":
        return
    print("Reading examples...")
    eval_examples = read_squad_examples(input_file=args.val_data,
        is_training=False, version_2_with_negative=False) 

    eval_features = []

    print("Creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    print("Converting examples to features...")
    def append_feature(feature):
        eval_features.append(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        output_fn=append_feature,
        verbose_logging=False) 


    print("Loading LoadGen logs...")
    results = load_loadgen_log(args.log_file, eval_features,args.model_name, args.output_transposed)

    print("Post-processing predictions...")
    write_predictions(eval_examples, eval_features, results, 20, 30, True, args.out_file)

    print("Evaluating predictions...")
    cmd = "python3 evaluate-v1.1.py datasets/dev-v1.1.json predictions.json"
    subprocess.check_call(cmd, shell=True)  # nosec
def evaluate_minilm_l8():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Minilm_l8", type=str, help="Model name for data")
    parser.add_argument("--vocab_file", default="datasets/vocab.txt", help="Path to vocab.txt")
    parser.add_argument("--val_data", default="datasets/dev-v1.1.json", help="Path to validation data")
    parser.add_argument("--log_file", default="./mlperf_output/mlperf_log_accuracy.json", help="Path to LoadGen accuracy log")
    parser.add_argument("--out_file", default="predictions.json", help="Path to output predictions file")
    parser.add_argument("--output_transposed", action="store_true", help="Transpose the output")
    args = parser.parse_args()
    if args.model_name != "Minilm_l8":
        print("please supply a curruct model_name")
        return
    print("Reading examples...")
    eval_examples = read_squad_examples(input_file=args.val_data,
        is_training=False, version_2_with_negative=False)

    eval_features = []
    # Load features if cached, convert from examples otherwise.

    print("Creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    new_eval_examples = {"question": [], "context": [], "id": []}
    for eval_example in eval_examples:
        new_eval_examples["question"].append(eval_example.question_text)
        new_eval_examples["context"].append(" ".join(eval_example.doc_tokens))
        new_eval_examples["id"].append(eval_example.qas_id)

    from datasets import Dataset

    eval_examples = Dataset.from_dict(new_eval_examples)
    question_column_name  = "question"
    context_column_name = "context"
    pad_on_right = True
    pad_to_max_length = True
    column_names = ['id', 'context', 'question']


    def prepare_validation_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
        )
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])


            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=1,
            remove_columns=column_names,)

    print("Loading LoadGen logs...")
    results = load_loadgen_log(args.log_file, eval_dataset, args.model_name, args.output_transposed)
    all_results = []
    for r in results:
        all_results.append((r.unique_id, r.start_logits, r.end_logits))
    sorted_results = sorted(all_results, key=lambda x: x[0])
    start_logits_list = []
    end_logits_list = []
    for each in sorted_results:
        start_logits_list.append([each[1]])
        end_logits_list.append([each[2]])

    start_logits = np.concatenate(start_logits_list, axis=0)
    end_logits = np.concatenate(end_logits_list, axis=0)
    results = (start_logits, end_logits)
    from utils_qa import postprocess_qa_predictions
    predictions = postprocess_qa_predictions(
            examples=eval_examples,
            features=eval_dataset,
            predictions=results,
            output_dir="./")


    print("Evaluating predictions...")
    cmd = "python3 evaluate-v1.1.py datasets/dev-v1.1.json predictions.json"
    subprocess.check_call(cmd, shell=True)  # nosec

if __name__ == "__main__":
    evaluate_minilm_l12()
    evaluate_minilm_l8()
