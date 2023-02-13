# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Data processing utils for run_mlm.py script
"""
import logging
import random
import os

import nltk

import datasets

from detokenizer import wikitext_detokenizer

logger = logging.getLogger(__name__)


def calc_max_seq_length(tokenizer, args):
    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    return max_seq_length


def line_by_line_process(datasets, tokenizer, args, text_column_name='text'):
    # When using line_by_line, we just tokenize each nonempty line.
    # TODO need to fix implementation to work with the format of 1 article per entry
    raise NotImplementedError("Need to fix implementation")
    padding = "max_length" if args.pad_to_max_length else False

    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [line for line in examples[text_column_name]
                                      if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=args.max_seq_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    tokenized_dataset = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not args.overwrite_cache,
    )
    return tokenized_dataset


def concatenate_process(datasets, tokenizer, args, text_column_name='text'):
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    # TODO fix implementation, need to take into account special tokens
    raise NotImplementedError("Need to fix implementation")

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    tokenized_dataset = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not args.overwrite_cache,
    )

    max_seq_length = calc_max_seq_length(tokenizer, args)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length]
                for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
    )

    return tokenized_dataset


def wikitext_preprocess(datasets, args):
    """ Return wikitext data detokenized, 1 article per entry"""
    def is_article_begin(line):
        def f(s):
            return s.strip().startswith("=") and s.strip().endswith("=")
        return f(line) and not f(line.strip().strip('='))

    def segment_articles(examples):
        articles = []
        for l in examples['text']:
            # TODO need to improve rule
            if is_article_begin(l):
                articles.append([])
            if l:
                articles[-1].append(l)
        return {'text': [' '.join(a) for a in articles if a]}

    def detokenize(example):
        article = wikitext_detokenizer(example['text']).split('\n')
        article = ' '.join(s.strip() for s in article if s.strip())
        return {'text': article}

    proc_data = datasets.map(
        segment_articles,
        batched=True,
        batch_size=0,
        remove_columns=['text'],
        load_from_cache_file=not args.overwrite_cache,
        # desc="Wikitext: segmenting articles"
    )
    proc_data = proc_data.map(
        detokenize,
        num_proc=args.preprocessing_num_workers,
        remove_columns=['text'],
        load_from_cache_file=not args.overwrite_cache,
        # desc="Wikitext: detokenizing"
    )

    return proc_data


def bookcorpusopen_preprocess(datasets, args):
    """ Return bookcorpusopen data, 1 article per entry"""
    def remove_line_delimiters(example):
        return {'text': ' '.join(s.strip() for s in example['text'].split('\n') if s.strip())}

    proc_datasets = datasets.map(
        remove_line_delimiters,
        num_proc=args.preprocessing_num_workers,
        remove_columns=['title'],
        load_from_cache_file=not args.overwrite_cache,
        desc="BookCorpusOpen: segmenting articles"
    )
    return proc_datasets


def wikipedia_preprocess(datasets, args):
    """ Return wikipedia data, 1 article per entry"""
    def remove_line_delimiters(example):
        return {'text': ' '.join(s.strip() for s in example['text'].split('\n') if s.strip())}

    proc_datasets = datasets.map(
        remove_line_delimiters,
        num_proc=args.preprocessing_num_workers,
        remove_columns=['title'],
        load_from_cache_file=not args.overwrite_cache,
        desc="Wikipedia: segmenting articles"
    )
    return proc_datasets


def segment_pair_nsp_process(datasets, tokenizer, args, text_column_name='text'):
    def sent_segment(example):
        return {'sentences': nltk.tokenize.sent_tokenize(example[text_column_name])}

    def tokenize(example):
        sequences = tokenizer(example['sentences'], add_special_tokens=False,
                              return_token_type_ids=False, return_attention_mask=False)
        sequences = [seq for seq in sequences['input_ids'] if len(seq) > 0]
        return {'input_ids': sequences}

    max_seq_length = calc_max_seq_length(tokenizer, args)
    max_seq_length_wo_special = max_seq_length - \
        tokenizer.num_special_tokens_to_add(pair=True)
    # rng = random.Random(args.dataset_seed)
    # Datasets hashing goes wrong when args object is passed into the processing
    # function. Therefore we inference the arguments we need in the function here
    # so that hashing will work properly. Wrong hashing will cause recompute of the
    # function everytime we run it instead of using cache
    short_seq_probability = args.short_seq_probability
    nsp_probability = args.nsp_probability
    dataset_seed = args.dataset_seed

    def create_sentence_pairs_for_nsp(examples, indices):
        docs = examples['input_ids']
        if len(docs) < 10:
            raise RuntimeError("Small number of documents per batch")
        instances = []
        rng = random.Random(dataset_seed + indices[0])
        next_sentence_label = []
        for doc_index, document in enumerate(docs):
            # We *usually* want to fill up the entire sequence since we are padding
            # to `max_seq_length_wo_special` anyways, so short sequences are generally wasted
            # computation. However, we *sometimes*
            # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
            # sequences to minimize the mismatch between pre-training and fine-tuning.
            # The `target_seq_length` is just a rough target however, whereas
            # `max_seq_length_wo_special` is a hard limit.
            target_seq_length = max_seq_length_wo_special
            if rng.random() < short_seq_probability:
                target_seq_length = rng.randint(2, max_seq_length_wo_special)

            # We DON'T just concatenate all of the tokens from a document into a long
            # sequence and choose an arbitrary split point because this would make the
            # next sentence prediction task too easy. Instead, we split the input into
            # segments "A" and "B" based on the actual "sentences" provided by the user
            # input.
            current_chunk = []
            current_length = 0
            i = 0

            while i < len(document):
                segment = document[i]
                current_chunk.append(segment)
                current_length += len(segment)
                if i == len(document) - 1 or current_length >= target_seq_length:
                    if current_chunk:
                        # `a_end` is how many segments from `current_chunk` go into the `A`
                        # (first) sentence.
                        a_end = 1
                        if len(current_chunk) >= 2:
                            a_end = rng.randint(1, len(current_chunk) - 1)

                        tokens_a = []
                        for j in range(a_end):
                            tokens_a.extend(current_chunk[j])

                        tokens_b = []

                        # Getting a random sentence to follow sentence A
                        if len(current_chunk) == 1 or rng.random() < nsp_probability:
                            is_random_next = True
                            target_b_length = target_seq_length - len(tokens_a)
                            for _ in range(10):
                                random_document_index = rng.randint(
                                    0, len(docs) - 1)
                                if random_document_index != doc_index:
                                    break
                            else:
                                raise RuntimeError(
                                    "Couldn't find a document different than current document")

                            random_document = docs[random_document_index]
                            random_start = rng.randint(
                                0, len(random_document) - 1)
                            for j in range(random_start, len(random_document)):
                                tokens_b.extend(random_document[j])
                                if len(tokens_b) >= target_b_length:
                                    break
                            # We didn't actually use these segments so we "put them back" so
                            # they don't go to waste.
                            num_unused_segments = len(current_chunk) - a_end
                            i -= num_unused_segments
                        # Actual next
                        else:
                            is_random_next = False
                            for j in range(a_end, len(current_chunk)):
                                tokens_b.extend(current_chunk[j])

                        assert len(tokens_a) >= 1
                        assert len(tokens_b) >= 1

                        instances.append((tokens_a, tokens_b))
                        next_sentence_label.append(int(is_random_next))
                    current_chunk = []
                    current_length = 0
                i += 1
        return {'instances': instances, 'next_sentence_label': next_sentence_label}

    padding = "max_length" if args.pad_to_max_length else False

    def prepare_for_model(example):
        example.update(tokenizer.prepare_for_model(
            *example['instances'],
            truncation='longest_first',
            max_length=max_seq_length,
            padding=padding,
        ))
        example['special_tokens_mask'] = tokenizer.get_special_tokens_mask(
            example['input_ids'], already_has_special_tokens=True)
        return example

    proc_dataset = datasets.shuffle(
        seed=args.dataset_seed,
        load_from_cache_file=not args.overwrite_cache
    )
    proc_dataset = proc_dataset.map(
        sent_segment,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not args.overwrite_cache,
        desc="Segmenting sentences"
    )
    proc_dataset = proc_dataset.map(
        tokenize,
        num_proc=args.preprocessing_num_workers,
        remove_columns=['sentences'],
        load_from_cache_file=not args.overwrite_cache,
        desc="Tokenizing"
    )
    proc_dataset = proc_dataset.map(
        create_sentence_pairs_for_nsp,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        with_indices=True,
        remove_columns=['input_ids'],
        load_from_cache_file=not args.overwrite_cache,
        desc="Creating sentences pairs"
    )
    proc_dataset = proc_dataset.map(
        prepare_for_model,
        remove_columns=['instances'],
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Preparing for model"
    )
    return proc_dataset


def doc_sentences_process(datasets, tokenizer, args, text_column_name='text'):
    def sent_segment(example):
        return {'sentences': nltk.tokenize.sent_tokenize(example[text_column_name])}

    def tokenize(example):
        sequences = tokenizer(example['sentences'], add_special_tokens=False,
                              return_token_type_ids=False, return_attention_mask=False)
        sequences = [seq for seq in sequences['input_ids'] if len(seq) > 0]
        return {'input_ids': sequences}

    max_seq_length = calc_max_seq_length(tokenizer, args)
    max_seq_length_wo_special = max_seq_length - \
        tokenizer.num_special_tokens_to_add(pair=False)

    def concatenate_sentences(example):
        document = example['input_ids']
        current_chunk = []
        current_length = 0
        docs = []
        i = 0
        while i < len(document):
            segment = document[i]
            # Add seperator token if this is not the first sentence
            if current_length > 0:
                segment = [tokenizer.sep_token_id] + segment
            current_chunk.extend(segment)
            current_length += len(segment)
            if current_chunk and i == len(document) - 1 or current_length + len(document[i + 1]) >= max_seq_length_wo_special:
                assert len(current_chunk) >= 1
                docs.append(current_chunk)
                current_chunk = []
                current_length = 0
            i += 1
        return {'documents': docs}

    def break_documents_to_sentences(examples):
        sentences = []
        for document in examples['documents']:
            sentences.extend(document)
        return {'sentences': sentences}

    padding = "max_length" if args.pad_to_max_length else False

    def prepare_for_model(example):
        example.update(tokenizer.prepare_for_model(
            example['sentences'],
            truncation=True,
            max_length=max_seq_length,
            padding=padding,
        ))
        example['special_tokens_mask'] = tokenizer.get_special_tokens_mask(
            example['input_ids'], already_has_special_tokens=True)
        return example

    proc_dataset = datasets.map(
        sent_segment,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not args.overwrite_cache,
        desc="Segmenting sentences"
    )
    proc_dataset = proc_dataset.map(
        tokenize,
        num_proc=args.preprocessing_num_workers,
        remove_columns=['sentences'],
        load_from_cache_file=not args.overwrite_cache,
        desc="Tokenizing"
    )
    proc_dataset = proc_dataset.shuffle(
        seed=args.dataset_seed,
        load_from_cache_file=not args.overwrite_cache
    )
    proc_dataset = proc_dataset.map(
        concatenate_sentences,
        num_proc=args.preprocessing_num_workers,
        remove_columns=['input_ids'],
        load_from_cache_file=not args.overwrite_cache,
        desc="Concatenating documents' sentences"
    )
    proc_dataset = proc_dataset.map(
        break_documents_to_sentences,
        num_proc=args.preprocessing_num_workers,
        remove_columns=['documents'],
        batched=True,
        load_from_cache_file=not args.overwrite_cache,
        desc="Breaking documents to sentences"
    )
    proc_dataset = proc_dataset.map(
        prepare_for_model,
        remove_columns=['sentences'],
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Preparing for model"
    )
    return proc_dataset


PROCESS_TYPE = {
    "concatenate": concatenate_process,
    "line_by_line": line_by_line_process,
    "doc_sentences": doc_sentences_process,
    "segment_pair_nsp": segment_pair_nsp_process,
}


DATASET_SPECIFIC_PREPROCESS = {
    "wikitext": wikitext_preprocess,
    "wikipedia": wikipedia_preprocess,
    "bookcorpusopen": bookcorpusopen_preprocess,
}


def load_dataset(name, config, args):
    dataset = datasets.load_dataset(
        name,
        config,
        keep_in_memory=args.keep_in_memory,
    )
    if "validation" not in dataset.keys():
        dataset["validation"] = datasets.load_dataset(
            name,
            config,
            split=f"train[:{args.validation_split_percentage}%]",
            keep_in_memory=args.keep_in_memory,
        )
        dataset["train"] = datasets.load_dataset(
            name,
            config,
            split=f"train[{args.validation_split_percentage}%:]",
            keep_in_memory=args.keep_in_memory,
        )
    return dataset


def _data_process_inner(tokenizer, args, text_column_name='text'):
    logger.info("Data preprocessing")
    # Load and prepare datasets for processing
    data_name_config_list = []
    for v in args.datasets_name_config:
        name = v.split(':')
        if len(name) == 2:
            name, config = name
        elif len(name) == 1:
            name, config = name[0], None
        else:
            raise RuntimeError(
                f"An invalide value was provided in dataset_name_config argument: {v}")
        data_name_config_list.append((name, config))
    data_list = []
    for name, config in data_name_config_list:
        data = load_dataset(name, config, args)
        try:
            data = DATASET_SPECIFIC_PREPROCESS[name](data, args)
            logger.info("Executing specific processing for {}".format(name))
        except KeyError:
            logger.info("No specific preprocessing for {}".format(name))
        data_list.append(data)
    # Concatenate datasets together
    merged_data = datasets.DatasetDict()
    merged_data.update(
        {split: datasets.concatenate_datasets(
            [data[split] for data in data_list]) for split in data_list[0]}
    )
    # Process dataset
    if args.data_process_type in ['segment_pair_nsp', 'doc_sentences']:
        nltk.download('punkt')
    proc_datasets = PROCESS_TYPE[args.data_process_type](
        merged_data, tokenizer, args, text_column_name)
    if args.dataset_cache_directory is not None:
        if os.path.exists(args.dataset_cache_directory) and not args.overwrite_cache:
            raise FileExistsError(
                f"Path: {args.dataset_cache_directory} already exists. Provide a different path or set --overwrite_cache")
        logger.info("Saving processed data in {}".format(
            args.dataset_cache_directory))
        proc_datasets.save_to_disk(args.dataset_cache_directory)
    return proc_datasets


def data_process(tokenizer, args, text_column_name='text'):
    if args.dataset_cache_directory is not None and not args.overwrite_cache:
        try:
            logger.info("Loading processed data from {}".format(
                args.dataset_cache_directory))
            proc_datasets = datasets.load_from_disk(
                args.dataset_cache_directory)
            logger.info("Loaded processed data from disk successfully.")
        except FileNotFoundError:
            if os.path.exists(args.dataset_cache_directory):
                raise FileNotFoundError(
                    f"Path {args.dataset_cache_directory} exists, but no dataset found.")
            proc_datasets = _data_process_inner(
                tokenizer, args, text_column_name)
    else:
        proc_datasets = _data_process_inner(tokenizer, args, text_column_name)
    return proc_datasets
