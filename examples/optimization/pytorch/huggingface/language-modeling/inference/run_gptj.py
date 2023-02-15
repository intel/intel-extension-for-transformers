
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import intel_extension_for_pytorch as ipex
import time
import sys
import argparse


# args
parser = argparse.ArgumentParser('GPT-J inference script', add_help=False)
parser.add_argument('--precision', default='bf16', type=str, help="fp32 or bf16")
parser.add_argument('--max-new-tokens', default=32, type=int, help="output max new tokens")
parser.add_argument('--greedy', action='store_true')
parser.add_argument("--block-size", type=int, default=None, help=(
        "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
        " this size for training. Default to the model max input length for single sentence inputs (take into"
        " account special tokens)."
    ))
parser.add_argument('--generation', action='store_true', help="inference for text-generation task.")
parser.add_argument('--clm', action='store_true', help="inference for causal language modeling task.")
parser.add_argument("--num_iter", type=int, default=0, help="iterations for performance benchmark.")
parser.add_argument("--num_warmup", type=int, default=0, help="warmup numbers.")
parser.add_argument("--batch-size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")

args = parser.parse_args()
print(args)

amp_enabled = True if args.precision != "fp32" else False
amp_dtype = torch.bfloat16 if args.precision != "fp32" else torch.float32

if args.greedy:
    generate_kwargs = dict(do_sample=False, temperature=0.9)
else:
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)

# load model
model_id = "EleutherAI/gpt-j-6B"
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = model.eval()

# prepare wikitext dataset for causal language modeling.
if args.clm:
    raw_datasets = load_dataset("wikitext","wikitext-2-raw-v1")
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    eval_dataset = lm_datasets["validation"]

    # DataLoaders creation
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size
    )

# to channels last
model = model.to(memory_format=torch.channels_last)
# to ipex
model = ipex.optimize(model, dtype=amp_dtype, inplace=True)

# for generation.
if args.generation:
    # 32 tokens input  prompt
    prompt = "Once upon a time, there existed a little girl, who liked to have adventures." + \
            " She wanted to go to places and meet new people, and have fun."

    # start
    total_time = 0.0
    with torch.cpu.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
        for i in range(args.num_iter):
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            gen_tokens = model.generate(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)
            gen_text = tokenizer.batch_decode(gen_tokens)[0]
            toc = time.time()
            print(gen_text, flush=True)
            if i >= args.num_warmup:
                total_time += (toc - tic)

    print("Inference latency: %.3f ms." % (total_time / (args.num_iter - args.num_warmup) * 1000))

# for casual language modeling.
if args.clm:
    total_time = 0.0
    losses = []
    for step, batch in enumerate(eval_dataloader):
        if args.num_iter > 0 and step > args.num_iter:
            break
        with torch.no_grad():
            with torch.cpu.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                tic = time.time()
                outputs = model(**batch)
                toc = time.time()
                if args.num_warmup > 0 and step >= args.num_warmup:
                    total_time += (toc - tic)
        loss = outputs.loss
        losses.append(loss.repeat(args.batch_size))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    print("eval_loss: %.3f ms." % eval_loss)
    print("perplexity: %.3f ms." % perplexity)
    print("Inference latency: %.3f ms." % (total_time / (args.num_iter * args.batch_size - args.num_warmup) * 1000))