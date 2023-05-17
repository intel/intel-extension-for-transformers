from transformers import  AutoTokenizer, ElectraForMaskedLM, ElectraModel, ElectraForPreTraining
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
from executor_utils import Neural_Engine
import torch
import os
import copy
import argparse

# args
parser = argparse.ArgumentParser('Electra-Base Inference', add_help=False)
parser.add_argument("--generator_ir_path",
        type=str,
        help="path to generator fp32 or bf16 IR files",
        default="./ir",
    )
parser.add_argument("--discriminator_ir_path",
        type=str,
        help="path to gdiscriminator fp32 or bf16 IR files",
        default="./ir",
    )
parser.add_argument("--generator_model_name",
        type=str,
        help="path to generator pytorch model file path or name",
        default="hfl/chinese-legal-electra-base-generator",
    )
parser.add_argument("--discriminator_model_name",
        type=str,
        help="path to discriminator pytorch model file path or name",
        default="hfl/chinese-electra-base-discriminator",
    )
parser.add_argument('--generator_or_discriminator', default='both', type=str,
                    choices=['generator', 'discriminator', 'both'])
parser.add_argument('--text',
                    default="其实了解一个人并不代[MASK]什么，人是会变的，今天他喜欢凤梨，明天他可以喜欢别的",
                    type=str, help="Chinese text with tag '[MASK]'")
parser.add_argument('--mode', default='accuracy', type=str, choices=['accuracy', 'performance'])
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--seq_len', default=128, type=int)
parser.add_argument("--warm_up",
                        default=5,
                        type=int,
                        help="Warm up iteration in performance mode.")
parser.add_argument("--iterations", default=10, type=int, help="Iteration in performance mode.")
parser.add_argument("--log_file",
                        default="executor.log",
                        type=str,
                        help="File path to log information.")
args = parser.parse_args()
print(args)

def get_fill_mask_text(text, tokenizer, inputs, token_logits):
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[:, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    for i in range(args.batch_size):
        top_3_tokens = torch.topk(mask_token_logits, 3, dim=-1).indices[i][0].tolist()
        generator_texts = []
        for token in top_3_tokens:
            t = text.replace(tokenizer.mask_token, tokenizer.decode([token]))
            generator_texts.append(t)
            print(f"'>>> {t}'")
        print("==========================================================================")
        return generator_texts[0]

def get_discrimination_labels(predictions):
    results = torch.round((torch.sign(predictions) + 1) / 2)
    for i in range(results.size()[0]):
        print(f"'>>> {results[i].tolist()}'")
    print("==========================================================================")

def generator():
    tokenizer = AutoTokenizer.from_pretrained(args.generator_model_name)
    pt_model = ElectraForMaskedLM.from_pretrained(args.generator_model_name, torchscript=True)
    pt_model.eval()
    text = args.text
    inputs = tokenizer(text, return_tensors="pt")
    bs = args.batch_size
    seq_len = inputs.input_ids.size()[1]
    input_ids = inputs.input_ids.repeat(bs, 1)
    attention_mask = inputs.attention_mask.repeat(bs, 1)
    token_type_ids = inputs.attention_mask.repeat(bs, 1)
    print("Masked text: {}".format(text))
    print("Complete text given by PyTorch model (Top-3): ")
    pt_logits = pt_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
    ret = get_fill_mask_text(text, tokenizer, inputs, pt_logits)
    print("Complete text given by Neural Engine model (Top-3): ")
    engine_model = Graph()
    engine_model.graph_init(os.path.join(args.generator_ir_path, 'conf.yaml'),
                            os.path.join(args.generator_ir_path, 'model.bin'))
    engine_out = copy.deepcopy(engine_model.inference([input_ids.detach().numpy(),
                                                       attention_mask.detach().numpy(),
                                                       token_type_ids.detach().numpy()]))
    engine_logits = torch.Tensor(list(engine_out.values())[0].reshape(bs, seq_len, -1))
    ret = get_fill_mask_text(text, tokenizer, inputs, engine_logits)
    return ret

def discriminator(text):
    tokenizer = AutoTokenizer.from_pretrained(args.discriminator_model_name)
    pt_model = ElectraForPreTraining.from_pretrained(args.discriminator_model_name, torchscript=True)
    pt_model.eval()
    tokens = tokenizer.tokenize(text, add_special_tokens=True)
    inputs = tokenizer(text, return_tensors="pt")
    bs = args.batch_size
    seq_len = inputs.input_ids.size()[1]
    input_ids = inputs.input_ids.repeat(bs, 1)
    attention_mask = inputs.attention_mask.repeat(bs, 1)
    token_type_ids = inputs.attention_mask.repeat(bs, 1)
    print("Discriminator input text token: {}".format(tokens))
    print("Discrimination labels given by PyTorch model: ")
    pt_logits = pt_model(input_ids=input_ids)[0]
    get_discrimination_labels(pt_logits)
    print("Discrimination labels given by Neural Engine model: ")
    engine_model = Graph()
    engine_model.graph_init(os.path.join(args.discriminator_ir_path, 'conf.yaml'),
                            os.path.join(args.discriminator_ir_path, 'model.bin'))
    engine_out = copy.deepcopy(engine_model.inference([input_ids.detach().numpy(),
                                                       attention_mask.detach().numpy(),
                                                       token_type_ids.detach().numpy()]))
    engine_logits = torch.Tensor(list(engine_out.values())[0]).reshape(bs, seq_len)
    get_discrimination_labels((engine_logits))

test_perf = True if args.mode == 'performance' else False
only_generator = True if args.generator_or_discriminator == "generator" else False
only_discriminator = True if args.generator_or_discriminator == "discriminator" else False

if not test_perf:
    ret_text = args.text
    if not only_discriminator:
        ret_text = generator()
    if not only_generator:
        discriminator(ret_text)
else:
    if not only_discriminator:
        print("Testing generator performance...")
        executor = Neural_Engine(args.generator_ir_path, args.log_file, "native")
        executor.performance(args.batch_size, args.seq_len, args.iterations, args.warm_up)
    if not only_generator:
        print("Testing discriminator performance...")
        executor = Neural_Engine(args.discriminator_ir_path, args.log_file, "native")
        executor.performance(args.batch_size, args.seq_len, args.iterations, args.warm_up)
