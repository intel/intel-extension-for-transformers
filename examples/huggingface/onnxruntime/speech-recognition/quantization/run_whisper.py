import os
import onnx
import numpy as np
import time
import argparse
from datasets import load_dataset
from transformers import WhisperProcessor, AutoConfig, LogitsProcessorList, PretrainedConfig
import torch
from evaluate import load
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

parser = argparse.ArgumentParser()
parser.add_argument('--tune', dest='tune', action='store_true', 
                    help='tune best int8 model with Neural Compressor')
parser.add_argument('--accuracy_only', dest='accuracy_only', action='store_true',
                    help='run accuracy_only')
parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                    help='run benchmark')
parser.add_argument('--batch_size', default=1, type=int,
                    help='for accuracy measurement only')
parser.add_argument('--iters', default=100, type=int,
                    help='for benchmark measurement only')
parser.add_argument('--warmup', default=5, type=int,
                    help='for benchmark measurement only')
parser.add_argument('--output_model', default="int8_models", type=str,
                    help='the folder path to save the int8 models')
parser.add_argument('--cache_dir', default=None, type=str,
                    help='the cache dir to dataset')
parser.add_argument('--input_model', default=None, type=str,
                    help='the folder path to fp32 models')
parser.add_argument('--approach', default='dynamic', type=str,
                    help='the quantization approach to use')
parser.add_argument('--model_name_or_path', default=None, type=str)
parser.add_argument('--cores_per_instance', default=4, type=int,
                    help='cores per instance during benchmark')
parser.add_argument('--max_new_tokens', default=16, type=int,
                    help='the maximum numbers of tokens to generate')

args = parser.parse_args()

def eval(model_path):
    config = PretrainedConfig.from_pretrained(args.model_name_or_path)
    predictions = []
    references = []
    sessions = ORTModelForSpeechSeq2Seq.load_model(
            os.path.join(model_path, 'encoder_model.onnx'),
            os.path.join(model_path, 'decoder_model.onnx'),
            os.path.join(model_path, 'decoder_with_past_model.onnx'))
    model = ORTModelForSpeechSeq2Seq(sessions[0], sessions[1], config, model_path, sessions[2])
    wer = load("wer") # metric
    librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test", cache_dir=args.cache_dir)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path)

    for idx, batch in enumerate(librispeech_test_clean):
        audio = batch["audio"]
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        reference = processor.tokenizer._normalize(batch['text'])
        references.append(reference)
        predicted_ids = model.generate(input_features)[0]
        transcription = processor.decode(predicted_ids)
        prediction = processor.tokenizer._normalize(transcription)
        predictions.append(prediction)
    wer_result = wer.compute(references=references, predictions=predictions)
    print(f"Result wer: {wer_result * 100}")
    accuracy = 1 - wer_result
    print("Accuracy: %.5f" % accuracy)
    return accuracy

class Dataloader:
    def __init__(self, batch_size=1, model_path=''):
        self.batch_size = batch_size
        self.logits_processor = LogitsProcessorList()
        self.encoder_sess = None
        self.decoder_sess = None
        self.decoder_kv_sess = None
        self.model_path = model_path
        self.librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test", cache_dir=args.cache_dir)
        if not model_path.endswith('encoder_model.onnx'):
            self.encoder_sess = ort.InferenceSession(os.path.join(os.path.dirname(model_path), 'encoder_model.onnx'))
            self.decoder_sess = ort.InferenceSession(os.path.join(os.path.dirname(model_path), 'decoder_model.onnx'))
            if model_path.endswith('decoder_with_past_model.onnx'):
                self.decoder_kv_sess = ort.InferenceSession(os.path.join(os.path.dirname(model_path), 'decoder_with_past_model.onnx'))

    def __iter__(self):
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        processor = WhisperProcessor.from_pretrained(args.model_name_or_path)
        for idx, batch in enumerate(self.librispeech_test_clean):
            audio = batch["audio"]
            input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features.detach().numpy()
            input_ids = torch.ones((args.batch_size, 1), dtype=torch.long, device='cpu') * config.decoder_start_token_id
            unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device='cpu')
            eos_token_id_tensor = torch.tensor([config.eos_token_id]).to('cpu')

            pad_token_id = config.pad_token_id
            if self.model_path.endswith('encoder_model.onnx'):
                yield input_features, 0
            else:
                ort_inputs = {'input_features': input_features}
                encoder_outputs = self.encoder_sess.run(None, ort_inputs)
                ort_inputs = {}
                ort_inputs['input_ids'] = input_ids.detach().numpy().astype('int64')
                ort_inputs['encoder_hidden_states'] = encoder_outputs[0]
 
                if self.model_path.endswith('decoder_model.onnx'):
                    yield ort_inputs, 0
                else:
                    kv = None
                    while True:
                        if kv is None:
                            outputs = self.decoder_sess.run(None, ort_inputs)
                            kv = outputs[1:]
                            ort_inputs.pop('encoder_hidden_states')
                            for i in range(int(len(kv) / 4)):
                                ort_inputs['past_key_values.{}.decoder.key'.format(i)] = kv[i * 4]
                                ort_inputs['past_key_values.{}.decoder.value'.format(i)] = kv[i * 4 + 1]
                                ort_inputs['past_key_values.{}.encoder.key'.format(i)] = kv[i * 4 + 2]
                                ort_inputs['past_key_values.{}.encoder.value'.format(i)] = kv[i * 4 + 3]
 
                        else:
                            outputs = self.decoder_kv_sess.run(None, ort_inputs)
                            kv = outputs[1:]
                            for i in range(int(len(kv) / 2)):
                                ort_inputs['past_key_values.{}.decoder.key'.format(i)] = kv[i * 2]
                                ort_inputs['past_key_values.{}.decoder.value'.format(i)] = kv[i * 2 + 1]
 
                        predicted_ids = torch.from_numpy(outputs[0])
                        next_token_logits = predicted_ids[:, -1, :]
                        next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
                        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
                        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                        unfinished_sequences = unfinished_sequences.mul(
                                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                            )
                        if unfinished_sequences.max() == 0:
                            break
                        ort_inputs['input_ids'] = input_ids[:, -1:].detach().numpy()
                        yield ort_inputs, 0
 
if __name__ == "__main__":
    if args.tune:
        if os.path.exists(args.output_model):
            import shutil
            shutil.rmtree(args.output_model, ignore_errors=True)
        os.makedirs(args.output_model)

        from neural_compressor import PostTrainingQuantConfig, quantization
        model_list = ['encoder_model.onnx', 'decoder_model.onnx', 'decoder_with_past_model.onnx']
        if args.approach == 'static':
            conf = PostTrainingQuantConfig(approach="static",
                    op_type_dict={'^((?!(MatMul|Gather|Conv)).)*$': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}}},
                    recipes={"optypes_to_exclude_output_quant": ["MatMul", "Gemm"]})
            for model in model_list:
                dataloader = Dataloader(args.batch_size, os.path.join(args.input_model, model))
                q_model = quantization.fit(os.path.join(args.input_model, model),
                                           conf=conf,
                                           calib_dataloader=dataloader
                                           )
                q_model.save(os.path.join(args.output_model, model))
        else:
            conf = PostTrainingQuantConfig(approach="dynamic",
                    op_type_dict={'^((?!(MatMul|Gather|Conv)).)*$': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}}},)
            for model in model_list:
                q_model = quantization.fit(os.path.join(args.input_model, model),
                                           conf=conf)
                q_model.save(os.path.join(args.output_model, model))

    if args.accuracy_only:
        eval(args.input_model)
    
    if args.benchmark:
        config = PretrainedConfig.from_pretrained(args.model_name_or_path)
        predictions = []
        references = []
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = args.cores_per_instance
        sessions = ORTModelForSpeechSeq2Seq.load_model(
                os.path.join(args.input_model, 'encoder_model.onnx'),
                os.path.join(args.input_model, 'decoder_model.onnx'),
                os.path.join(args.input_model, 'decoder_with_past_model.onnx'),
                session_options=sess_options)
        model = ORTModelForSpeechSeq2Seq(sessions[0], sessions[1], config, args.input_model, sessions[2])

        librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test", cache_dir=args.cache_dir)
        processor = WhisperProcessor.from_pretrained(args.model_name_or_path)
        total_time = 0
        for idx, batch in enumerate(librispeech_test_clean):
            if idx > args.iters:
                break
            audio = batch["audio"]
            input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
            tic = time.time()
            predicted_ids = model.generate(input_features, max_new_tokens=args.max_new_tokens)
            toc = time.time()
            if idx >= args.warmup:
                total_time += (toc - tic)
        latency = total_time / (args.iters - args.warmup)
        print('Latency: %.3f ms' % (latency * 1000))
        print('Throughput: %.3f images/sec' % (args.batch_size / latency))
        print('Batch size = %d' % args.batch_size)
