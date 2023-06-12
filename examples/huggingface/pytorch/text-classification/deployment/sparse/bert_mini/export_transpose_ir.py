from intel_extension_for_transformers.backends.neural_engine.compile import compile
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', default="./model_and_tokenizer/int8-model.onnx",
                        type=str, help="Input model path.")
    parser.add_argument('--output_dir',
                        help='directory to save data to',
                        type=str, default='./sparse_int8_ir')
    args = parser.parse_args()

    graph = compile(args.input_model,'./bertmini_sparse_pattern.conf')
    graph.save()
    model = Graph()
    model.graph_init('./ir/conf.yaml', './ir/model.bin', load_weight=True)
    model.transpose_mode_int8()
    model.save(args.output_dir)
