from intel_extension_for_transformers.backends.neural_engine.compile import compile
from intel_extension_for_transformers.backends.neural_engine.compile.ops.tensor import Tensor
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
import numpy as np
import os
import shutil
import wget
import torch
import torch.nn as nn
import unittest


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(30, 50)

    def forward(self, x):
        x = self.linear(x)
        return x

class TestNeuralEngine(unittest.TestCase):
    def test_Bert_Mini_int8_Onnx_Neural_Engine(self):
        bert_mini_int8_onnx_model_url = 'https://huggingface.co/Intel/bert-mini-sst2-distilled-sparse-90-1X4-block/resolve/main/int8-model.onnx'
        try:
            filename = wget.download(bert_mini_int8_onnx_model_url)
        except:
            print(
                    "The onnx model was not successfully downloaded, therefore test may cannot run"
                )
            return
        
        model = compile(filename)
        input_0 = np.random.randint(0, 384, (1, 32)).reshape(1, 32)
        input_1 = np.random.randint(1, 2, (1, 32)).reshape(1, 32)
        input_2 = np.random.randint(1, 2, (1, 32)).reshape(1, 32)
        # test of inference
        out = model.inference([input_0, input_1, input_2])
        self.assertEqual(1, len(out))
        os.remove(filename)


    def test_torch_model_Neural_Engine(self):
        n = Net()
        example_in = torch.rand(3, 30)
        traced_model = torch.jit.trace(n, example_in)
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        torch.jit.save(traced_model, '{}.pt'.format(file_name))
        ref_out = traced_model(example_in).detach().numpy()

        graph = compile('{}.pt'.format(file_name))
        graph.save(file_name)
        newgraph = Graph()
        newgraph.graph_init(file_name + '/conf.yaml', file_name + '/model.bin')
        out = newgraph.inference([example_in.numpy()])
        
        np.testing.assert_almost_equal(ref_out, [*out.values()][0], decimal=5)
        os.remove('{}.pt'.format(file_name))
        shutil.rmtree(file_name)
        
        
if __name__ == "__main__":
    unittest.main()
