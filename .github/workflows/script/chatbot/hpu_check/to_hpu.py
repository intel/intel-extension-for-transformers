import torch
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

ly = torch.nn.Linear(2, 4)
ly.to("hpu")
print("hpu is available")

