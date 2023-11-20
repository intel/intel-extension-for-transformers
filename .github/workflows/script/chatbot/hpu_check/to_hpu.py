import torch

ly = torch.nn.Linear(2, 4)
ly.to("hpu")
print("hpu is available")

