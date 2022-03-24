import nncf
import torch
from torch import Tensor, device, nn


NNCF_PT_STATE_NAME = "nncf_state.bin"

def get_nncf_train_dataloader_for_init(args, train_dataset, data_collator=None):
    from torch.utils.data import RandomSampler
    from torch.utils.data import DistributedSampler
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )

    if data_collator is None:
        from transformers.data.data_collator import default_data_collator
        data_collator = default_data_collator

    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=args.dataloader_drop_last,
    )
    return data_loader


@nncf.torch.register_module()
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
