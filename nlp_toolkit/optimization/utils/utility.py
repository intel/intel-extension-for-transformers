import importlib
import os
from neural_compressor.utils.utility import LazyImport

def distributed_init(backend="gloo", world_size=1, rank=-1, init_method=None,
                     master_addr='127.0.0.1', master_port='12345'):
    torch = LazyImport("torch")
    rank = int(os.environ.get("RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", world_size))
    if init_method is None:
        master_addr = os.environ.get("MASTER_ADDR", master_addr)
        master_port = os.environ.get("MASTER_PORT", master_port)
        init_method = 'env://{addr}:{port}'.format(addr=master_addr, port=master_port)
    torch.distributed.init_process_group(
        backend,
        init_method=init_method,
        world_size=world_size, 
        rank=rank
    )