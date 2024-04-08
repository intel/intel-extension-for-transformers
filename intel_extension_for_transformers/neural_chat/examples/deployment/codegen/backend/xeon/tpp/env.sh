torch_ccl_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))" 2> /dev/null)
source $torch_ccl_path/env/setvars.sh

export TPP_CACHE_REMAPPED_WEIGHTS=0
export USE_MXFP4=1
export KV_CACHE_INC_SIZE=512

export KMP_AFFINITY=compact,1,granularity=fine
export KMP_BLOCKTIME=1
export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so
