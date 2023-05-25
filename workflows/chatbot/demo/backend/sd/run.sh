#!/bin/bash

# IOMP
export OMP_NUM_THREADS=32
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so

# Jemalloc
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

# Cache clean
sync; sh -c "echo 3 > /proc/sys/vm/drop_caches"


# Kill the exist and re-run
ps -ef | grep 'inference_server_ipex.py' |awk '{print $2}' | xargs kill -9
numactl -l -C 0-31 python inference_server_ipex.py 2>&1 | tee run.log
