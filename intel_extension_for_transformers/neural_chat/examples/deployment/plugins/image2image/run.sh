# KMP
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# OMP
export OMP_NUM_THREADS=56
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so


# tc malloc
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

numactl -l -C 0-55 python -m inference_server 2>&1 | tee run.log
