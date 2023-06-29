# Kill the exist and re-run
ps -ef |grep 'fastrag_service' |awk '{print $2}' |xargs kill -9

# KMP
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# OMP
export OMP_NUM_THREADS=32
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so

# tc malloc
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

#python -m fastrag_service --model-path ./llama-alpaca-fastrag 2>&1 | tee run.log
numactl -l -C 0-31 python -m fastrag_service --model-path ./mpt-7b-chat/ 2>&1 | tee run.log
