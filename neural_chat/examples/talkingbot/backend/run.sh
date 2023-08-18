# Kill the exist and re-run
ps -ef |grep 'talkingbot' |awk '{print $2}' |xargs kill -9

# KMP
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# OMP
export OMP_NUM_THREADS=56
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so

# tc malloc
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

numactl -l -C 0-55 python -m talkingbot --model-path "meta-llama/Llama-2-7b-chat-hf" 2>&1 | tee run.log
