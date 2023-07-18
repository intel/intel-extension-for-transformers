# Kill the exist and re-run
ps -ef |grep 'controller' |awk '{print $2}' |xargs kill -9
ps -ef |grep 'model_worker' |awk '{print $2}' |xargs kill -9

python -m controller &

# KMP
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# OMP
export OMP_NUM_THREADS=32
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so

# tc malloc
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

# ipex
# numactl -l -C 0-31 python -m model_worker --ipex --model-path ./llama-7b-hf-conv --controller-address http://localhost:80 --worker-address http://localhost:8080 --device "cpu"
numactl -l -C 0-31 python -m model_worker --ipex --model-path ./mpt-7b-chat --controller-address http://localhost:80 --worker-address http://localhost:8080 --device "cpu"