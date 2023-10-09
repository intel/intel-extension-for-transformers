# Kill the exist and re-run
ps -ef |grep 'controller' |awk '{print $2}' |xargs kill -9
ps -ef |grep 'model_worker' |awk '{print $2}' |xargs kill -9

python -m controller --port 8000 &
sleep 10

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
numactl -l -C 0-31 python -m model_worker --ipex --model-path /home/tme/mpt-7b-chat --controller-address http://localhost:8000 --worker-address http://localhost:8080 --device "cpu"