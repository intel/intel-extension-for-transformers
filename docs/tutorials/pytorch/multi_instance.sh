set -eo pipefail
set -x
PATTERN='[-a-zA-Z0-9_]*='
for i in "$@"
do
    case $i in
        --model=*)
            model=`echo $i | sed "s/${PATTERN}//"`;;
        --core_per_instance=*)
            core_per_instance=`echo $i | sed "s/${PATTERN}//"`;;
        --data_type=*)
            data_type=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done
ncores_per_socket=${ncores_per_socket:=$( lscpu | grep 'Core(s) per socket' | cut -d: -f2 | xargs echo -n)}
log_name="${model}.log"
cmd="python benchmark.py --data_type=${data_type} --model_name_or_path=${model}"
echo "Executing multi instance benchmark"
echo -e ">>> Executing multi instance benchmark $core_per_instance $cmd" >>"$log_name"
for ((j = 0; $(($j + $core_per_instance)) <= $ncores_per_socket; j = $(($j + ${core_per_instance})))); do
    numa_prefix="numactl -m 0 -C $j-$((j + core_per_instance - 1)) "
    # Make it works on machines with no numa support
    if [[ -n $(numactl -s | grep "No NUMA support available") ]]; then
        echo "No NUMA support available"
        echo "Please install numactl"
        exit 1
    fi
    echo "${numa_prefix}${cmd}" >>$log_name
    ${numa_prefix}${cmd} |
        tee -a $log_name &
done
wait
echo -e "<<< Executing multi instance benchmark $core_per_instance $2" >>"$log_name"

status="SUCCESS"

for pid in "${benchmark_pids[@]}"; do
    wait $pid
    exit_code=$?
    echo "Detected exit code: ${exit_code}"
    if [ ${exit_code} == 0 ]; then
        echo "Process ${pid} succeeded"
    else
        echo "Process ${pid} failed"
        status="FAILURE"
    fi
done
echo "Benchmark process status: ${status}"
if [ ${status} == "FAILURE" ]; then
    echo "Benchmark process returned non-zero exit code."
    exit 1
fi
Total_Throughput=$(cat $log_name | grep -Po "Throughput:\s+(\d+(\.\d+)?)" | cut -f 2 -d ' ' | awk '{ SUM += $1} END { print SUM }')
echo "Throughput : $Total_Throughput"
Batch_size=$(cat $log_name | grep -Po "Batch\s+size\s+=\s+\d+" | tail -1)
echo $Batch_size
Accuray=$(cat $log_name | grep -Po "Finally Eval .* Accuracy.*\d+" | tail -1)
echo $Accuray
Total_Latency=$(cat $log_name | grep -Po "Latency:\s+(\d+(\.\d+)?)" | cut -f 2 -d ' ' | awk '{ SUM += $1} END { print SUM }')
echo "Latency : $Total_Latency"