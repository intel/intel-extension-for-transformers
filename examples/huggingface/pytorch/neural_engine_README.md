# Launcher Script
We develop a launcher script to support typical usage of AI workload measurement 
To get the minimum latency or maximum throughput of an example is not easy. Typically, there are some tuning knbos, including instance number, cores per instance, batch size, memory settings, etc. To simplify the usage, we develop an automatic launcher script to measure the performance easily.

## Usage
Here is the usage of launcher script:
```
python -m launcher [launcher_parameters] model_script.py [script_args]
```

## Parameters
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Value Description</th>
            <th><b>Default Value</b></th>
        </tr>
    </thead>
        <tr>
            <td align="center">mode</td>
            <td>min_latency: minimum latency without core/instance constraints </br>max_throughput: maximum throughput without core/instance constraints</br>default_throughput: instance per socket and multi-instances</br>default_latency: 4 cores per instance and multi-instances</td>
            <td align="center"><b>default_throughput</b></td>
        </tr>
        <tr>
            <td align="center">instance_num</td>
            <td>1,2,4,8: use instance number 1, 2, 4, 8</br>auto: enable automatic instance tuning</br>1: use instance number 1</td>
            <td align="center"><b>         1</b></td>
        </tr>
        <tr>
            <td align="center">latency_constraint</td>
            <td>10: latency constraint </td>
            <td align="center"><b>         0</b></td>
        </tr>
        <tr>
            <td align="center">batch_size</td>
            <td>1,32: use batch size 1, 32 </br>auto: enable automatic batch size tuning</td>
            <td align="center"><b>1</b></td>
        </tr>
        <tr>
            <td align="center">memory_allocator</td>
            <td>jemalloc: use jemalloc</br>default: use default malloc</br>auto: enable automatic memory allocator tuning</td>
            <td align="center"><b>default</b></td>
        </tr>
        <tr>
            <td align="center">weight sharing</td>
            <td>enabled: enable weight sharing across instances</br>disabled</br>auto: enable automatic weight sharing tuning</td>
            <td align="center"><b>disabled</b></td>
        </tr>
        <tr>
            <td align="center">memory_planning</td>
            <td>unified_buffer: memory planning by a unified buffer</br>cycle_buffer: memory planning by cycle buffers</br>auto: enable autmatic memory planning tuning</td>
            <td align="center"><b>cycle_buffer</b></td>
        </tr>
        <tr>
            <td align="center">output_file</td>
            <td>output csv file</td>
            <td align="center"><b>out.csv</b></td>
        </tr>

</table>


## Typical Usage
Default model script
```
python mrpc/bert_mini/run_executor.py  --input_model=mrpc/bert_mini/ref_model --mode=performance --batch_size=8 --seq_len=128 --iteration=10 --warm_up=5
```

Max throughput
```
python -m launcher --mode=max_throughput  --latency_constraint=0.9  mrpc/bert_mini/run_executor.py  --input_model=mrpc/bert_mini/ref_model --mode=performance --batch_size=8 --seq_len=128 --iteration=10 --warm_up=5
```

Min latency
```
python -m launcher --mode=min_latency mrpc/bert_mini/run_executor.py  --input_model=mrpc/bert_mini/ref_model --mode=performance --batch_size=8 --seq_len=128 --iteration=10 --warm_up=5
```

Default latency
```
python -m launcher --mode=default_latency  mrpc/bert_mini/run_executor.py  --input_model=mrpc/bert_mini/ref_model --mode=performance --batch_size=8 --seq_len=128 --iteration=10 --warm_up=5
```

Default throughput
```
python -m launcher --mode=default_throughput  mrpc/bert_mini/run_executor.py  --input_model=mrpc/bert_mini/ref_model --mode=performance --batch_size=8 --seq_len=128 --iteration=10 --warm_up=5
```

## Automatic Performance Tuning
Here is an example to enable automatic tuning for all the supported parameters:
```
python -m launcher --mode=max_throughput --instance_num=auto --batch_size=auto --weight_sharing=auto --memory_planning=auto --memory_allocator=auto --latency_constraint=0.9  mrpc/bert_mini/run_executor.py  --input_model=mrpc/bert_mini/ref_model --mode=performance --batch_size=8 --seq_len=128 --iteration=10 --warm_up=5
```

## Output
Here is a sample output which describes the performance result and reproducible command:</br>
| batch  | instance | cores per instance | Throughput | Average Latency | P50 Latency | P90 Latency | P99 Latency | cmds |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|7  |4  |347.26 |55.06 |50.03 | 58.01 | 59.11 | OMP_NUM_THREADS=2 numactl --localalloc --physcpubind=0,1... | 

P50 latency: the 50th latency percentile (namely 50% of the requests faster than p50 value)</br>
P90 latency: the 90th latency percentile (namely 90% of the requests faster than p90 value)<br/>
P99 latency: the 99th latency percentile (namely 99% of the requests faster than p99 value)<br/>
<br />

Please note that the output file records all the commands during automatic tuning, and therefore users can reproduce easily.
```
UNIFIED_BUFFER=1 OMP_NUM_THREADS=2 numactl --localalloc --physcpubind=18,19  /home/xxx/anaconda3/envs/bin/python -u mrpc/bert_mini/run_executor.py --input_model=mrpc/bert_mini/ref_model --mode=performance --batch_size=8 --seq_len=128 2>&1|tee /home/xxx/test/intel/intel-extension-for-transformers/examples/deploy/ref_executor/12_2_9_disabled_default_unified_buffer.log &
```
For more sample outputs, please refer to the [example](details.csv).<br>

