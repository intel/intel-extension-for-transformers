
# LLM Carbon Calculator 
llm_carbon_calc.py is a simple calculator script of LLM inference carbon emission. 
User can use it to calculate rough carbon emission in one inference procedure of LLMs given some parameters.
The calculate formula is quite simple, you can refer to script source code for more detail. 

## Usages
```
$ python llm_carbon_calc.py -h
usage: llm_carbon_calc.py [-h] [-c C] [-t T] [--fl FTL] [--nl NTL] [-n N] --tdp TDP -m M

LLM carbon calculator - simple estimator of LLM inference carbon emission

options:
    -h, --help            show this help message and exit
    -c C, --carbon-intensity C
                          carbon intensity of electricity of your country or cloud provider (default: 0.475 - world average)
    -t T, --time T        total time of one inference procedurein mini-seconds
    --fl FTL, --first-latency FTL
                          first token latency in mini-seconds
    --nl NTL, --next-latency NTL
                          next token latency in mini-seconds
    -n N, --token-size N  output token number in one inference (default: 32)
    --tdp TDP             device TDP in Watts, it could be CPU/GPU/Accelerators
    -m M, --mem M         memory consumption in MB
```

## Examples

When the total time of one inference procedure is measured, you can specific -t or --time like following command line
```
    $python llm_carbon_calc.py -m 27412.98 --tdp 350 -c 0.56 -t 6510.3
    TDP (W):  350
    Memory Consumption (MB): 27412.98
    Output token number:  32
    Total time of one inference (ms): 6510.3
    Carbon emission in one inference (kgCO2e): 0.0003575115963544682
```
When the total time of one inference procedure is not measured but first token latency and next token latency are measured, you can refer to following example
```
    $ python llm_carbon_calc.py -m 27412.98 --tdp 350 -c 0.56 --fl 2284.75 --nl 136.31 -n 32
    TDP (W):  350
    Memory Consumption (MB): 27412.98
    Output token number:  32
    Total time of one inference (ms): 6510.36
    Carbon emission in one inference (kgCO2e): 0.00035751489124038457
```    
