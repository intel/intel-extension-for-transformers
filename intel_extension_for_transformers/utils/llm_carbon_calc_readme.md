# LLM Carbon Calculator 
llm_carbon_calc.py is a simple calculator script of LLM infernce carbon emission. 
User can use it to calculate rough carbon emission in one infernce procedure of LLMs given some parameters.
The calculate formula is quite simple, you can refer to script source code for more detail. 

## Usage

llm_carbon_calc.py [-h] [-c C] [-t T] [--fl FTL] [--nl NTL] [-n N] --ct CT -m M

LLM carbon calculator - simple calculator of LLM inference carbon emission

options:<br>
  &emsp; &emsp; -h, --help &emsp; &emsp; &emsp; &emsp; &emsp; &ensp; show this help message and exit <br>
  &emsp; &emsp; -c C, --carbon-intensity C <br>
  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; carbon intensity of electricity of your country or cloud provider (default: 0.475 - world average)<br>
  &emsp; &emsp; -t T, --time T &emsp; &emsp; &emsp; &emsp; &nbsp; total time of one inference procedurein mini-seconds<br>
  &emsp; &emsp; --fl FTL, --first-latency FTL<br>
  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; first token latency in mini-seconds<br>
  &emsp; &emsp; --nl NTL, --next-latency NTL<br>
  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; next token latency in mini-seconds<br>
  &emsp; &emsp; -n N, --token-size N &emsp; &ensp; &nbsp; output token number in one inference (default: 32)<br>
  &emsp; &emsp; --ct CT, --cpu-tdp CT<br>
  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; CPU TDP in Watts<br>
  &emsp; &emsp; -m M, --mem M &emsp; &emsp; &emsp; &nbsp; memory consumption in MB<br>

## Examples

When the total time of one inference procedure is measured, you can specific -t or --time like following command line

        $python llm_carbon_calc.py -m 27412.98 --ct 350 -c 0.56 -t 6510.3
        CPU TDP (W):  350
        Memory Consumption (MB) 27412.98
        Output token number:  32
        Total time of one infernce (ms) 6510.3
        Carbon emission in one inference (kgCO2e) 0.0003575115963544682

When the total time of one inference procedure is not measured but first token latency and next token latency are measured, you can refer to following example

        $ python llm_carbon_calc.py -m 27412.98 --ct 350 -c 0.56 --fl 2284.75 --nl 136.31 -n 32
        CPU TDP (W):  350
        Memory Consumption (MB) 27412.98
        Output token number:  32
        Total time of one infernce (ms) 6510.36
        Carbon emission in one inference (kgCO2e) 0.00035751489124038457