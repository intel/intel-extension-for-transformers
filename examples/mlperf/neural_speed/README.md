# MLPerf2024

We used Neural Speed, an innovated library designed to provide the most efficient inference of LLMs, for MLPerf 2024 submission. Here, we provide the script to setup the runtime environment.

Please check [our mlperf submission (May not be available yet)](https://github.com/mlcommons/inference_results_v4.0/tree/main/closed/Intel/code/gptj-99/ITREX/README.md) for detailed instructions.

For similar applications, the following code can prepare Neural Speed on the environment.

```bash
rm -rf neural-speed
git clone https://github.com/intel/neural-speed.git -b mlperf-v4-0
pip install -r neural-speed/requirements.txt
pip install -ve ./neural-speed
```
