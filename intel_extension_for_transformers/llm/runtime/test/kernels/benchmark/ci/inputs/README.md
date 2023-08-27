# Inputs format
- First non-empty line should starts with 3 `#`s, labeling each input arguments divided by whitespace (lables cannot contain whitespace)
- List test cases line by line
    - First argument should always be the number of cores per instance
    - Following arguments will be append to benchmark command after the `kernel_type` (skipping `mode` (both will be run) and `kernel_type` (defined in `run_ci.sh`))
- Lines starting with `$` are considered as environment variable overwriting, affecting following execution of `benchmark.sh`. e.g. `$it_per_core` will change the number of iterations to be run for following test cases
- Lines starting with `#` are commited and will be discard during testing
- Benchmarking
    - To run benchmark with a new input file, run 
      ``` bash
      ./benchmark.sh --modes=<acc,perf> --op=<kernel_type> --batch=<new_input_file>  --medium_n=[medium_n] --it_per_core=[it_per_core]
      ```
      where,
      - `kernel_type` is the same as that for [benchmark](../../README.md#usage)
      - `medium_n` is the number of number benchmarking repeated to get the medium of GFLOPS
      - `it_per_core` times number of cores per instance is the number of iterations per benchmarking
    - To add a new input file to CI, add an additional command to [`run_ci.sh`](../run_ci.sh)
