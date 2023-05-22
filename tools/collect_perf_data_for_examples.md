# Collect Performance Data for the Xetla Examples

There are some examples in the examples folder in the Xetla folder. This document is about how to collect the performance data for those examples.

## Record the performance data.

When running the example, you can add the option `--gtest_output=json` to dump the performance data to a JSON file. The JSON file's name is `test_detail.json`

In the JSON file, there are some sections like 

```json
":kernel_time:ms": "minimum:maximum:median:mean",
":kernel_time:gflops": "minimum:maximum:median:mean",
"kernel_time:ms:fused_gemm": "1.1 5.5 3.3 4.4",
"kernel_time:gflops:fused_gemm": "10.1 50.5 30.3 40.4",

```

The key starts with ":" indicates a performance indicator. 

For example, `":kernel_time:ms"` means the performance indicator `kernel_time` with the unit `ms`.

`":kernel_time:gflops"` means the performance indicator `kernel_time` with the unit `gflops`

`"minimum:maximum:median:mean"` means there are 4 filed for those indicator.

Then the key `"kernel_time:ms:fused_gemm"` is the performance data for the kernel `fused_gemm`

## Collect the performance data to a csv file

We may want to run examples in different scenarios, like the driver version, and record all performance data in a file to look at its trends.

We can use the `perf.py` to do it. The usage is `perf.py CSV_FILE RESULT [--meta=meta_info_file]`

The script uses the python package docopt. Use the `pip install docopt` to install it.

**`CSV_FILE` is the CSV file path to write the performance data into.** 

1. If the file doesn't exist, it creates a new file. And all the fields of performance indicators are written into the file. And the performance data for all the test scenarios and kernels are written into the file.

2. If the file exists, it attaches the performance data at the end of the file.

   And only the existing fields and existing performance data will be written into the CSV file.

The usage mode could be like

1. Use `perf.py` to create a new CSV file for one or several examples.
2. Edit the CSV file and delete the unwanted columns and rows. Then, those data will be skipped in the future.

**`RESULT` is the path to find the `test_detail.json`.** 

You can use a file path or a directory path. If using the directory path, all the  `test_detail.json` under the directory will recursively be collected.

**`--meta=meta_info_file` is to attach other meta information.**

You may also want to track the driver or application versions when you run the example multiple times. You can bring that information into the CSV file using a JSON file like this.

```json
{
    "time": "2023.1.1",
    "driver ver": "v12.3",
    "compiler ver": "dpcpp.1234"
}
```

You can define any fields and put the values you want. The `perf.py` will write all those fields and values to the CSV file unchanged. 

