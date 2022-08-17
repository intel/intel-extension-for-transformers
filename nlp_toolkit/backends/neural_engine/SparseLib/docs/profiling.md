### Performance and Profiling
* We support a brief verbose logging for kernel execution
```shell
SPARSE_LIB_VERBOSE=1 ./test_spmm_vnni_kernel
```

* For advanced users we also support vtune profling for kernels execution through [ITT Tasks](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/api-support/instrumentation-and-tracing-technology-apis/basic-usage-and-configuration/viewing-itt-api-task-data.html), to enable it you can follow the instructions:

```shell
mkdir build
cd build
cmake .. -DSPARSE_LIB_USE_VTUNE=True
# if the path to vtune is not the default /opt/intel/oneapi/vtune/latest, you can determine the path manually like
# cmake .. -DSPARSE_LIB_USE_VTUNE=True -DCMAKE_VTUNE_HOME=/path/to/vtune
make -j
...
SPARSE_LIB_VTUNE=1 ./{executable}
```
