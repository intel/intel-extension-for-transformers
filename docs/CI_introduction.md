# CI Introduction
Intel Extension for Transformers use Github Action (GHA) for CI test, if you are new to GHA, please refer to [GHA](https://docs.github.com/en/actions)


Generally We use [Azure Cloud](https://azure.microsoft.com/en-us/pricing/purchase-options/pay-as-you-go) to deploy CI on Es-v5, Bas-v2. 


|     Test Name                 |     Test Scope                        |     Test Pass Criteria    |
|-------------------------------|-----------------------------------------------|---------------|
|     Format Scan               |     Pylint/Cpplint/bandit/cloc/clangformat Pass           |     PASS         |
|     Spell Check               |     Spelling Check                               |     PASS         |
|     Copyright Check           |     Copyright Check        |     PASS         |
|     [DCO](https://github.com/apps/dco/)                       |     Use git commit -s to sign off              |     PASS         |
|     Unit Test    |   pytest + coverage + gtest |     PASS(No failure, No core dump, No segmentation fault, No coverage drop)         |
|     Kernel Benchmark          |      Benchmark [Details](../intel_extension_for_transformers/llm/runtime/deprecated/test/kernels/benchmark)               |     PASS(No performance regression)        |
|     Model Test       |   Pytorch + Tensorflow + Neural Engine + IPEX + CPP Graph   |     PASS(FP32/INT8 No performance regression)         |

# FAQ
1. How to add examples

    (1) Still need to pass detected CI tests

    (2) Use [Extension Test](https://inteltf-jenk.sh.intel.com/view/nlp-toolkit-validation/job/nlp-toolkit-validation-top-mr-extension/) to cover new examples. If you do not have permission for extension test, please contact [Maintainers](inc.maintainers@intel.com)


2. How to test specific component version

    (1) Use [Extension Test](https://inteltf-jenk.sh.intel.com/view/nlp-toolkit-validation/job/nlp-toolkit-validation-top-mr-extension/) to specify the version. If you do not have permission for extension test, please contact [Maintainers](inc.maintainers@intel.com)

3. How to run tests locally

    Please refer to test scripts and usage [README](../.github/workflows/README.md)
