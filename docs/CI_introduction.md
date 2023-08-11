# CI Introduction
Intel Extension for Transformers use Github Action (GHA) for CI test, if you are new to GHA, please refer to [GHA](https://docs.github.com/en/actions)
All the test scripts will be maintained under [.github](../.github/workflows).



|     Test Name                 |     Test Pass Criteria                        |     Checks    |
|-------------------------------|-----------------------------------------------|---------------|
|     Format Scan               |     Pylint/Cpplint/bandit/cloc Pass           |     4         |
|     Spell Check               |     Check Pass                                |     1         |
|     Copyright Check           |     Add License on top of code/scripts        |     1         |
|     DCO                       |     Use git commit -s to signoff              |     1         |
|     Unit Test    |     a. No failure, no core dump, no segmentation fault b. No coverage drop|     5         |
|     Kernel Benchmark          |     No failure, No performance regression              |     1         |
|     Model Test       |     a. Fp32/Int8 inference no regression  b. API no fucntionality success |     5         |
|     CPP Graph Test            |     No failure, No performance regression      |     1         |

