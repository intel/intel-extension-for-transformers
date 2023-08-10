Contribution Guidelines
=======================

1. [General](#General)

    1.1 [Pull Request Checklist](#pull-request-checklist)

    1.2 [Pull Request Template](#pull-request-template)

    1.3 [Pull Request Acceptance Criteria](#pull-request-acceptance-criteria)

2. [CI Introduction](#ci-introduction)

3. [FAQ](#faq)

4. [Support](#support)

5. [Contributor Covenant Code of Conduct](#contributor-covenant-code-of-conduct)


## General

If you have improvements to Intel® Extension for Transformers, pull requests for
[review](https://github.com/intel/intel-extension-for-transformers/pulls). If you are new to Github, view the pull request [How To](https://help.github.com/articles/using-pull-requests/).


### Pull Request Checklist


Before sending your pull requests, follow the information below:

- Changes are consistent with the Python [Coding Style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).
- Use pylint to check your Python code.
- Use flake8 and autopep8 to make Python code clean.

### Pull Request Template
See [PR template](../.github/PULL_REQUEST_TEMPLATE/pull_request_template.md)

### Pull Request Acceptance Criteria

(1) At least two approval from reviewers (1 approval from the component [owner](../.github/CODEOWNERS))

(2) ALL detected CI checks pass

(3) ALL conversation solved


## CI Introduction



|     Test Name                 |     Test Pass Criteria                        |     Checks    |
|-------------------------------|-----------------------------------------------|---------------|
|     Format Scan               |     Pylint/Cpplint/bandit/cloc Pass           |     4         |
|     Spell Check               |     Check Pass                                |     1         |
|     Copyright Check           |     Check Pass                                |     1         |
|     DCO                       |     Check Pass                                |     1         |
|     Optimize UT + Coverage    |     a. No failure, no core dump, no segmentation fault b. No coverage drop|     1         |
|     Engine UT + Coverage      |     a. No failure, no core dump, no segmentation fault b. No coverage drop|     3         |
|     Kernel UT                 |     No failure, no core dump, no segmentation fault   |     1         |
|     Kernel Benchmark          |     No failure, No performance regresion              |     1         |
|     Optimize Model Test       |     a. Quantize success  b. Fp32 Benchmark/Throughput/Accuracy success and no regression  c. Int8 Benchmark/Throughput/Accuracy success and no regression|     2         |
|     Backend Model Test        |     a. Fp32 inference no regression   b. Int8 inference no regression  c. C++ API inference no regression|     2         |
|     LLM Test                  |     No failure, No performance regresion      |     1         |
|     CPP Graph Test            |     No failure, No performance regresion      |     1         |


## FAQ


1. How to add third-party dependency
	
    If you are using third-party component by: (1) import lib and uses its API (2) static/dynamic linked (3) copy code and modified (4) other uncertain usage, please contact [Maintainers](inc.maintainers@intel.com) to check license compliance before push code to repo.


2. How to apply for repo access

    Create a [ticket](https://opensource.intel.com/jira/servicedesk/customer/portal/1/create/29) to join “intel-extension-for-transformers-write” team. 

    if you don't have permision to create ticket, please contact [Maintainers](inc.maintainers@intel.com) to apply for help.


## Support

Submit your questions, feature requests, and bug reports to the
[GitHub issues](https://github.com/intel/intel-extension-for-transformers/issues) page. You may also reach out to [Maintainers](inc.maintainers@intel.com).


## Contributor Covenant Code of Conduct

This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md).
