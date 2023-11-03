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
- Use pylint to check your Python code, more detail please go to [pylint script](../.github/workflows/script/formatScan/pylint.sh)

### Pull Request Template
See [PR template](../.github/PULL_REQUEST_TEMPLATE/pull_request_template.md)

### Pull Request Acceptance Criteria

(1) At least two approvals from reviewers (1 approval from the component [owner](./component_owner.md))

(2) ALL detected CI checks pass

(3) ALL conversations solved

(4) Third-party dependency license Compatible


## CI Introduction
See [CI Introduction](./CI_introduction.md)

## FAQ


1. How to check third-party dependency license compliance
	
    If you are using third-party component by: (1) import lib and uses its API (2) static/dynamic linked (3) copy code and modified (4) other uncertain usage, please contact [Maintainers](mailto:itrex.maintainers@intel.com) to check license compliance before push code to repo.


## Support

Submit your questions, feature requests, and bug reports to the
[GitHub issues](https://github.com/intel/intel-extension-for-transformers/issues) page. You may also reach out to [Maintainers](mailto:itrex.maintainers@intel.com).


## Contributor Covenant Code of Conduct

This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md).
