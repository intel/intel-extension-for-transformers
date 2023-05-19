# Contributing Guidelines

## Code Contribution Guidelines

When submitting your contribution, please make sure that it is:

* *Tested*: Intel® XeTLA uses gtests for lightweight functional testing. Be sure to extend the existing tests when fixing an issue.
* *Documented*: Intel® XeTLA uses Doxygen for inline comments in public header files that is used to build reference manual and markdown (also processed by Doxygen) for user guide.

All code in Intel® XeTLA gets promoted to main branch only through GitHub pull requests. Requirements for promotion:

- The request is reviewed and approved by maintainers for all affected components.
- All discussions in the pull request are resolved.
- Continuous integration pipeline passed without errors.
- The pull request author is responsible for collecting all the necessary approvals, rebasing of the changes, and resolving the discussions.

To simplify the work of reviewers, make sure that the commits in the pull request adhere to the following requirements:

- Commit message should follow the format:
  `<scope>: <short description>`
  Scope examples:
  * `feature`, `api`, `doc`, `tests`, `common`
  * Example commit message:
    ~~~git
      doc: update environment section in readme
    ~~~
* Intel® XeTLA branches maintain linear history. Rebase the changes on top of target branch before creating a pull request. Rebase again after resolving all the discussions, as well as in case of merge conflicts.

- Use `git add -p`  and `git rebase -i` liberally to split unrelated changes into multiple self-contained commits. This is a courtesy to reviewers: smaller commits are easier to comprehend. It also helps with bisecting in the future. Of course judgement needs to be applied whether to split changes or not. For example, split code cleanup and the actual fix into two separate patches.

## Coding Standards

### Automatic Detection

Intel® XeTLA uses [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) in order to diagnose and fix common style violations and easy-to-fix issues in the code base. For instructions on how to use `clang-tidy`, please refer to the [clang-tidy RFC](https://github.com/oneapi-src/oneDNN/blob/rfcs/rfcs/20200813-clang-tidy/README.md).
The list of clang-tidy checks the Intel® XeTLA code base follows is available in the `.clang-tidy` file found in the Intel® XeTLA top-level directory.

### Automatic Doxygen Comments Generation
If you use vscode, you can use *Doxygen Documentation Generator* plugin to automatically generate the comments.
For consistency, please set Comment Prefix to `/// `, First Line to ` `, Last line to `/// `


### Coding Style

The coding style consistency in Intel® XeTLA is maintained using `clang-format`. When submitting your contribution, please make sure that it adheres to the existing coding style with the following command:
```sh
clang-format-14 -style=file:tools/clang-format/_clang-format -i foo.cpp
```
This will format the code using file [_clang-format](./tools/clang-format/_clang-format).

A pre-commit hook is also provided to ensure you do the clang-format when doing git commit, please refer to [enable clang-format in git](./tools/scripts/clang_format.sh).

### General

- Use properly named constants whenever possible (unless this code is auto-generated).
  * For example,
    ~~~cpp
    if (x < 4) y = 64;
    ~~~
  
    In this example, 4 and 64 should be named, in which case the code becomes:
    ~~~cpp
    if (x < sizeof(float)) y = cache_line_size;
    ~~~
  
- Don't use `using namespace XXX` in header files.

- Avoid code duplication (look for similarities), unless it is necessary.

- Declare variables in the innermost possible scope until there are some circumstances that make you declare them somewhere else.
  
- Consider using utils to improve readability (`IMPLICATION`, `one_of`,`everyone_is`).
