# How to setup clang-format hook

## Install clang-format

Make sure you've install clang-format and put the executable in the **PATH**

- Windows

  Find windows pre-build binary from [LLVM release page]( http://releases.llvm.org/download.html )

- Linux

  ```
  sudo apt-get install clang-format
  ```

## Configure pre-commit

`pre-commit` is a script that runs automatically when we use `git commit`, that is the time that we want to do clang-format for the files that we changed.

The `pre-commit` should be found in `.git/hooks/pre-commit`, if we don't have such a file in that folder, it means you didn't configure any pre-commit action before, please create one.

Copy the content of `$REPO_ROOT/hooks/clang-format.hook` and paste it into **file** .`git/hooks/pre-commit`.

If you are using linux, also add the execution permission to the file

```
cp hooks/clang-fomat.hook .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Example

If you make a change to file with suffix **.cpp**/ **.h**/ **.hpp**

During your commit, the pre-commit hook will run and do clang-format on the files

In the following example, we did some change to the file format, but after commit we can see there's no diff, because this file has been formatted before, and the format change would be clear by the hook.

```
root@chengxi_gpu_sim_dev_ww19:~/working_dir/libraries.gpu.xetla# git diff
diff --git a/include/core/xetla_core_raw_send.hpp b/include/core/xetla_core_raw_send.hpp
index a1a0c76..49c9fb2 100644
--- a/include/core/xetla_core_raw_send.hpp
+++ b/include/core/xetla_core_raw_send.hpp
@@ -165,8 +165,7 @@ __XETLA_API void xetla_raw_send(__XETLA_CORE_NS::xetla_vector<T1, n1> msgSrc0,
     __ESIMD_EXT_NS::raw_sends_load(msgSrc0, msgSrc1, exDesc, msgDesc, execSize,
             sfid, numSrc0, numSrc1, isEOT, isSendc, mask);
 #else
-    cm_raw_send(0, msgSrc0, msgSrc1, exDesc, msgDesc, execSize, sfid, numSrc0,
-            numSrc1, 0, isEOT, isSendc, mask);
+    cm_raw_send(0, msgSrc0, msgSrc1, exDesc, msgDesc, execSize, sfid, numSrc0, numSrc1, 0, isEOT, isSendc, mask);
 #endif
 }
 
root@chengxi_gpu_sim_dev_ww19:~/working_dir/libraries.gpu.xetla# git add include/core/xetla_core_raw_send.hpp
root@chengxi_gpu_sim_dev_ww19:~/working_dir/libraries.gpu.xetla# git commit -m "try clang hook"
[pre-commit hook] Run clang-format on  include/core/xetla_core_raw_send.hpp
[wcx/add_clang_format ddb3c9e] try clang hook
root@chengxi_gpu_sim_dev_ww19:~/working_dir/libraries.gpu.xetla# git diff HEAD
```


## Known issues
### Unstable llvm in DPCPP leads to CI formating check failed
Clang-format uses llvm compiler frotend to analyze cpp syntax. After you exporting envs like `CC` and `CXX`, llvm in DPCPP will be in use. Because llvm in DPCPP is not stable, clang-formatted files in local docker container might be different files in CI. 

W/A: 
1. starting a new command line session before using clang-format; or 
2. using git GUI to commit changes if you’ve set up git commit hook.
