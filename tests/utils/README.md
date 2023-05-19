## Profiling  
1.Define a profiling_helper object  
- Analyze execution time only  
  *profiling_helper prof;*  
- Analyze performance data such as gflops  
  *long work_amount = 2 * matrix_m * matrix_n * matrix_k;*  
  *profiling_helper prof(work_amount, "gflops");*  
- Multiple kernels need to be analyzed  
  *profiling_helper(string kernel_name, long work_amount, string work_name, int kernel_nums = 1)*  
  Subsequent prof.cpu_end(), add_gpu_event need to pass in the corresponding kernel_id as a parameter  
- Multiple kernels need to be analyzed and corresponding to different work_amount and work_name 
  *profiling_helper(vector<string> kernel_name, vector<long> work_amount, vector<string> work_name, int kernel_nums = 1)*  
  Items without work_amount and work_name can be written with 0 and "" respectively  
  
2.Record CPU time  
- Use cpu_start() and cpu_end() at the beginning and end of the kernel function respectively。  
  *prof.cpu_start();*  
  *kernel();*  
  *prof.cpu_end();*  
  
3.Record GPU time  
- Use add_gpu_event() to log kernel events  
  *prof.add_gpu_event(e_esimd);*  
  
4.Print result of profiling  
- Printing CPU/GPU/ALL corresponds to profiling_selector::CPU/GPU/ALL respectively。  
  *prof.print_profiling_result(profiling_selector::ALL);*  
