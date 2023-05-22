"""
This script is for benchmarking the inference
"""
from argparse import ArgumentParser, REMAINDER
from argparse import RawTextHelpFormatter
import signal
import os
import platform
import subprocess
import sys
import re
import csv
from pathlib import Path
import numpy as np

class CPUinfo():
    '''
    Get CPU inforamation, such as cores list and NUMA information.
    '''
    def __init__(self):
        self.cores = 0
        self.sockets = 0
        self.cpuinfo = []
        if platform.system() == "Windows":
            raise RuntimeError("Windows platform is not supported!!!")
        elif platform.system() == "Linux":
            args = ["lscpu"]
            lscpu_info = subprocess.check_output(args, universal_newlines=True).split("\n")
            for line in lscpu_info:
                int_re = re.compile(r'\d+')
                if line.find("Core(s) per socket:") >= 0:
                    core_per_socket_int = [int(i) for i in int_re.findall(line)]
                    self.cores = core_per_socket_int[0]
                elif line.find("Socket(s):") >= 0:
                    socket_int = [int(i) for i in int_re.findall(line)]
                    self.sockets = socket_int[0]

    def get_cores_per_socket(self):
        '''
        Get get_cores_per_socket
        '''
        return self.cores

    def get_sockets(self):
        '''
        Get sockets
        '''
        return self.sockets

class Configs:
    """
    record the configs like batch, cores...
    """
    def __init__(self, batch, instance, cores, weight_sharing, memory_allocator,
                 memory_planning, cmds, mode):
        self.batch = batch
        self.instance = instance
        self.cores = cores
        self.weight_sharing = weight_sharing
        self.memory_allocator = memory_allocator
        self.memory_planning = memory_planning
        self.cmds = cmds
        self.mode = mode

    def set_batch(self, batch):
        """assign batch size."""
        self.batch = batch

    def set_instance(self, instance):
        """assign instance."""
        self.instance = instance

    def set_cores_per_instance(self, cores):
        """assign cores_per_instance."""
        self.cores = cores

    def set_weight_sharing(self, weight_sharing):
        """assign weight sharing."""
        self.weight_sharing = weight_sharing

    def set_memory_allocator(self, memory_allocator):
        """assign memory allocator."""
        self.memory_allocator = memory_allocator

    def set_memory_planning(self, memory_planning):
        """assign memory planning."""
        self.memory_planning = memory_planning

    def set_cmds(self, cmds):
        """assign cmds."""
        self.cmds = cmds

    def set_mode(self, mode):
        """assign script mode."""
        self.mode = mode

    def get_batch(self):
        """assign weight sharing."""
        return self.batch

    def get_instance(self):
        """get instance."""
        return self.instance

    def get_cores_per_instance(self):
        """get cores_per_instance."""
        return self.cores

    def get_weight_sharing(self):
        """get weight sharing."""
        return self.weight_sharing

    def get_memory_allocator(self):
        """get memory allocator."""
        return self.memory_allocator

    def get_memory_planning(self):
        """get memory planning."""
        return self.memory_planning

    def get_cmds(self):
        """get cmds."""
        return self.cmds

    def get_mode(self):
        """get script mode."""
        return self.mode

def get_tmp_log_path(config, input_path, instanc_index):
    """get path of tmp log."""
    return "{}/{}_{}_{}_{}_{}_{}_{}.log".format(input_path, config.get_batch(), \
           config.get_instance(), config.get_cores_per_instance(), instanc_index, \
           config.get_weight_sharing(), config.get_memory_allocator(), config.get_memory_planning())

def get_formal_log_path(name, input_path):
    """get path of output."""
    return "{}/{}".format(input_path, name)

def get_min_latency_output_log_path(config, input_path):
    """get path of min_latency."""
    return "{}/min_latency_batch_{}_instance_{}_cores_{}_{}_{}_{}.log".format(input_path, \
           config.get_batch(), config.get_instance(), config.get_cores_per_instance(), \
           config.get_weight_sharing(), config.get_memory_allocator(), config.get_memory_planning())

def get_output_log_path(args, config, input_path):
    """get path of other modes."""
    return "{}/{}_batch_{}_instance_{}_cores_{}_{}_{}_{}.log".format(input_path, args.mode, \
           config.get_batch(), config.get_instance(), config.get_cores_per_instance(), \
           config.get_weight_sharing(), config.get_memory_allocator(), config.get_memory_planning())

def latency_mode_grab_log(input_path, output, config, is_best_write, first_write, latency_constraint):
    """
    grep logs
    """
    all_latency = []
    batch_size = config.get_batch()
    avg_latency = float(0)
    p50_latency = float(0)
    p90_latency = float(0)
    p99_latency = float(0)
    throughput = float(0)
    throughput_str = ""
    avg_latency_str = ""
    i = 0
    while i < config.get_instance():
        log_path = get_tmp_log_path(config, input_path, i)
        latency_path = get_tmp_log_path(config, input_path + "/all_latency", i)
        latency_path = latency_path.replace('.log', '.npy')
        instance_all_latency = np.load(latency_path)
        all_latency.append(instance_all_latency)
        i += 1
        
        try:
            with open(log_path, 'r', errors='ignore') as src_fp:
                for line in src_fp.readlines():
                    if line.find("Throughput:") >= 0:
                        throughput_str = line
                    elif line.find("Average Latency:") >= 0:
                        avg_latency_str = line

            float_re = re.compile(r'\d+\.\d+')
            floats_throughput = [float(i) for i in float_re.findall(throughput_str)]
            floats_latency = [float(i) for i in float_re.findall(avg_latency_str)]

            throughput += floats_throughput[0]
            avg_latency += floats_latency[0]
        except OSError as ex:
            print(ex)
            src_fp.close()
        finally:
            src_fp.close()
    avg_latency = avg_latency / config.instance
    all_latency = np.array(all_latency)
    p50_latency = (np.percentile(all_latency, 50) / batch_size) * 1000
    p90_latency = (np.percentile(all_latency, 90) / batch_size) * 1000
    p99_latency = (np.percentile(all_latency, 99) / batch_size) * 1000

    write_mode = 'a'

    write_data = False
    if latency_constraint != 0 and latency_constraint > avg_latency:
        write_data = True
    if latency_constraint == 0:
        write_data = True

    if write_data:
        try:
            with open(output, write_mode) as dst_fp:
                if '.csv' in output:
                    fields = ['batch', 'instance', 'cores per instance', 'Troughput',
                              'Average Latency', 'P50 Latency', 'P90 Latency', 'P99 Latency', 'cmd']

                    fields_blank = ['', '', '', '', '', '', '', '', '']
                    fields_best = ['best', '', '', '', '', '', '', '', '']
                                 
                    csvwriter = csv.writer(dst_fp)
                    if first_write == True:
                        csvwriter.writerow(fields)
                    if is_best_write == True:
                        csvwriter.writerow(fields_blank)
                        csvwriter.writerow(fields_blank)
                        csvwriter.writerow(fields_best)
                        csvwriter.writerow(fields)
                    row = [[config.get_batch(), config.get_instance(),
                            config.get_cores_per_instance(), throughput, avg_latency,
                            p50_latency, p90_latency, p99_latency, config.get_cmds()]]
                    csvwriter.writerows(row)
                else:
                    dst_fp.write("cmd: {}\n".format(config.get_cmds()))
                    dst_fp.write("**************************************\n")
                    dst_fp.write("batch: {}\n".format(config.get_batch()))
                    dst_fp.write("instance: {}, cores per instance: {}\n".
                                 format(config.get_instance(), config.get_cores_per_instance()))
                    dst_fp.write("--------------------------------------\n")
                    dst_fp.write("Troughput: {} images/sec\n".format(throughput))
                    dst_fp.write("Average Latency: {} ms\n".format(avg_latency))
                    dst_fp.write("P50 Latency: {} ms\n".format(p50_latency))
                    dst_fp.write("P90 Latency: {} ms\n".format(p90_latency))
                    dst_fp.write("P99 Latency: {} ms\n".format(p99_latency))
                    dst_fp.write("--------------------------------------\n")
        except OSError as ex:
            print(ex)
            dst_fp.close()
        finally:
            dst_fp.close()
            if config.get_mode() == "default_throughput" or config.get_mode() == "max_throughput":
                return throughput
            return avg_latency
    else:
        if config.get_mode() == "default_throughput" or config.get_mode() == "max_throughput":
            return throughput
        return avg_latency


def replace_instance_num(cmd_str, instance):
    """set instance numbers"""
    return cmd_str.replace("INST_NUM=", "INST_NUM="+str(instance))

def get_cmd_prefix(core_list):
    """get memory prefix of cmd"""
    return 'OMP_NUM_THREADS={} numactl --localalloc --physcpubind={} '. \
            format(len(core_list), ','.join(core_list.astype(str)))

def get_cmd_prefix2(core_list):
    """get memory prefix of cmd"""
    return 'numactl --localalloc --physcpubind={}'. \
            format(','.join(core_list.astype(str)))

def get_weight_sharing(cmd_str):
    """get weight sharing """
    if cmd_str.find("WEIGHT_SHARING=") >= 0:
        return "enabled"
    return "disabled"

def get_memory_planning(cmd_str):
    """get memory planning."""
    if cmd_str.find("UNIFIED_BUFFER=") >= 0:
        return "unified_buffer"
    return "cycle_buffer"

def get_memory_allocator(cmd_str):
    """get memory allocator."""
    if cmd_str.find("jemalloc") >= 0:
        return "jemalloc"
    return "default"

def add_weight_sharing_flag(prefix_list, weight_sharing):
    """set weight sharing flag."""
    if weight_sharing == "enabled":
        prefix_list[:] = [prefix+ " WEIGHT_SHARING=1 " for prefix in prefix_list]
    elif weight_sharing == "disabled":
        print("weight sharing disabled")
    elif weight_sharing == "auto":
        prefix_list[:] = [prefix+ " WEIGHT_SHARING=1 " for prefix in prefix_list]
    else:
        print("weight sharing incorrect")
    return prefix_list

def add_memory_planning_flag(prefix_list, memory_planning):
    """set memory planning."""
    if memory_planning == "cycle_buffer":
        print("cycle buffer")
    elif memory_planning == "unified_buffer":
        prefix_list[:] = [prefix+ " UNIFIED_BUFFER=1 " for prefix in prefix_list]
        print("unified_buffer")
    elif memory_planning == "auto":
        prefix_list += [prefix+ " UNIFIED_BUFFER=1 " for prefix in prefix_list]
        print("auto memory planning")
    else:
        print("memory incorrect incorrect")
    return prefix_list

def add_instance_num_flag(prefix_list):
    """set instance num."""
    prefix_list[:] = [prefix+ " INST_NUM= " for prefix in prefix_list]
    return prefix_list

def get_memory_settings(path, args):
    """append memory setting."""
    memory_prefix_list = []
    jemalloc_prefix = "LD_PRELOAD={}/intel_extension_for_transformers/backends/neural_engine/"\
                      "third_party/jemalloc/lib/libjemalloc.so:$LD_PRELOAD ".format(path)
    if args.memory_allocator == "jemalloc":
        memory_prefix_list.append(jemalloc_prefix)

    elif args.memory_allocator == "default":
        memory_prefix_list.append("")
    elif args.memory_allocator == "auto":
        memory_prefix_list.append(jemalloc_prefix)
        memory_prefix_list.append("")
    else:
        print("please enter correct setting")

    tmp_list = add_weight_sharing_flag(memory_prefix_list, args.weight_sharing)
    tmp_list = add_memory_planning_flag(tmp_list, args.memory_planning)
    tmp_list = add_instance_num_flag(tmp_list)

    return tmp_list

def set_cmd_prefix(cmd, core_list):
    """set memory prefix of cmd"""
    cmd.append('numactl')
    cmd.append('--localalloc')
    cmd.append('--physcpubind={}'.format(','.join(core_list.astype(str))))

def set_numactl_env(env_cmd, core_list):
    """set numactl env"""
    env_cmd["PATH"] = "/usr/bin:" + env_cmd["PATH"]
    env_cmd["OMP_NUM_THREADS"] = str(len(core_list))

def set_jemalloc_env(env_cmd, memory_allocator, path):
    """set jemalloc env"""
    if memory_allocator == "jemalloc":
        env_cmd["LD_PRELOAD"] = "{}/intel_extension_for_transformers/backends/neural_engine/executor/" \
                                "third_party/jemalloc/lib/libjemalloc.so:$".format(path) \
                                + env_cmd["LD_PRELOAD"]        

def set_unified_buffer_env(env_cmd, memory_planning):
    """set unified buffer env"""
    if memory_planning == "unified_buffer":
        env_cmd["UNIFIED_BUFFER"] = str(1)

def set_weight_sharing(env_cmd, weight_sharing):
    """set weight sharing env"""
    if weight_sharing == "enabled":
        env_cmd["WEIGHT_SHARING"] = str(1)

def set_instance_num(env_cmd, instance):
    """set instance num"""
    env_cmd["INST_NUM"] = str(instance)


def replace_batch(cmd, args, batch):
    """set batch if your script input had batch."""
    batch_return = batch
    batch_str = '--batch_size=1'.replace('1', str(batch))
    for arg_str in args.program_args:
        if "batch" not in arg_str:
            cmd.append(arg_str)
        else:
            int_re = re.compile(r'\d')
            tmp = [int(i) for i in int_re.findall(arg_str)]
            if batch_return == 0:
                batch_return = tmp[0]
                batch_str = arg_str
                
    cmd.append(batch_str)
    return batch_return

class Launcher():
    r"""
     Base class for launcher
    """
    cores_per_socket = CPUinfo().get_cores_per_socket()
    sockets = CPUinfo().get_sockets()
    current_path = os.path.abspath(os.getcwd())
    launcher_env = os.environ.copy()
    project_path = Path(os.path.abspath(os.getcwd())).parent.parent.parent.parent.absolute()

    def __init__(self):
        print("Launcher init")

    def launch(self, args):
        """
        launch
        """
        pass

class OneInstanceLauncher(Launcher):
    r"""
     Launcher for latency
    """
    def launch(self, args, memory_prefix_list):
        processes = []
        cmd = []
        cmd_for_print = []
        processes = []
        tmp_log_path = ""
        cores = 1

        current_path = os.path.abspath(os.getcwd())
        batch_size_list = []
        if args.batch_size == "auto":
            batch_size_list = "1,2,4,8,16,32,64,128".split(',')
        else:
            batch_size_list = args.batch_size.split(',')

        first_write = True
        instance = 1
        if args.instance_num == "1":
            cores = self.cores_per_socket
            core_list = np.arange(0, cores)
            min_latency = 1000000000

            min_config = Configs(1, instance, cores, "disabled", "default",
                                 "cycle_buffer", "", args.mode)

            for batch_str in batch_size_list:
                batch_size = int(batch_str)
                for mp_list_idx, mp_list_item in enumerate(memory_prefix_list):
                    cmd_prefix = get_cmd_prefix(core_list)

                    cmd_prefix = replace_instance_num(
                        memory_prefix_list[mp_list_idx], instance) + cmd_prefix
                    cmd.clear()
                    cmd_for_print.clear()
                    set_cmd_prefix(cmd, core_list)


                    cmd_for_print.append(cmd_prefix)
                    cmd.append(sys.executable)
                    cmd.append("-u")
                    cmd_for_print.append(sys.executable)
                    cmd_for_print.append("-u")

                    weight_sharing = get_weight_sharing(cmd_prefix)
                    memory_allocator = get_memory_allocator(cmd_prefix)
                    memory_planning = get_memory_planning(cmd_prefix)


                    import shlex
                    cmd.append(shlex.quote(args.program))
                    cmd_for_print.append(args.program) 
                    batch_size = replace_batch(cmd, args, batch_size)
                    replace_batch(cmd_for_print, args, batch_size)

                    tmp_config = Configs(batch_size, instance, cores, weight_sharing,
                                         memory_allocator, memory_planning,
                                         memory_prefix_list[mp_list_idx], args.mode)
                    tmp_log_path = get_tmp_log_path(tmp_config, current_path, 0)
                    if os.path.exists(current_path + "/all_latency") == 0 :
                        os.mkdir(current_path + "/all_latency")

                    cmd.append("--log_file="+tmp_log_path)
                    cmd_for_print.append("--log_file="+tmp_log_path)

                    cmd_s = " ".join(cmd_for_print)
                    tmp_config.set_cmds(cmd_s)

                    env_cmd = self.launcher_env
                    set_numactl_env(env_cmd, core_list)
                    set_jemalloc_env(env_cmd, memory_allocator, self.project_path)
                    set_unified_buffer_env(env_cmd, memory_planning)
                    set_weight_sharing(env_cmd, weight_sharing)
                    set_instance_num(env_cmd, instance)
                    #print("env is "+str(env_cmd))

                    process = subprocess.Popen(cmd, env=env_cmd, shell=False, 
                                                stdout=subprocess.PIPE)
                    processes.append(process)
                    for process in processes:
                        process.wait()
                        if process.returncode != 0:
                            raise subprocess.CalledProcessError(
                                returncode=process.returncode, cmd=cmd_s)

                    output_log_name = get_formal_log_path(args.output_file , current_path)
                    tmp_latency = latency_mode_grab_log(current_path, output_log_name,
                                                        tmp_config, False, first_write, 0)
                    first_write = False
                    if tmp_latency < min_latency:
                        min_latency = tmp_latency
                        min_config.set_batch(batch_size)
                        min_config.set_instance(instance)
                        min_config.set_cores_per_instance(cores)
                        min_config.set_weight_sharing(weight_sharing)
                        min_config.set_memory_allocator(memory_allocator)
                        min_config.set_memory_planning(memory_planning)
                        min_config.set_cmds(cmd_s)


                latency_mode_grab_log(current_path, output_log_name, min_config, True, first_write, 0)

        elif args.instance_num == "auto":
            cores = self.cores_per_socket
            min_latency = 1000000000
            min_config = Configs(1, instance, cores,
                                 "disabled", "default", "cycle_buffer", "", args.mode)

            for batch_str in batch_size_list:
                batch_size = int(batch_str)
                cores_iterator = int(cores/2)
                while cores_iterator <= cores:
                    core_list = np.arange(0, cores_iterator)
                    cmd_prefix = get_cmd_prefix(core_list)

                    for mp_list_idx, mp_list_item in enumerate(memory_prefix_list):
                        cmd_prefix = replace_instance_num(memory_prefix_list[mp_list_idx],
                                                                   instance) + cmd_prefix
                        cmd.clear()
                        cmd_for_print.clear()
                        set_cmd_prefix(cmd, core_list)

                        cmd_for_print.append(cmd_prefix)
                        cmd.append(sys.executable)
                        cmd.append("-u")
                        cmd_for_print.append(sys.executable)
                        cmd_for_print.append("-u")

                        weight_sharing = get_weight_sharing(cmd_prefix)
                        memory_allocator = get_memory_allocator(cmd_prefix)
                        memory_planning = get_memory_planning(cmd_prefix)


                        import shlex
                        cmd.append(shlex.quote(args.program))
                        cmd_for_print.append(args.program)
                        batch_size = replace_batch(cmd, args, batch_size)
                        replace_batch(cmd_for_print, args, batch_size)
                        tmp_config = Configs(batch_size, instance, cores_iterator, weight_sharing,
                                             memory_allocator, memory_planning,
                                             memory_prefix_list[mp_list_idx], args.mode)
                        tmp_log_path = get_tmp_log_path(tmp_config, current_path, 0)
                        if os.path.exists(current_path + "/all_latency") == 0 :
                            os.mkdir(current_path + "/all_latency")
                        log_file_path = '--log_file={}'.format(tmp_log_path)
                        cmd.append (log_file_path)
                        cmd_for_print.append("--log_file="+tmp_log_path)
                        cmd.append("--log_file="+tmp_log_path)

                        cmd_s = " ".join(cmd)
                        tmp_config.set_cmds(cmd_s)

                        env_cmd = self.launcher_env
                        set_numactl_env(env_cmd, core_list)
                        set_jemalloc_env(env_cmd, memory_allocator, self.project_path)
                        set_unified_buffer_env(env_cmd, memory_planning)
                        set_weight_sharing(env_cmd, weight_sharing)
                        set_instance_num(env_cmd, instance)

                        process = subprocess.Popen(cmd, env=env_cmd, shell=False,
                                                    stdout=subprocess.PIPE)
                        processes.append(process)
                        for process in processes:
                            process.wait()
                            if process.returncode != 0:
                                raise subprocess.CalledProcessError(
                                    returncode=process.returncode, cmd=cmd)

                        output_log_name = get_formal_log_path(args.output_file , current_path)
                        latency_tmp = latency_mode_grab_log(current_path, output_log_name,
                                                            tmp_config, False, first_write, 0)

                        first_write = False
                        if latency_tmp < min_latency:
                            min_latency = latency_tmp
                            min_config.set_batch(batch_size)
                            min_config.set_instance(instance)
                            min_config.set_cores_per_instance(cores_iterator)
                            min_config.set_weight_sharing(weight_sharing)
                            min_config.set_memory_allocator(memory_allocator)
                            min_config.set_memory_planning(memory_planning)
                            min_config.set_cmds(tmp_config.get_cmds())
                    cores_iterator += 1

            output_log_name = get_min_latency_output_log_path(min_config, current_path)
            if args.output_file != "":
                output_log_name = args.output_file
            latency_mode_grab_log(current_path, output_log_name, min_config, True, first_write, 0)
        else:
            print("Latency mode only support instance=auto or instance=1 !!!")

class MultiInstanceLauncher(Launcher):
    r"""
     Launcher for Throughput
    """
    def launch(self, args, memory_prefix_list):
        current_path = os.path.abspath(os.getcwd())
        instance_num = []
        if args.mode == "default_throughput":
            instance_num.append(self.sockets)
        elif args.mode == "default_latency":
            instance_num.append(int(self.cores_per_socket * self.sockets/4))
        else:
            if args.instance_num != "auto":
                instance_num = args.instance_num.split(',')
            else:
                max_instance = int(self.cores_per_socket/2)
                instance_num.append(max_instance)
                while max_instance%2 == 0:
                    max_instance = max_instance/2
                    instance_num.append(max_instance)

        batch_size_list = []
        if args.batch_size == "auto":
            batch_size_list = "1,2,4,8,16,32,64,128".split(',')
        else:
            batch_size_list = args.batch_size.split(',')

        first_write = True
        max_throughput = 0
        max_config = Configs(1, self.sockets, int(self.cores_per_socket),
                             "disabled", "default", "cycle_buffer", "", args.mode)
        for batch_str in batch_size_list:
            batch_size = int(batch_str)
            for instance_str in instance_num:
                instance = int(instance_str)
                cores = int(self.cores_per_socket*self.sockets/instance)
                cmd = []

                for mp_list_idx, mp_list_item in enumerate(memory_prefix_list):
                    cmd_s = " "
                    for i in range(instance):
                        cmd.clear()
                        core_list = np.arange(0, cores) + i*cores
                        cmd_prefix = get_cmd_prefix(core_list)
                        cmd_prefix = replace_instance_num(
                            memory_prefix_list[mp_list_idx], instance) + cmd_prefix

                        weight_sharing = get_weight_sharing(cmd_prefix)
                        memory_allocator = get_memory_allocator(cmd_prefix)
                        memory_planning = get_memory_planning(cmd_prefix)
                        cmd.append(cmd_prefix)
                        cmd.append(sys.executable)
                        cmd.append("-u")

                        cmd.append(args.program)
                        batch_size = replace_batch(cmd, args, batch_size)
                        tmp_config = Configs(batch_size, instance, cores, weight_sharing,
                                             memory_allocator, memory_planning,
                                             memory_prefix_list[mp_list_idx], args.mode)
                        tmp_log_path = get_tmp_log_path(tmp_config, current_path, i)
                        if os.path.exists(current_path + "/all_latency") == 0 :
                            os.mkdir(current_path + "/all_latency")
                        log_file_path = '--log_file={}'.format(tmp_log_path)
                        cmd.append (log_file_path)
                        cmd_tmp = " ".join(cmd)
                        cmd_postfix = ' 2>&1|tee {} & \\\n'.format(tmp_log_path)
                        cmd_tmp += cmd_postfix
                        cmd_s += cmd_tmp

                    cmd_s += " wait "
                    tmp_config.set_cmds(cmd_s)

                    try:
                        f = open("tmp.sh", 'w')
                    except OSError as e:
                        print(e)
                        print("shell store failed!!!!")
                        sys.exit()
                        
                    with f:
                        f.write(cmd_s)
                   
                    #ipex use this command, but bandit reject the cmd
                    #the performance of two cmds may be different
                    #p_cmd = subprocess.Popen(cmd_s, preexec_fn=os.setsid, shell=True)
                    p_cmd = subprocess.Popen(["sh", "tmp.sh"])

                    try:
                        p_cmd.communicate()
                    except KeyboardInterrupt:
                        os.killpg(os.getpgid(p_cmd.pid), signal.SIGKILL)
                    output_log_name = get_formal_log_path(args.output_file , current_path)

                    #output_log_name = get_output_log_path(args, max_config, current_path)

                    #if args.output_file != "":
                    #    output_log_name = args.output_file
                    tmp_throughput = latency_mode_grab_log(current_path, output_log_name,
                                                           tmp_config, False, first_write,
                                                           args.latency_constraint)

                    first_write = False
                    if tmp_throughput > max_throughput:
                        max_throughput = tmp_throughput
                        max_config.set_batch(batch_size)
                        max_config.set_instance(instance)
                        max_config.set_cores_per_instance(cores)
                        max_config.set_weight_sharing(weight_sharing)
                        max_config.set_memory_allocator(memory_allocator)
                        max_config.set_memory_planning(memory_planning)
                        max_config.set_cmds(cmd_s)

        latency_mode_grab_log(current_path, output_log_name, max_config,
                              True, first_write, args.latency_constraint)


def parse_args():
    """
    arguments
    """
    parser = ArgumentParser(description="This is a script for inference on Intel Xeon CPU",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument("-m", "--module", default=False, action="store_true",
                        help="Changes each process to interpret the launch script "
                             "as a python module, executing with the same behavior as"
                             "'python -m'.")

    parser.add_argument("--mode", metavar='\b', default="default_throughput", type=str,
                        help="latency or throughput")

    parser.add_argument("--instance_num", metavar='\b', default="1", type=str,
                        help="1,2,4 or auto")

    parser.add_argument("--batch_size", metavar='\b', default="0", type=str,
                        help="1,32 or auto")

#    parser.add_argument("--dump_all", action='store_true', help="dump flag")

    parser.add_argument("--latency_constraint", metavar='\b', type=float, default=0.0,
                        help="all results not exceeded max latency in throughput mode")

    parser.add_argument("--memory_allocator", metavar='\b', default="default", type=str,
                        help="memory alloc")

    parser.add_argument("--weight_sharing", metavar='\b', default="disabled", type=str,
                        help="memory alloc")

    parser.add_argument("--memory_planning", metavar='\b', default="cycle_buffer", type=str,
                        help="memory planning")

    parser.add_argument("--output_file", metavar='\b', default="out.csv", type=str,
                        help="output file name")

    parser.add_argument("program", type=str,
                        help="The full path to the proram/script to be launched. "
                             "followed by all the arguments for the script")
    parser.add_argument('program_args', nargs=REMAINDER)
    return parser.parse_args()

def main():
    """
    get path and memory info
    """
    print("------")
    args = parse_args()

    script_path = Path(os.path.abspath(os.getcwd()))
    project_path = script_path.parent.parent.parent.parent.absolute()

    dump_log_path = "{}/{}".format(script_path, args.output_file)
    if os.path.exists(dump_log_path):
        os.remove(dump_log_path)

    memory_prefix_list = get_memory_settings(project_path, args)

    launcher = None
    if args.mode == "min_latency":
        launcher = OneInstanceLauncher()
    elif args.mode == "default_latency":
        launcher = MultiInstanceLauncher()
    elif args.mode == "max_throughput": #throughput
        launcher = MultiInstanceLauncher()
    elif args.mode == "default_throughput":
        launcher = MultiInstanceLauncher()

    launcher.launch(args, memory_prefix_list)

if __name__ == "__main__":
    main()
