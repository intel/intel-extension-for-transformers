/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

enum class profiling_selector : uint8_t { CPU = 0, GPU = 1, ALL = 2 };

struct profiling_statistics {
    double first_time = 0.0;
    double min = 0.0;
    double max = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double variance = 0.0;
};

using std::string;
using std::vector;

class profiling_helper {
    //Statistics on CPU and GPU
    profiling_statistics gpu_statistics;
    profiling_statistics cpu_statistics;

    //Array for recording CPU/GPU time and GPU events
    vector<vector<sycl::event>> gpu_event_vec;
    vector<vector<double>> gpu_time_vec;
    vector<vector<double>> cpu_time_vec;

    //Some information for measuring performance
    vector<long> work_amount;
    vector<string> work_name;
    vector<string> kernel_name;

    //Used when profiling multiple kernels, defaults to 1
    int kernel_nums;

    std::chrono::time_point<std::chrono::high_resolution_clock> cpu_start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> cpu_end_time;

    void get_gpu_time_from_events(int kernel_id) {
        for (const auto gpu_event : gpu_event_vec[kernel_id]) {
            auto gpu_start = gpu_event.get_profiling_info<
                    sycl::info::event_profiling::command_start>();
            auto gpu_end = gpu_event.get_profiling_info<
                    sycl::info::event_profiling::command_end>();
            double gpu_time = (gpu_end - gpu_start) / 1000000.0;
            gpu_time_vec[kernel_id].push_back(gpu_time);
        }
    }

    void get_statistics(vector<double> &time, profiling_statistics &stat) {
        //first execution time
        stat.first_time = time[0];

        int iter = time.size();
        std::sort(time.begin(), time.end());

        //minimum time
        stat.min = time[0];
        //maximum time(include first execution time)
        stat.max = time[iter - 1];
        //median time
        int mid = int(iter / 2);
        stat.median
                = iter % 2 == 0 ? (time[mid] + time[mid + 1]) * 0.5 : time[mid];

        //mean time(exclude first execution time)
        double total = 0.0;
        for (int i = 1; i < iter; i++) {
            total += time[i];
        }
        stat.mean = (iter == 1 ? time[0] : (total / (iter - 1)));

        //time mean square error
        for (int i = 1; i < iter; i++) {
            stat.variance += pow(time[i] - stat.mean, 2);
        }
        stat.variance /= iter;
    }

    void print_info(string &label, string &info, double &value, string &unit) {
        std::cout << label << info << value << unit << std::endl;
    }

    // string label = "[kernel time]";
    // if (device == "CPU") { label = "[Wall time]"; }
    void print_statistics(int kernel_id, profiling_statistics &stat,
            string label = "[kernel time]", string device = "GPU") {
        vector<double> value = {stat.first_time, stat.min, stat.max,
                stat.median, stat.mean, stat.variance};
        vector<string> desc = {"first   ", "minimum ", "maximum ", "median  ",
                "mean(exclude the first trial) ",
                "variance(exclude the first trial) "};
        string unit = "ms";

        for (int i = 0; i < value.size(); i++) {
            string info = "The " + desc[i] + "running(" + device
                    + "_time) time is ";
            print_info(label, info, value[i], unit);
        }

        string time_string = std::to_string(stat.min) + " "
                + std::to_string(stat.max) + " " + std::to_string(stat.median)
                + " " + std::to_string(stat.mean);
        ::testing::Test::RecordProperty(
                "kernel_time:ms:" + this->kernel_name[kernel_id], time_string);
    }

    void print_performance(int kernel_id, profiling_statistics &stat,
            int scaling_ratio, string label = "[kernel time]",
            string device = "GPU") {
        vector<double> value = {stat.max, stat.min, stat.median, stat.mean};
        vector<string> desc = {"minimum ", "maximum ", "median  ", "mean    "};
        string unit = "";
        string perf_string = "";
        for (int i = 0; i < value.size(); i++) {
            string info = "The " + desc[i] + work_name[kernel_id] + "(" + device
                    + "_time) is ";
            double perf = ((double)work_amount[kernel_id] / scaling_ratio)
                    / value[i];
            print_info(label, info, perf, unit);

            perf_string = perf_string + std::to_string(perf) + " ";
        }

        ::testing::Test::RecordProperty("kernel_time:" + work_name[kernel_id]
                        + ":" + this->kernel_name[kernel_id],
                perf_string);
    }

    void print_profiling_data(int kernel_id, vector<double> &time,
            profiling_statistics stat, string label = "[kernel time]",
            string device = "GPU") {
        get_statistics(time, stat);
        std::cout << "============= Profiling for " << label << " "
                  << "=============" << std::endl;
        print_statistics(kernel_id, stat, label, device);
        std::cout << "======================================================"
                  << std::endl;
        if (this->work_amount[kernel_id] != 0) {
            std::cout << "============== " << label << " "
                      << work_name[kernel_id]
                      << "   ================== " << std::endl;

            //Different performance data correspond to different scaling ratios
            if (this->work_name[kernel_id] == "gflops") {
                print_performance(kernel_id, stat, 1000000, label, device);
            } else if (this->work_name[kernel_id] == "mhashs") {
                print_performance(kernel_id, stat, 1000, label, device);
            } else if (this->work_name[kernel_id] == "GB/s") {
                print_performance(kernel_id, stat, 1000000, label, device);
            } else {
                std::cout << "Not sure how much workload scales" << std::endl;
            }

            std::cout
                    << "======================================================"
                    << std::endl;
        }
        std::cout << std::endl;
    }

    void set_time_vecs() {
        gpu_event_vec.resize(this->kernel_nums);
        gpu_time_vec.resize(this->kernel_nums);
        cpu_time_vec.resize(this->kernel_nums);
        assert((this->kernel_nums >= this->kernel_name.size())
                && "kernel num is less than actual");
    }

    void write_performance_metrics_into_report() {
        //Output performance data to json file
        ::testing::Test::RecordProperty(
                ":kernel_time:ms", "minimum:maximum:median:mean");
        vector<string> metrics = {"gflops", "mhashs", "GB/s"};
        vector<bool> is_printed(metrics.size(), 0);
        for (int kernel_id = 0; kernel_id < this->kernel_nums; kernel_id++) {
            for (int i = 0; i < metrics.size(); i++) {
                if (!is_printed[i]
                        && (this->work_name[kernel_id] == metrics[i])) {
                    ::testing::Test::RecordProperty(
                            ":kernel_time:" + metrics[i],
                            "minimum:maximum:median:mean");
                    is_printed[i] = 1;
                }
            }
        }
    }

public:
    profiling_helper(string kernel_name, long work_amount,
            string work_name = "gflops", int kernel_nums = 1) {
        for (int i = 0; i < kernel_nums; i++) {
            this->kernel_name.push_back(kernel_name);
            this->work_name.push_back(work_name);
            this->work_amount.push_back(work_amount);
        }
        this->kernel_nums = kernel_nums;
        set_time_vecs();
    }

    profiling_helper(vector<string> kernel_name, vector<long> work_amount,
            vector<string> work_name, int kernel_nums = 1) {
        this->kernel_name = kernel_name;
        this->work_name = work_name;
        this->work_amount = work_amount;
        this->kernel_nums = kernel_nums;
        set_time_vecs();
    }

    profiling_helper() {
        this->kernel_name.emplace_back("");
        this->work_name.emplace_back("");
        this->work_amount.push_back(0);
        this->kernel_nums = 1;
        set_time_vecs();
    }

    void cpu_start() {
        cpu_start_time = std::chrono::high_resolution_clock::now();
    }

    void cpu_end(int kernel_id = 0) {
        using namespace std::chrono;
        cpu_end_time = high_resolution_clock::now();
        double cpu_time = (double)duration_cast<nanoseconds>(
                                  cpu_end_time - cpu_start_time)
                                  .count()
                / 1000000;
        cpu_time_vec[kernel_id].push_back(cpu_time);
    }

    void add_gpu_event(sycl::event gpu_event, int kernel_id = 0) {
        gpu_event_vec[kernel_id].push_back(gpu_event);
    }

    void print_profiling_result(profiling_selector selector) {
        write_performance_metrics_into_report();
        for (int i = 0; i < kernel_nums; i++) {
            std::cout << "\n***************** PROFILING FOR KERNEL" << i
                      << " ***********************" << std::endl;
            if (selector != profiling_selector::CPU) {
                get_gpu_time_from_events(i);
                print_profiling_data(i, gpu_time_vec[i], gpu_statistics,
                        "[kernel time]", "GPU");
            }
            if (selector != profiling_selector::GPU) {
                print_profiling_data(i, cpu_time_vec[i], cpu_statistics,
                        "[Wall time]", "CPU");
            }
        }
    }
};
