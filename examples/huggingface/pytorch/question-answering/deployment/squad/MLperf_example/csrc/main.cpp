//  Copyright (c) 2021 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#include <loadgen.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <thread>   // NOLINT
#include "executor.hpp"

#include "cxxopts.hpp"
#include "sut.hpp"
#include "../inference/loadgen/test_settings.h"
#include "bert_qsl.hpp"

int main(int argc, char** argv) {
  executor::GlobalInit(argv[0]);

  cxxopts::Options opts (
    "bert_inference", "MLPerf Benchmark, BERT Inference");
  opts.allow_unrecognised_options();
  opts.add_options()
    ("m,model_conf", "Torch Model Conf",
      cxxopts::value<std::string>())

    ("i,model_weight", "Torch Model Weight",
     cxxopts::value<std::string>())

    ("s,sample_file", "SQuAD Sample File",
     cxxopts::value<std::string>())

    ("k,test_scenario", "Test scenario [Offline, Server]",
     cxxopts::value<std::string>()->default_value("Offline"))

    ("n,inter_parallel", "Instance Number",
     cxxopts::value<int>()->default_value("20"))

    ("j,intra_parallel", "Thread Number Per-Instance",
     cxxopts::value<int>()->default_value("4"))

    ("c,mlperf_config", "Configuration File for LoadGen",
     cxxopts::value<std::string>()->default_value("mlperf.conf"))

    ("u,user_config", "User Configuration for LoadGen",
     cxxopts::value<std::string>()->default_value("user.conf"))

    ("o,output_dir", "Test Output Directory",
     cxxopts::value<std::string>()->default_value("mlperf_output"))

    ("b,batch", "Offline Model Batch Size",
     cxxopts::value<int>()->default_value("4"))

    ("w,watermark", "Offline Model sequence length watermark",
     cxxopts::value<int>()->default_value("875"))

    ("disable-hyperthreading", "Whether system enabled hyper-threading or not",
     cxxopts::value<bool>()->default_value("false"))

    ("a,accuracy", "Run test in accuracy mode instead of performance",
     cxxopts::value<bool>()->default_value("false"))

    ("p,profiler", "Whether output trace json or not",
     cxxopts::value<bool>()->default_value("false"))

    ("f,profiler_folder", "If profiler is True, output json in profiler_folder",
     cxxopts::value<std::string>()->default_value("logs"))

    ("l,minilm", "Whether run minilm l12 model ",
     cxxopts::value<bool>()->default_value("false"))

    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto model_conf = parsed_opts["model_conf"].as<std::string>();
  auto model_weight = parsed_opts["model_weight"].as<std::string>();
  auto sample_file = parsed_opts["sample_file"].as<std::string>();
  auto inter_parallel = parsed_opts["inter_parallel"].as<int>();
  auto intra_parallel = parsed_opts["intra_parallel"].as<int>();
  auto output_dir = parsed_opts["output_dir"].as<std::string>();
  auto mlperf_conf = parsed_opts["mlperf_config"].as<std::string>();
  auto user_conf = parsed_opts["user_config"].as<std::string>();
  auto batch_size = parsed_opts["batch"].as<int>();
  auto watermark = parsed_opts["watermark"].as<int>();
  auto disable_ht = parsed_opts["disable-hyperthreading"].as<bool>();
  auto test_scenario = parsed_opts["test_scenario"].as<std::string>();
  auto accuracy_mode = parsed_opts["accuracy"].as<bool>();
  auto profiler_flag = parsed_opts["profiler"].as<bool>();
  auto minilm_flag = parsed_opts["minilm"].as<bool>();
  auto profiler_folder = parsed_opts["profiler_folder"].as<std::string>();
  mlperf::TestSettings testSettings;
  mlperf::LogSettings logSettings;
  // Create SUT as well as QSL

  qsl::SquadQuerySampleLibrary qsl_(sample_file);
  if (minilm_flag) {
    qsl_.minilm = true;
  }

  BertSUT sut(model_conf, model_weight, sample_file, test_scenario,
    inter_parallel, intra_parallel, batch_size, true, minilm_flag);
  if (test_scenario == "Offline") {
    testSettings.scenario = mlperf::TestScenario::Offline;
    testSettings.FromConfig(mlperf_conf, "bert", "Offline");
    testSettings.FromConfig(user_conf, "bert", "Offline");
  } else {
    testSettings.scenario = mlperf::TestScenario::Server;
    testSettings.FromConfig(mlperf_conf, "bert", "Server");
    testSettings.FromConfig(user_conf, "bert", "Server");
  }

  if (accuracy_mode)
    testSettings.mode = mlperf::TestMode::AccuracyOnly;

  logSettings.log_output.outdir = output_dir;
  if (!accuracy_mode) {
    std::cout << "mlperf sut sleep 400s for model initialization \n";
    std::this_thread::sleep_for(std::chrono::seconds(400));
  }
  mlperf::StartTest(&sut, sut.GetQSL(), testSettings, logSettings);
  return 0;
}
