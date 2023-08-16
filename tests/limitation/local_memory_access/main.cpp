/******************************************************************************
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

#define _XOPEN_SOURCE

#include <stdio.h>
#include "utils/utils.hpp"
#include "xetla.hpp"
#include <gtest/gtest.h>

using namespace cl::sycl;
using namespace std;

static bool file_search(const char *file, const char *keyword) {
    return true;
}

static std::string get_self_path() {
    char path[256] = {0};
    auto nchar = readlink("/proc/self/exe", path, sizeof(path));
    assert(nchar != -1);

    std::string filename(path);
    const size_t last_slash_idx = filename.rfind('\/');
    assert(std::string::npos != last_slash_idx);

    return filename.substr(0, last_slash_idx);
}

static void run_cmd(std::string &cmd, bool success = true) {
    const auto BUFSIZE = 512;
    char buf[BUFSIZE] = {0};
    FILE *fp;

    if ((fp = popen(cmd.c_str(), "r")) == NULL) { FAIL(); }

    while (fgets(buf, BUFSIZE, fp) != NULL) {
        std::string msg(buf);
        if (msg.find("Unresolved Symbol") != std::string::npos) { FAIL(); }
    }

    auto code = pclose(fp);
    if (code != 0 && success) {
        std::cout << "Error code: " << code << std::endl;
        FAIL();
    }
}

static void run_case(const char *id, bool success = true) {
    std::string test_path = get_self_path();
    std::string xetla_path = test_path + "/../../../../";
    std::string cpp_file = test_path + "/" + id + ".cpp";
    std::string a_out = test_path + "/" + id + ".out";
    std::string log_file = test_path + "/" + id + ".log";
    std::string clean_cache = "rm -rf " + a_out + "; rm -rf " + log_file;

    std::string cmd = clean_cache + "; icpx -fsycl -std=c++20 -DXETPP_NEW_XMAIN -DDEBUG -isystem " +
                          xetla_path + " -isystem " + xetla_path + "/include " +
                          " -lblas -Xs '-doubleGRF -Xfinalizer -printregusage -Xfinalizer -DPASTokenReduction  -Xfinalizer -enableBCR' " +
                          cpp_file + " -o " + a_out + " && " + a_out + "&> " + log_file;

    run_cmd(cmd, success);
}

// Base-address of the matrix must be 64B aligned.
TEST(local_memory_access, base_address) {
    std::string id = "base_address";
    run_case(id.c_str());

    std::string test_path = get_self_path();
    std::string xetla_path = test_path + "/../../../../";
    std::string cpp_file = test_path + "/" + id + ".cpp";
    std::string a_out = test_path + "/" + id + ".out";
    std::string log_file = test_path + "/" + id + ".log";
    file_search(log_file.c_str(), "Base-address of SLM must be 4B aligned");
}

// Leading-dimension size of the matrix must be multiple of 8B aligned and must be equal or greater than 64B
TEST(local_memory_access, layout) {
    //run_case("layout");
}
