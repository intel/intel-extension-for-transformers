#!/bin/bash
set -x
#set -eo pipefail
WORKSPACE=generated
last_log_path=FinalReport
summaryLog=${WORKSPACE}/summary.log
summaryLogLast=${last_log_path}/summary.log
tuneLog=${WORKSPACE}/tuning_info.log
tuneLogLast=${last_log_path}/tuning_info.log
llmsummaryLog=${WORKSPACE}/log/llm/llm_summary.log
llmsummaryLogLast=${last_log_path}/log/llm/llm_summary.log
cppsummaryLog=${WORKSPACE}/log/cpp_graph/cpp_graph_summary.log
cppsummaryLogLast=${last_log_path}/log/cpp_graph/cpp_graph_summary.log
inferencerSummaryLog=${WORKSPACE}/inferencer.log
inferencerSummaryLogLast=${last_log_path}/inferencer.log
PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"; do
    case $i in
    --workflow=*)
        workflow=$(echo $i | sed "s/${PATTERN}//")
        ;;
    *)
        echo "Parameter $i not recognized."
        exit 1
        ;;
    esac
done

function main {
    echo "summaryLog: ${summaryLog}"
    echo "summaryLogLast: ${summaryLogLast}"
    echo "tunelog: ${tuneLog}"
    echo "last tunelog: ${tuneLogLast}"
    echo "llmsummaryLog: ${llmsummaryLog}"
    echo "llmsummaryLogLast: ${llmsummaryLogLast}"
    echo "cppsummaryLog: ${cppsummaryLog}"
    echo "cppsummaryLogLast: ${cppsummaryLogLast}"
    echo "is_perf_reg=false" >> "$GITHUB_ENV"

    generate_html_head
    generate_html_overview
    if [[ -f ${summaryLog} ]]; then
        if [[ $workflow == "optimize" ]]; then
            generate_optimize_results
        elif [[ $workflow == "deploy" ]]; then
            generate_deploy_results
            generate_deploy_benchmark
        fi
    fi
    if [[ -f ${llmsummaryLog} ]]; then
        generate_llm_results
    fi
    if [[ -f ${cppsummaryLog} ]]; then
        generate_cpp_graph_benchmark
    fi
    generate_html_footer
}

function generate_html_overview {
    Test_Info_Title="<th colspan="4">Test Branch</th> <th colspan="4">Commit ID</th> "
    Test_Info="<th colspan="4">${MR_source_branch}</th> <th colspan="4">${ghprbActualCommit}</th> "

    cat >>${WORKSPACE}/report.html <<eof

<body>
    <div id="main">
        <h1 align="center">ITREX Tests
        [ <a href="${RUN_DISPLAY_URL}">Job-${BUILD_NUMBER}</a> ]</h1>
      <h1 align="center">Test Status: ${JOB_STATUS}</h1>
        <h2>Summary</h2>
        <table class="features-table">
            <tr>
              <th>Repo</th>
              ${Test_Info_Title}
              </tr>
              <tr>
                    <td><a href="https://github.com/intel/intel-extension-for-transformers">ITREX</a></td>
              ${Test_Info}
                </tr>
        </table>
eof
}

function generate_optimize_results {

    cat >>${WORKSPACE}/report.html <<eof
    <h2>Optimize Result</h2>
      <table class="features-table">
          <tr>
                <th rowspan="2">Platform</th>
                <th rowspan="2">System</th>
                <th rowspan="2">Framework</th>
                <th rowspan="2">Version</th>
                <th rowspan="2">Model</th>
                <th rowspan="2">VS</th>
                <th rowspan="2">Tuning<br>Time(s)</th>
                <th rowspan="2">Tuning<br>Count</th>
                <th colspan="6">INT8/BF16</th>
                <th colspan="6">FP32</th>
                <th colspan="3" class="col-cell col-cell1 col-cellh">Ratio</th>
          </tr>
          <tr>
                <th>bs</th>
                <th>imgs/s</th>
                <th>bs</th>
                <th>imgs/s</th>
                <th>bs</th>
                <th>top1</th>

                <th>bs</th>
                <th>imgs/s</th>
                <th>bs</th>
                <th>imgs/s</th>
                <th>bs</th>
                <th>top1</th>

                <th class="col-cell col-cell1">Throughput<br><font size="2px">INT8/FP32>=2</font></th>
                <th class="col-cell col-cell1">Benchmark<br><font size="2px">INT8/FP32>=2</font></th>
                <th class="col-cell col-cell1">Accuracy<br><font size="2px">(INT8-FP32)/FP32>=-0.01</font></th>
          </tr>
eof

    oses=$(sed '1d' ${summaryLog} | cut -d';' -f1 | awk '!a[$0]++')
    for os in ${oses[@]}; do
        platforms=$(sed '1d' ${summaryLog} | grep "^${os}" | cut -d';' -f2 | awk '!a[$0]++')
        for platform in ${platforms[@]}; do
            frameworks=$(sed '1d' ${summaryLog} | grep "^${os};${platform};optimize" | cut -d';' -f4 | awk '!a[$0]++')
            for framework in ${frameworks[@]}; do
                fw_versions=$(sed '1d' ${summaryLog} | grep "^${os};${platform};optimize;${framework}" | cut -d';' -f5 | awk '!a[$0]++')
                for fw_version in ${fw_versions[@]}; do
                    models=$(sed '1d' ${summaryLog} | grep "^${os};${platform};optimize;${framework};${fw_version}" | cut -d';' -f7 | awk '!a[$0]++')
                    for model in ${models[@]}; do
                        current_values=$(generate_inference ${summaryLog} "optimize")
                        last_values=$(generate_inference ${summaryLogLast} "optimize")
                        if [[ ${model} == "gpt-j-6b" ]] || [[ ${model} == "llama-7b-hf" ]] || [[ ${model} == "llama-2-7b-chat" ]] || [[ ${model} == "stable_diffusion" ]] || [[ ${model} == "gpt-j-6b-pruned" ]]; then
                            local_mode="latency"
                        else
                            local_mode="performance"
                        fi
                        generate_tuning_core "optimize" "${local_mode}"
                    done
                done
            done
        done
    done

    cat >>${WORKSPACE}/report.html <<eof
    </table>
eof
}

function generate_deploy_results {

    cat >>${WORKSPACE}/report.html <<eof
    <h2>Deploy Result</h2>
      <table class="features-table">
          <tr>
                <th rowspan="2">Platform</th>
                <th rowspan="2">System</th>
                <th rowspan="2">Framework</th>
                <th rowspan="2">Version</th>
                <th rowspan="2">Model</th>
                <th rowspan="2">VS</th>
                <th rowspan="2">Tuning<br>Time(s)</th>
                <th rowspan="2">Tuning<br>Count</th>
                <th colspan="4">INT8</th>
                <th colspan="4">FP32</th>
                <th colspan="4">BF16</th>
                <th colspan="4">FP8</th>
                <th colspan="4">DynamicINT8</th>
                <th colspan="8" class="col-cell col-cell1 col-cellh">Ratio</th>
          </tr>
          <tr>
                <th>bs</th>
                <th>imgs/s</th>
                <th>bs</th>
                <th>top1</th>

                <th>bs</th>
                <th>imgs/s</th>
                <th>bs</th>
                <th>top1</th>

                <th>bs</th>
                <th>imgs/s</th>
                <th>bs</th>
                <th>top1</th>

                <th>bs</th>
                <th>imgs/s</th>
                <th>bs</th>
                <th>top1</th>

                <th>bs</th>
                <th>imgs/s</th>
                <th>bs</th>
                <th>top1</th>

                <th class="col-cell col-cell1">Throughput<br><font size="2px">INT8/FP32>=2</font></th>
                <th class="col-cell col-cell1">Accuracy<br><font size="2px">(INT8-FP32)/FP32>=-0.01</font></th>
                <th class="col-cell col-cell1">Throughput<br><font size="2px">BF16/FP32>=2</font></th>
                <th class="col-cell col-cell1">Accuracy<br><font size="2px">(BF16-FP32)/FP32>=-0.01</font></th>
                <th class="col-cell col-cell1">Throughput<br><font size="2px">FP8/FP32>=2</font></th>
                <th class="col-cell col-cell1">Accuracy<br><font size="2px">(FP8-FP32)/FP32>=-0.01</font></th>
                <th class="col-cell col-cell1">Throughput<br><font size="2px">DynamicINT8/FP32>=2</font></th>
                <th class="col-cell col-cell1">Accuracy<br><font size="2px">(DynamicINT8-FP32)/FP32>=-0.01</font></th>
          </tr>
eof

    oses=$(sed '1d' ${summaryLog} | cut -d';' -f1 | awk '!a[$0]++')

    for os in ${oses[@]}; do
        platforms=$(sed '1d' ${summaryLog} | grep "^${os}" | cut -d';' -f2 | awk '!a[$0]++')
        for platform in ${platforms[@]}; do
            frameworks=$(sed '1d' ${summaryLog} | grep "^${os};${platform};deploy" | cut -d';' -f4 | awk '!a[$0]++')
            for framework in ${frameworks[@]}; do
                fw_versions=$(sed '1d' ${summaryLog} | grep "^${os};${platform};deploy;${framework}" | cut -d';' -f5 | awk '!a[$0]++')
                for fw_version in ${fw_versions[@]}; do
                    models=$(sed '1d' ${summaryLog} | grep "^${os};${platform};deploy;${framework};${fw_version}" | cut -d';' -f7 | awk '!a[$0]++')
                    for model in ${models[@]}; do
                        current_values=$(generate_inference ${summaryLog} "deploy")
                        last_values=$(generate_inference ${summaryLogLast} "deploy")
                        echo $last_values
                        if [[ ${model} == "gpt-j-6b" ]] || [[ ${model} == "llama-7b-hf" ]] || [[ ${model} == "llama-2-7b-chat" ]] || [[ ${model} == "stable_diffusion" ]] || [[ ${model} == "gpt-j-6b-pruned" ]]; then
                            local_mode="latency"
                        else
                            local_mode="performance"
                        fi
                        generate_tuning_core "deploy" "${local_mode}"
                    done
                done
            done
        done
    done

    cat >>${WORKSPACE}/report.html <<eof
    </table>
eof
}

function generate_deploy_benchmark {

    cat >>${WORKSPACE}/report.html <<eof
    <h2>Deploy Inferencer</h2>
      <table class="features-table">
        <tr>
          <th rowspan="2">Model</th>
          <th rowspan="2">Seq_len</th>
          <th rowspan="2">VS</th>
          <th rowspan="2">Full<br>Cores</th>
          <th rowspan="2">NCores<br>per Instance</th>
          <th rowspan="2">BS</th>
          <th>INT8</th>
          <th>FP32</th>
          <th colspan="2" class="col-cell col-cell1 col-cellh">Ratio</th>
        </tr>
        <tr>
          <th>throughput</th>
          <th>throughput</th>
          <th colspan="2" class="col-cell col-cell1"><font size="2px">FP32/INT8</font></th>
        </tr>
eof

    mode='throughput'
    models=$(cat ${inferencerSummaryLog} | grep "${mode}," | cut -d',' -f3 | awk '!a[$0]++')
    for model in ${models[@]}; do
        seq_lens=$(cat ${inferencerSummaryLog} | grep "${mode},${model}," | cut -d',' -f4 | awk '!a[$0]++')
        for seq_len in ${seq_lens[@]}; do
            full_cores=$(cat ${inferencerSummaryLog} | grep "${mode},${model},${seq_len}," | cut -d',' -f5 | awk '!a[$0]++')
            for full_core in ${full_cores[@]}; do
                core_per_inss=$(cat ${inferencerSummaryLog} | grep "${mode},${model},${seq_len},${full_core}," | cut -d',' -f6 | awk '!a[$0]++')
                for core_per_ins in ${core_per_inss[@]}; do
                    bss=$(cat ${inferencerSummaryLog} | grep "${mode},${model},${seq_len},${full_core},${core_per_ins}," | cut -d',' -f7 | awk '!a[$0]++')
                    for bs in ${bss[@]}; do
                        benchmark_pattern="${mode},${model},${seq_len},${full_core},${core_per_ins},${bs}"
                        benchmark_int8=$(cat ${inferencerSummaryLog} | grep "${benchmark_pattern},int8" | cut -d',' -f9)
                        benchmark_int8_url=$(cat ${inferencerSummaryLog} | grep "${benchmark_pattern}," | tail -1 | cut -d',' -f10)
                        benchmark_fp32=$(cat ${inferencerSummaryLog} | grep "${benchmark_pattern},fp32" | cut -d',' -f9)
                        benchmark_fp32_url=$(cat ${inferencerSummaryLog} | grep "${benchmark_pattern},fp32" | cut -d',' -f10)
                        if [ $(cat ${inferencerSummaryLogLast} | grep -c "${benchmark_pattern},int8") == 0 ]; then
                            benchmark_int8_last=nan
                            benchmark_int8_url_last=nan
                            benchmark_fp32_last=nan
                            benchmark_fp32_url_last=nan
                        else
                            benchmark_int8_last=$(cat ${inferencerSummaryLogLast} | grep "${benchmark_pattern},int8" | cut -d',' -f9)
                            benchmark_int8_url_last=$(cat ${inferencerSummaryLogLast} | grep "${benchmark_pattern},int8" | cut -d',' -f10)
                            benchmark_fp32_last=$(cat ${inferencerSummaryLogLast} | grep "${benchmark_pattern},fp32" | cut -d',' -f9)
                            benchmark_fp32_url_last=$(cat ${inferencerSummaryLogLast} | grep "${benchmark_pattern},fp32" | cut -d',' -f10)
                        fi
                        generate_perf_core
                    done
                done
            done
        done
    done
    cat >>${WORKSPACE}/report.html <<eof
    </table>
eof
}

function generate_perf_core {
    echo "<tr><td rowspan=3>${model}</td><td rowspan=3>${seq_len}</td><td>New</td><td rowspan=2>${full_core}</td><td rowspan=2>${core_per_ins}</td><td rowspan=2>${bs}</td>" >>${WORKSPACE}/report.html

    echo | awk -v b_int8=${benchmark_int8} -v b_int8_url=${benchmark_int8_url} -v b_fp32=${benchmark_fp32} -v b_fp32_url=${benchmark_fp32_url} -v b_int8_l=${benchmark_int8_last} -v b_int8_url_l=${benchmark_int8_url_last} -v b_fp32_l=${benchmark_fp32_last} -v b_fp32_url_l=${benchmark_fp32_url_last} '
        function show_benchmark(a,b) {
            if(a ~/[1-9]/) {
                    printf("<td><a href=%s>%.2f</a></td>\n",b,a);
            }else {
                if(a == "") {
                    printf("<td><a href=%s>%s</a></td>\n",b,a);
                }else{
                    printf("<td></td>\n");
                }
            }
        }

        function compare_current(a,b) {

            if(a ~/[1-9]/ && b ~/[1-9]/) {
                target = a / b;
                if(target >= 2) {
                   printf("<td style=\"background-color:#90EE90\">%.2f</td>", target);
                }else if(target < 1) {
                   printf("<td style=\"background-color:#FFD2D2\">%.2f</td>", target);
                   job_status = "fail"
                }else{
                   printf("<td>%.2f</td>", target);
                }
            }else{
                printf("<td></td>");
            }

        }

        function compare_new_last(a,b){
            if(a ~/[1-9]/ && b ~/[1-9]/) {
                target = a / b;
                if(target >= 0.945) {
                    status_png = "background-color:#90EE90";
                }else {
                    status_png = "background-color:#FFD2D2";
                    perf_status = "fail"
                }
                printf("<td style=\"%s\">%.2f</td>", status_png, target);
            }else{
                if(a == ""){
                    job_status = "fail"
                    status_png = "background-color:#FFD2D2";
                    printf("<td style=\"%s\"></td>", status_png);
                }else{
                    printf("<td class=\"col-cell col-cell3\"></td>");
                }
            }
        }

        function compare_ratio(int8_perf_value, fp32_perf_value, last_int8_perf_value, last_fp32_perf_value) {
            if (int8_perf_value ~/[1-9]/ && fp32_perf_value ~/[1-9]/ && last_int8_perf_value ~/[1-9]/ && last_fp32_perf_value ~/[1-9]/) {
                new_result = int8_perf_value / fp32_perf_value
                previous_result = last_int8_perf_value / last_fp32_perf_value
                target = new_result / previous_result;
                if (target <= 1.054 && target >= 0.945) {
                    status_png = "background-color:#90EE90";
                } else {
                    status_png = "background-color:#FFD2D2";
                    ratio_status = "fail"
                }
                printf("<td style=\"%s\">%.2f</td>", status_png, target);
            } else {
                if (new_result == nan && previous_result == nan) {
                    printf("<td class=\"col-cell col-cell3\"></td>");
                } else {
                    if (new_result == nan) {
                        ratio_status = "fail"
                        status_png = "background-color:#FFD2D2";
                        printf("<td style=\"%s\"></td>", status_png);
                    } else {
                        printf("<td class=\"col-cell col-cell3\"></td>");
                    }
                }
            }
        }


        BEGIN {
            job_status = "pass"
            perf_status = "pass"
            ratio_status = "pass"
        }{
            // current
            show_benchmark(b_int8,b_int8_url)
            show_benchmark(b_fp32,b_fp32_url)

            // current comparison
            compare_current(b_int8,b_fp32)

            // Last
            printf("</tr>\n<tr><td>Last</td>")
            show_benchmark(b_int8_l,b_int8_url_l)
            show_benchmark(b_fp32_l,b_fp32_url_l)

            compare_current(b_int8_l,b_fp32_l)

            // current vs last
            printf("</tr>\n<tr><td>New/Last</td><td colspan=3 class=\"col-cell3\"></td>");
            compare_new_last(b_int8,b_int8_l)
            compare_new_last(b_fp32,b_fp32_l)


            // Compare INT8 FP32 Performance ratio
            compare_ratio(b_int8, b_fp32, b_int8_l, b_fp32_l);

            printf("</tr>\n");

            status = (perf_status == "fail" && ratio_status == "fail") ? "fail" : "pass"
            status = (job_status == "fail") ? "fail" : status
        } END{
          printf("\n%s", status);
        }
    ' >>${WORKSPACE}/report.html
    job_state=$(tail -1 ${WORKSPACE}/report.html)
    sed -i '$s/.*//' ${WORKSPACE}/report.html
    if [ ${job_state} == 'fail' ]; then
        echo "is_perf_reg=true" >> "$GITHUB_ENV"
    fi
}

function generate_inference {
    local workflow=$2
    awk -v framework="${framework}" -v workflow="${workflow}" -v fw_version="${fw_version}" -v model="${model}" -v os="${os}" -v platform=${platform} -F ';' '
        BEGINE {
            fp32_perf_bs = "nan";
            fp32_perf_value = "nan";
            fp32_perf_url = "nan";
            fp32_acc_bs = "nan";
            fp32_acc_value = "nan";
            fp32_acc_url = "nan";
            fp32_benchmark_value = "nan";
            fp32_benchmark_url = "nan";

            int8_perf_bs = "nan";
            int8_perf_value = "nan";
            int8_perf_url = "nan";
            int8_acc_bs = "nan";
            int8_acc_value = "nan";
            int8_acc_url = "nan";
            int8_benchmark_value = "nan";
            int8_benchmark_url = "nan";

            bf16_perf_bs = "nan";
            bf16_perf_value = "nan";
            bf16_perf_url = "nan";
            bf16_acc_bs = "nan";
            bf16_acc_value = "nan";
            bf16_acc_url = "nan";
            bf16_benchmark_value = "nan";
            bf16_benchmark_url = "nan";

            fp8_perf_bs = "nan";
            fp8_perf_value = "nan";
            fp8_perf_url = "nan";
            fp8_acc_bs = "nan";
            fp8_acc_value = "nan";
            fp8_acc_url = "nan";
            fp8_benchmark_value = "nan";
            fp8_benchmark_url = "nan";

            dint8_perf_bs = "nan";
            dint8_perf_value = "nan";
            dint8_perf_url = "nan";
            dint8_acc_bs = "nan";
            dint8_acc_value = "nan";
            dint8_acc_url = "nan";
            dint8_benchmark_value = "nan";
            dint8_benchmark_url = "nan";
        }{
            if($1 == os && $2 == platform && $3 == workflow && $4 == framework && $5 == fw_version && $7 == model) {
                // FP32
                if($6 == "FP32") {
                    // Performance
                    if($9 == "Performance" || $9 == "Latency") {
                        fp32_perf_bs = $10;
                        fp32_perf_value = $11;
                        fp32_perf_url = $12;
                    }
                    // Accuracy
                    if($9 == "Accuracy" || $9 == "accuracy") {
                        fp32_acc_bs = $10;
                        fp32_acc_value = $11;
                        fp32_acc_url = $12;
                    }
                    // Benchmark
                    if($9 == "Benchmark" || $9 == "benchmark_only") {
                        fp32_bench_bs = $10;
                        fp32_bench_value = $11;
                        fp32_bench_url = $12;
                    }
                }
                // INT8
                if($6 == "INT8") {
                    // Performance
                    if($9 == "Performance" || $9 == "Latency") {
                        int8_perf_bs = $10;
                        int8_perf_value = $11;
                        int8_perf_url = $12;
                    }
                    // Accuracy
                    if($9 == "Accuracy" || $9 == "accuracy") {
                        int8_acc_bs = $10;
                        int8_acc_value = $11;
                        int8_acc_url = $12;
                    }
                    // Benchmark
                    if($9 == "Benchmark" || $9 == "benchmark_only") {
                        int8_bench_bs = $10;
                        int8_bench_value = $11;
                        int8_bench_url = $12;
                    }
                }
                if($6 == "BF16") {
                    // Performance
                    if($9 == "Performance" || $9 == "Latency") {
                        bf16_perf_bs = $10;
                        bf16_perf_value = $11;
                        bf16_perf_url = $12;
                    }
                    // Accuracy
                    if($9 == "Accuracy" || $9 == "accuracy") {
                        bf16_acc_bs = $10;
                        bf16_acc_value = $11;
                        bf16_acc_url = $12;
                    }
                    // Benchmark
                    if($9 == "Benchmark" || $9 == "benchmark_only") {
                        bf16_bench_bs = $10;
                        bf16_bench_value = $11;
                        bf16_bench_url = $12;
                    }
                }
                if($6 == "DYNAMIC_INT8") {
                    // Performance
                    if($9 == "Performance" || $9 == "Latency") {
                        dint8_perf_bs = $10;
                        dint8_perf_value = $11;
                        dint8_perf_url = $12;
                    }
                    // Accuracy
                    if($9 == "Accuracy" || $9 == "accuracy") {
                        dint8_acc_bs = $10;
                        dint8_acc_value = $11;
                        dint8_acc_url = $12;
                    }
                    // Benchmark
                    if($9 == "Benchmark" || $9 == "benchmark_only") {
                        dint8_bench_bs = $10;
                        dint8_bench_value = $11;
                        dint8_bench_url = $12;
                    }
                }
                if($6 == "FP8") {
                    // Performance
                    if($9 == "Performance" || $9 == "Latency") {
                        fp8_perf_bs = $10;
                        fp8_perf_value = $11;
                        fp8_perf_url = $12;
                    }
                    // Accuracy
                    if($9 == "Accuracy" || $9 == "accuracy") {
                        fp8_acc_bs = $10;
                        fp8_acc_value = $11;
                        fp8_acc_url = $12;
                    }
                    // Benchmark
                    if($9 == "Benchmark" || $9 == "benchmark_only") {
                        fp8_bench_bs = $10;
                        fp8_bench_value = $11;
                        fp8_bench_url = $12;
                    }
                }
            }
        }END {
            printf("%s;%s;%s;%s;%s;%s;", int8_perf_bs,int8_perf_value,int8_bench_bs,int8_bench_value,int8_acc_bs,int8_acc_value);
            printf("%s;%s;%s;%s;%s;%s;", fp32_perf_bs,fp32_perf_value,fp32_bench_bs,fp32_bench_value,fp32_acc_bs,fp32_acc_value);
            printf("%s;%s;%s;%s;%s;%s;", int8_perf_url,int8_bench_url,int8_acc_url,fp32_perf_url,fp32_bench_url,fp32_acc_url);
            printf("%s;%s;%s;%s;%s;%s;%s;%s;%s;", bf16_perf_bs,bf16_perf_value,bf16_perf_url,bf16_acc_bs,bf16_acc_value,bf16_acc_url,bf16_bench_bs,bf16_bench_value,bf16_bench_url);
            printf("%s;%s;%s;%s;%s;%s;%s;%s;%s;", dint8_perf_bs,dint8_perf_value,dint8_perf_url,dint8_acc_bs,dint8_acc_value,dint8_acc_url,dint8_bench_bs,dint8_bench_value,dint8_bench_url);
            printf("%s;%s;%s;%s;%s;%s;%s;%s;%s;", fp8_perf_bs,fp8_perf_value,fp8_perf_url,fp8_acc_bs,fp8_acc_value,fp8_acc_url,fp8_bench_bs,fp8_bench_value,fp8_bench_url);
        }
    ' "$1"
}

function generate_llm_results {
    cat >>${WORKSPACE}/report.html <<eof
    <h2>LLM Inferencer</h2>
      <table class="features-table">
        <tr>
          <th>Model</th>
          <th>Input</th>
          <th>Output</th>
          <th>Batchsize</th>
          <th>Cores/Instance</th>
          <th>Precision</th>
          <th>Beam</th>
          <th>VS</th>
          <th>Avg Latency</th>
          <th>Memory</th>
        </tr>
eof

    mode='latency'
    models=$(cat ${llmsummaryLog} | grep "${mode}," | cut -d',' -f3 | awk '!a[$0]++')
    for model in ${models[@]}; do
        precisions=$(cat ${llmsummaryLog} | grep "${mode},${model}," | cut -d',' -f4 | awk '!a[$0]++')
        for precision in ${precisions[@]}; do
            batch_size_list=$(cat ${llmsummaryLog} | grep "${mode},${model},${precision}," | cut -d',' -f5 | awk '!a[$0]++')
            for batch_size in ${batch_size_list[@]}; do
                input_token_list=$(cat ${llmsummaryLog} | grep "${mode},${model},${precision},${batch_size}," | cut -d',' -f6 | awk '!a[$0]++')
                for input_token in ${input_token_list[@]}; do
                    beam_search=$(cat ${llmsummaryLog} | grep "${mode},${model},${precision},${batch_size},${input_token}," | cut -d',' -f8 | awk '!a[$0]++')
                    for beam in ${beam_search[@]}; do
                        benchmark_pattern="${mode},${model},${precision},${batch_size},${input_token},"
                        output_token=$(cat ${llmsummaryLog} | grep "${benchmark_pattern}" | cut -d',' -f7 | awk '!a[$0]++')
                        cores_per_instance=$(cat ${llmsummaryLog} | grep "${benchmark_pattern}" | cut -d',' -f10 | awk '!a[$0]++')
                        count=$(cat ${llmsummaryLog} | grep "${benchmark_pattern}" | cut -d',' -f11 | awk '!a[$0]++')
                        throughput=$(cat ${llmsummaryLog} | grep "${benchmark_pattern}" | cut -d',' -f12 | awk '!a[$0]++')
                        link=$(cat ${llmsummaryLog} | grep "${benchmark_pattern}" | cut -d',' -f13 | awk '!a[$0]++')
                        memory=$(cat ${llmsummaryLog} | grep "${benchmark_pattern}" | cut -d',' -f9 | awk '!a[$0]++')
                        total_latency=$(cat ${llmsummaryLog} | grep "${benchmark_pattern}" | cut -d',' -f14 | awk '!a[$0]++')
                        avg_latency=$(cat ${llmsummaryLog} | grep "${benchmark_pattern}" | cut -d',' -f15 | awk '!a[$0]++')
                        fst_latency=$(cat ${llmsummaryLog} | grep "${benchmark_pattern}" | cut -d',' -f16 | awk '!a[$0]++')
                        p90_latency=$(cat ${llmsummaryLog} | grep "${benchmark_pattern}" | cut -d',' -f17 | awk '!a[$0]++')
                        if [ $(cat ${llmsummaryLogLast} | grep -c "${benchmark_pattern}") == 0 ]; then
                            throughput_last=nan
                            link_last=nan
                            memory_last=nan
                            total_latency_last=nan
                            avg_latency_last=nan
                            fst_latency_last=nan
                            p90_latency_last=nan
                        else
                            throughput_last=$(cat ${llmsummaryLogLast} | grep "${benchmark_pattern}" | cut -d',' -f12 | awk '!a[$0]++')
                            link_last=$(cat ${llmsummaryLogLast} | grep "${benchmark_pattern}" | cut -d',' -f13 | awk '!a[$0]++')
                            memory_last=$(cat ${llmsummaryLogLast} | grep "${benchmark_pattern}" | cut -d',' -f9 | awk '!a[$0]++')
                            total_latency_last=$(cat ${llmsummaryLogLast} | grep "${benchmark_pattern}" | cut -d',' -f14 | awk '!a[$0]++')
                            avg_latency_last=$(cat ${llmsummaryLogLast} | grep "${benchmark_pattern}" | cut -d',' -f15 | awk '!a[$0]++')
                            fst_latency_last=$(cat ${llmsummaryLogLast} | grep "${benchmark_pattern}" | cut -d',' -f16 | awk '!a[$0]++')
                            p90_latency_last=$(cat ${llmsummaryLogLast} | grep "${benchmark_pattern}" | cut -d',' -f17 | awk '!a[$0]++')
                        fi
                        generate_llm_core
                    done
                done
            done
        done
    done
    cat >>${WORKSPACE}/report.html <<eof
    </table>
eof
}

function generate_tuning_core {
    local workflow=$1
    local mode=$2

    tuning_time=$(grep "^${os};${platform};${workflow};${framework};${fw_version};${model};" ${tuneLog} | awk -F';' '{print $8}')
    tuning_count=$(grep "^${os};${platform};${workflow};${framework};${fw_version};${model};" ${tuneLog} | awk -F';' '{print $9}')
    tuning_log=$(grep "^${os};${platform};${workflow};${framework};${fw_version};${model};" ${tuneLog} | awk -F';' '{print $10}')

    echo "<tr><td rowspan=3>${platform}</td><td rowspan=3>${os}</td><td rowspan=3>${framework}</td><td rowspan=3>${fw_version}</td><td rowspan=3>${model}</td><td>New</td>" >>${WORKSPACE}/report.html
    echo "<td><a href=${tuning_log}>${tuning_time}</a></td><td><a href=${tuning_log}>${tuning_count}</a></td>" >>${WORKSPACE}/report.html

    tuning_time=$(grep "^${os};${platform};${workflow};${framework};${fw_version};${model};" ${tuneLogLast} | awk -F';' '{print $8}')
    tuning_count=$(grep "^${os};${platform};${workflow};${framework};${fw_version};${model};" ${tuneLogLast} | awk -F';' '{print $9}')
    tuning_log=$(grep "^${os};${platform};${workflow};${framework};${fw_version};${model};" ${tuneLogLast} | awk -F';' '{print $10}')

    echo | awk -F ';' -v current_values="${current_values}" -v last_values="${last_values}" \
        -v tuning_time="${tuning_time}" \
        -v tuning_count="${tuning_count}" -v tuning_log="${tuning_log}" -v workflow="${workflow}" -v mode=${mode} '

        function abs(x) { return x < 0 ? -x : x }

        function show_new_last(batch, link, value, metric) {
            if(value ~/[1-9]/) {
                if (metric == "perf" || metric == "bench") {
                    printf("<td>%s</td> <td><a href=%s>%.2f</a></td>\n",batch,link,value);
                } else {
                    if (value <= 1){
                        printf("<td>%s</td> <td><a href=%s>%.2f%</a></td>\n",batch,link,value*100);
                    }else{
                        printf("<td>%s</td> <td><a href=%s>%.2f</a></td>\n",batch,link,value);
                    }
                }
            } else {
                if(link == "" || value == "N/A") {
                    printf("<td></td> <td></td>\n");
                } else {
                    printf("<td>%s</td> <td><a href=%s>Failure</a></td>\n",batch,link);
                }
            }
        }

        function compare_current(int8_result, fp32_result, metric) {

            if(int8_result ~/[1-9]/ && fp32_result ~/[1-9]/) {
                if(metric == "acc") {
                    target = (int8_result - fp32_result) / fp32_result;
                    if(target >= -0.01) {
                       printf("<td rowspan=3 style=\"background-color:#90EE90\">%.2f%</td>", target*100);
                    }else if(target < -0.05) {
                       printf("<td rowspan=3 style=\"background-color:#FFD2D2\">%.2f%</td>", target*100);
                       job_status = "fail"
                    }else{
                       printf("<td rowspan=3>%.2f%</td>", target*100);
                    }
                } else if(metric == "perf" || metric == "bench") {
                    target = int8_result / fp32_result;
                    if(target >= 2) {
                       printf("<td style=\"background-color:#90EE90\">%.2f</td>", target);
                    }else if(target < 1) {
                       printf("<td style=\"background-color:#FFD2D2\">%.2f</td>", target);
                       job_status = "fail"
                    }else{
                       printf("<td>%.2f</td>", target);
                    }
                } else {
                    // latency mode
                    target = fp32_result / int8_result;
                    if(target >= 2) {
                       printf("<td rowspan=3 style=\"background-color:#90EE90\">%.2f</td>", target);
                    }else if(target < 1) {
                       printf("<td rowspan=3 style=\"background-color:#FFD2D2\">%.2f</td>", target);
                       job_status = "fail"
                    }else{
                       printf("<td rowspan=3>%.2f</td>", target);
                    }
                }
            }else {
                printf("<td rowspan=3></td>");
            }
        }

        function compare_result(new_result, previous_result, metric) {

            if (new_result ~/[1-9]/ && previous_result ~/[1-9]/) {
                if(metric == "acc") {
                    target = new_result - previous_result;
                    if(target >= -0.0001 && target <= 0.0001) {
                        status_png = "background-color:#90EE90";
                    } else {
                        status_png = "background-color:#FFD2D2";
                        job_status = "fail"
                    }
                    if (new_result <= 1){
                        printf("<td style=\"%s\" colspan=2>%.2f%</td>", status_png, target*100);
                    }else{
                        printf("<td style=\"%s\" colspan=2>%.2f</td>", status_png, target);
                    }
                } else if (metric == "perf" || metric == "bench"){
                    target = new_result / previous_result;
                    if(target <= 1.084 && target >= 0.915) {
                        status_png = "background-color:#90EE90";
                    } else {
                        status_png = "background-color:#FFD2D2";
                        perf_status = "fail"
                    }
                    printf("<td style=\"%s\" colspan=2>%.2f</td>", status_png, target);
                } else {
                    target = previous_result / new_result;
                    if(target >= 0.95) {
                        status_png = "background-color:#90EE90";
                    } else {
                        status_png = "background-color:#FFD2D2";
                        job_status = "fail"
                    }
                    printf("<td style=\"%s\" colspan=2>%.2f</td>", status_png, target);
                }
            } else {
                if(new_result == "nan" || previous_result == "nan") {
                    printf("<td class=\"col-cell col-cell3\" colspan=2></td>");
                }else {
                    printf("<td style=\"col-cell col-cell3\" colspan=2></td>");
                    job_red++;
                }
            }
        }

        function compare_ratio(int8_perf_value, fp32_perf_value, last_int8_perf_value, last_fp32_perf_value) {
            if (int8_perf_value ~/[1-9]/ && fp32_perf_value ~/[1-9]/ && last_int8_perf_value ~/[1-9]/ && last_fp32_perf_value ~/[1-9]/) {
                new_result = int8_perf_value / fp32_perf_value
                previous_result = last_int8_perf_value / last_fp32_perf_value
                target = new_result / previous_result;
                if (target <= 1.084 && target >= 0.915) {
                    status_png = "background-color:#90EE90";
                } else {
                    status_png = "background-color:#FFD2D2";
                    ratio_status = "fail"
                }
                printf("<td style=\"%s\">%.2f</td>", status_png, target);
            } else {
                if (new_result == nan && previous_result == nan) {
                    printf("<td class=\"col-cell col-cell3\"></td>");
                } else {
                    if (new_result == nan) {
                        ratio_status = "fail"
                        status_png = "background-color:#FFD2D2";
                        printf("<td style=\"%s\"></td>", status_png);
                    } else {
                        printf("<td class=\"col-cell col-cell3\"></td>");
                    }
                }
            }
        }

        BEGIN {
            job_status = "pass"
            perf_status = "pass"
            ratio_status = "pass"
        }{
            // Current values
            split(current_values,current_value,";");

            // INT8 Performance results
            int8_perf_batch=current_value[1]
            int8_perf_value=current_value[2]
            int8_perf_url=current_value[13]
            show_new_last(int8_perf_batch, int8_perf_url, int8_perf_value, "perf");
            if (workflow == "optimize") {
                // INT8 Bench results
                int8_bench_batch=current_value[3]
                int8_bench_value=current_value[4]
                int8_bench_url=current_value[14]
                show_new_last(int8_bench_batch, int8_bench_url, int8_bench_value, "bench");
            }
            // INT8 Accuracy results
            int8_acc_batch=current_value[5]
            int8_acc_value=current_value[6]
            int8_acc_url=current_value[15]
            show_new_last(int8_acc_batch, int8_acc_url, int8_acc_value, "acc");

            // FP32 Performance results
            fp32_perf_batch=current_value[7]
            fp32_perf_value=current_value[8]
            fp32_perf_url=current_value[16]
            show_new_last(fp32_perf_batch, fp32_perf_url, fp32_perf_value, "perf");
            if (workflow == "optimize") {
                // FP32 Bench results
                fp32_bench_batch=current_value[9]
                fp32_bench_value=current_value[10]
                fp32_bench_url=current_value[17]
                show_new_last(fp32_bench_batch, fp32_bench_url, fp32_bench_value, "bench");
            }
            // FP32 Accuracy results
            fp32_acc_batch=current_value[11]
            fp32_acc_value=current_value[12]
            fp32_acc_url=current_value[18]
            show_new_last(fp32_acc_batch, fp32_acc_url, fp32_acc_value, "acc");

            // BF16 Performance results
            if (workflow == "deploy") {
                bf16_perf_batch=current_value[19]
                bf16_perf_value=current_value[20]
                bf16_perf_url=current_value[21]
                show_new_last(bf16_perf_batch, bf16_perf_url, bf16_perf_value, "perf");

                // BF16 Accuracy results
                bf16_acc_batch=current_value[22]
                bf16_acc_value=current_value[23]
                bf16_acc_url=current_value[24]
                show_new_last(bf16_acc_batch, bf16_acc_url, bf16_acc_value, "acc");

                fp8_perf_batch=current_value[37]
                fp8_perf_value=current_value[38]
                fp8_perf_url=current_value[39]
                show_new_last(fp8_perf_batch, fp8_perf_url, fp8_perf_value, "perf");

                // fp8 Accuracy results
                fp8_acc_batch=current_value[40]
                fp8_acc_value=current_value[41]
                fp8_acc_url=current_value[42]
                show_new_last(fp8_acc_batch, fp8_acc_url, fp8_acc_value, "acc");

                dint8_perf_batch=current_value[28]
                dint8_perf_value=current_value[29]
                dint8_perf_url=current_value[30]
                show_new_last(dint8_perf_batch, dint8_perf_url, dint8_perf_value, "perf");

                // Dynamic INT8 Accuracy results
                dint8_acc_batch=current_value[31]
                dint8_acc_value=current_value[32]
                dint8_acc_url=current_value[33]
                show_new_last(dint8_acc_batch, dint8_acc_url, dint8_acc_value, "acc");
            }
            
            // Compare Current
            if (mode == "performance") {
                compare_current(int8_perf_value, fp32_perf_value, "perf");
            } else {
                compare_current(int8_perf_value, fp32_perf_value, "latency");
            }
            if (workflow == "optimize") {
                compare_current(int8_bench_value, fp32_bench_value, "bench")
            }
            compare_current(int8_acc_value, fp32_acc_value, "acc");

            if (workflow == "deploy") {
                if (mode == "performance") {
                    compare_current(bf16_perf_value, fp32_perf_value, "perf");
                } else {
                    compare_current(bf16_perf_value, fp32_perf_value, "latency");
                }
                compare_current(bf16_acc_value, fp32_acc_value, "acc");
                
                if (mode == "performance") {
                    compare_current(fp8_perf_value, fp32_perf_value, "perf");
                } else {
                    compare_current(fp8_perf_value, fp32_perf_value, "latency");
                }
                compare_current(fp8_acc_value, fp32_acc_value, "acc");

                if (mode == "performance") {
                    compare_current(dint8_perf_value, fp32_perf_value, "perf");
                } else {
                    compare_current(dint8_perf_value, fp32_perf_value, "latency");
                }
                compare_current(dint8_acc_value, fp32_acc_value, "acc");
            }

            // Last values
            split(last_values,last_value,";");

            // Last
            printf("</tr>\n<tr><td>Last</td><td><a href=%3$s>%1$s</a></td><td><a href=%3$s>%2$s</a></td>", tuning_time, tuning_count, tuning_log);

            // Show last INT8 Performance results
            last_int8_perf_batch=last_value[1]
            last_int8_perf_value=last_value[2]
            last_int8_perf_url=last_value[13]
            show_new_last(last_int8_perf_batch, last_int8_perf_url, last_int8_perf_value, "perf");
            if (workflow == "optimize") {
                // INT8 Bench results
                last_int8_bench_batch=last_value[3]
                last_int8_bench_value=last_value[4]
                last_int8_bench_url=last_value[14]
                show_new_last(last_int8_bench_batch, last_int8_bench_url, last_int8_bench_value, "bench");
            }
            // Show last INT8 Accuracy results
            last_int8_acc_batch=last_value[5]
            last_int8_acc_value=last_value[6]
            last_int8_acc_url=last_value[15]
            show_new_last(last_int8_acc_batch, last_int8_acc_url, last_int8_acc_value, "acc");

            // Show last FP32 Performance results
            last_fp32_perf_batch=last_value[7]
            last_fp32_perf_value=last_value[8]
            last_fp32_perf_url=last_value[16]
            show_new_last(last_fp32_perf_batch, last_fp32_perf_url, last_fp32_perf_value, "perf");
            if (workflow == "optimize") {
                // FP32 Bench results
                last_fp32_bench_batch=last_value[9]
                last_fp32_bench_value=last_value[10]
                last_fp32_bench_url=last_value[17]
                show_new_last(last_fp32_bench_batch, last_fp32_bench_url, last_fp32_bench_value, "bench");
            }
            // Show last FP32 Accuracy results
            last_fp32_acc_batch=last_value[11]
            last_fp32_acc_value=last_value[12]
            last_fp32_acc_url=last_value[18]
            show_new_last(last_fp32_acc_batch, last_fp32_acc_url, last_fp32_acc_value, "acc");

            if (workflow == "deploy") {
                // Show last BF16 Performance results
                last_bf16_perf_batch=last_value[19]
                last_bf16_perf_value=last_value[20]
                last_bf16_perf_url=last_value[21]
                show_new_last(last_bf16_perf_batch, last_bf16_perf_url, last_bf16_perf_value, "perf");

                // Show last BF16 Accuracy results
                last_bf16_acc_batch=last_value[22]
                last_bf16_acc_value=last_value[23]
                last_bf16_acc_url=last_value[24]
                show_new_last(last_bf16_acc_batch, last_bf16_acc_url, last_bf16_acc_value, "acc");

                last_fp8_perf_batch=last_value[37]
                last_fp8_perf_value=last_value[38]
                last_fp8_perf_url=last_value[39]
                show_new_last(last_fp8_perf_batch, last_fp8_perf_url, last_fp8_perf_value, "perf");

                // Show last fp8 Accuracy results
                last_fp8_acc_batch=last_value[40]
                last_fp8_acc_value=last_value[41]
                last_fp8_acc_url=last_value[42]
                show_new_last(last_fp8_acc_batch, last_fp8_acc_url, last_fp8_acc_value, "acc");

                // Show last dynamic int8 Performance results
                last_dint8_perf_batch=last_value[28]
                last_dint8_perf_value=last_value[29]
                last_dint8_perf_url=last_value[30]
                show_new_last(last_dint8_perf_batch, last_dint8_perf_url, last_dint8_perf_value, "perf");
                
                // Show last dynamic int8 Accuracy results
                last_dint8_acc_batch=last_value[31]
                last_dint8_acc_value=last_value[32]
                last_dint8_acc_url=last_value[33]
                show_new_last(last_dint8_acc_batch, last_dint8_acc_url, last_dint8_acc_value, "acc");
            }

            compare_current(last_int8_perf_value,last_fp32_perf_value,"perf")
            if (workflow == "optimize") {
                compare_current(last_int8_bench_value, last_fp32_bench_value, "bench")
            }
            if (workflow == "deploy" && dint8_perf_value!="") {
                if (mode == "performance") {
                    compare_current(dint8_perf_value, fp32_perf_value, "perf");
                } else {
                    compare_current(dint8_perf_value, fp32_perf_value, "latency");
                }
            }

            // current vs last
            printf("</tr>\n<tr><td>New/Last</td><td colspan=2 class=\"col-cell3\"></td>");

            // Compare INT8 Performance results
            if (mode == "performance") {
                compare_result(int8_perf_value, last_int8_perf_value,"perf");
            } else {
                compare_result(int8_perf_value, last_int8_perf_value,"latency");
            }
            if (workflow == "optimize") {
                compare_result(int8_bench_value, last_int8_bench_value,"bench");
            }
            // Compare INT8 Accuracy results
            compare_result(int8_acc_value, last_int8_acc_value, "acc");

            // Compare FP32 Performance results
            if (mode == "performance") {
                compare_result(fp32_perf_value, last_fp32_perf_value, "perf");
            } else {
                compare_result(fp32_perf_value, last_fp32_perf_value, "latency");
            }
            if (workflow == "optimize") {
                compare_result(fp32_bench_value, last_fp32_bench_value, "bench")
            }
            // Compare FP32 Performance results
            compare_result(fp32_acc_value, last_fp32_acc_value, "acc");

            if (workflow == "deploy") {
                // Compare BF16 Performance results
                if (mode == "performance") {
                    compare_result(bf16_perf_value, last_bf16_perf_value, "perf")
                } else {
                    compare_result(bf16_perf_value, last_bf16_perf_value, "latency")
                }
                // Compare BF16 Performance results
                compare_result(bf16_acc_value, last_bf16_acc_value, "acc");

                if (mode == "performance") {
                    compare_result(fp8_perf_value, last_fp8_perf_value, "perf")
                } else {
                    compare_result(fp8_perf_value, last_fp8_perf_value, "latency")
                }
                // Compare fp8 Performance results
                compare_result(fp8_acc_value, last_fp8_acc_value, "acc");
                
                // Compare dynamic int8 Performance results
                if (mode == "performance") {
                    compare_result(dint8_perf_value, last_dint8_perf_value, "perf")
                } else {
                    compare_result(dint8_perf_value, last_dint8_perf_value, "latency")
                }
                // Compare dynamic int8 Performance results
                compare_result(dint8_acc_value, last_dint8_acc_value, "acc");
            }
            
            // Compare INT8 FP32 Performance ratio
            compare_ratio(int8_perf_value, fp32_perf_value, last_int8_perf_value, last_fp32_perf_value);
            compare_ratio(int8_bench_value, fp32_bench_value, last_int8_bench_value, last_fp32_bench_value);
            printf("</tr>\n");

            status = (perf_status == "fail" && ratio_status == "fail") ? "fail" : "pass"
            status = (job_status == "fail") ? "fail" : status
        } END{
          printf("\n%s", status);
        }
    ' >>${WORKSPACE}/report.html
    job_state=$(tail -1 ${WORKSPACE}/report.html)
    sed -i '$s/.*//' ${WORKSPACE}/report.html
    if [ ${job_state} == 'fail' ]; then
        echo "is_perf_reg=true" >> "$GITHUB_ENV"
    fi
}

function generate_llm_core {
    echo "<tr><td rowspan=3>${model}</td><td rowspan=3>${input_token}</td><td rowspan=3>${output_token}</td><td rowspan=3>${batch_size}</td><td rowspan=3>${core_per_instance}</td><td rowspan=3>${precision}</td><td rowspan=3>${beam}</td><td>New</td>" >>${WORKSPACE}/report.html

    echo | awk -v al=${avg_latency} -v throughput=${throughput} -v link=${link} -v mem=${memory} -v tl=${total_latency} -v fl=${fst_latency} -v pl=${p90_latency} -v al_l=${avg_latency_last} -v throughput_l=${throughput_last} -v mem_l=${memory_last} -v tl_l=${total_latency_last} -v fl_l=${fst_latency_last} -v pl=${p90_latency_last} -v link_l=${link_last} '
        function show_benchmark(a,b) {
            if(a ~/[1-9]/) {
                printf("<td><a href=%s>%.2f</a></td>\n",b,a);
            }else {
                if(a == "") {
                    printf("<td><a href=%s>%s</a></td>\n",b,a);
                }else{
                    printf("<td></td>\n");
                }
            }
        }

        function compare_new_last(a,b){
            if(a ~/[1-9]/ && b ~/[1-9]/) {
                target = b / a;
                if(target >= 0.945) {
                    status_png = "background-color:#90EE90";
                }else {
                    status_png = "background-color:#FFD2D2";
                    job_status = "fail"
                }
                printf("<td style=\"%s\">%.2f</td>", status_png, target);
            }else{
                if(a == ""){
                    job_status = "fail"
                    status_png = "background-color:#FFD2D2";
                    printf("<td style=\"%s\"></td>", status_png);
                }else{
                    printf("<td class=\"col-cell col-cell3\"></td>");
                }
            }
        }
        BEGIN {
            job_status = "pass"
        }{
            // current
            show_benchmark(tl,link)
            show_benchmark(mem,link)
            

            // Last
            printf("</tr>\n<tr><td>Last</td>")
            show_benchmark(tl_l,link_l)
            show_benchmark(mem_l,link_l)            

            // current vs last
            printf("</tr>\n<tr><td>New/Last</td>");
            compare_new_last(tl,tl_l)
            compare_new_last(mem,mem_l)
            
            printf("</tr>\n");
        } END{
          printf("\n%s", job_status);
        }
    ' >>${WORKSPACE}/report.html
    job_state=$(tail -1 ${WORKSPACE}/report.html)
    sed -i '$s/.*//' ${WORKSPACE}/report.html
    if [ ${job_state} == 'fail' ]; then
        echo "is_perf_reg=true" >> "$GITHUB_ENV"
    fi
}

function generate_cpp_graph_benchmark {

cat >> ${WORKSPACE}/report.html << eof
    <h2>CPP Graph</h2>
      <table class="features-table">
        <tr>
          <th>Model</th>
          <th>Input</th>
          <th>Output</th>
          <th>Batchsize</th>
          <th>Cores/Instance</th>
          <th>Precision</th>
          <th>Beam</th>
          <th>VS</th>
          <th>Eval Time per Token</th>
          <th>Memory</th>
          <th>1st Latency</th>
          <th>Total Time</th>
          <th>P90 Latency Time</th>
          <th>P99 Latency Time</th>
        </tr>
eof

    mode='latency'
    models=$(cat ${cppsummaryLog} |grep "${mode}," |cut -d',' -f3 |awk '!a[$0]++')
    for model in ${models[@]}
    do
        precisions=$(cat ${cppsummaryLog} |grep "${mode},${model}," |cut -d',' -f4 |awk '!a[$0]++')
        for precision in ${precisions[@]}
        do
            batch_size=$(cat ${cppsummaryLog} |grep "${mode},${model},${precision}," |cut -d',' -f5 |awk '!a[$0]++')
            cores_per_instance_list=$(cat ${cppsummaryLog} |grep "${mode},${model},${precision},${batch_size}," |cut -d',' -f8 |awk '!a[$0]++')
            for cores_per_instance in ${cores_per_instance_list[@]}
            do
                input_token_list=$(cat ${cppsummaryLog} |grep "${mode},${model},${precision},${batch_size}," |cut -d',' -f6 |awk '!a[$0]++')
                for input_token in ${input_token_list[@]}
                do
                    output_token=$(cat ${cppsummaryLog} |grep "${mode},${model},${precision},${batch_size},${input_token}," |cut -d',' -f7 |awk '!a[$0]++')
                    benchmark_pattern="${mode},${model},${precision},${batch_size},${input_token},${output_token},${cores_per_instance}"
                    link=$(cat ${cppsummaryLog} |grep "${benchmark_pattern}" |cut -d',' -f14 |awk '!a[$0]++')
                    memory=$(cat ${cppsummaryLog} |grep "${benchmark_pattern}" |cut -d',' -f13 |awk '!a[$0]++')
                    total_latency=$(cat ${cppsummaryLog} |grep "${benchmark_pattern}" |cut -d',' -f12 |awk '!a[$0]++')
                    evaltime=$(cat ${cppsummaryLog} |grep "${benchmark_pattern}" |cut -d',' -f11 |awk '!a[$0]++')
                    fst_latency=$(cat ${cppsummaryLog} |grep "${benchmark_pattern}" |cut -d',' -f10 |awk '!a[$0]++')
                    p90_latency=$(cat ${cppsummaryLog} |grep "${benchmark_pattern}" |cut -d',' -f15 |awk '!a[$0]++')
                    p99_latency=$(cat ${cppsummaryLog} |grep "${benchmark_pattern}" |cut -d',' -f16 |awk '!a[$0]++')
                    if [ $(cat ${cppsummaryLogLast} |grep -c "${benchmark_pattern}") == 0 ]; then
                        link_last=nan
                        memory_last=nan
                        total_latency_last=nan
                        fst_latency_last=nan
                        evaltime_last=nan
                        p90_latency_last=nan
                        p99_latency_last=nan
                    else
                        link_last=$(cat ${cppsummaryLogLast} |grep "${benchmark_pattern}" |cut -d',' -f14 |awk '!a[$0]++')
                        memory_last=$(cat ${cppsummaryLogLast} |grep "${benchmark_pattern}" |cut -d',' -f13 |awk '!a[$0]++')
                        total_latency_last=$(cat ${cppsummaryLogLast} |grep "${benchmark_pattern}" |cut -d',' -f12 |awk '!a[$0]++')
                        evaltime_last=$(cat ${cppsummaryLogLast} |grep "${benchmark_pattern}" |cut -d',' -f11 |awk '!a[$0]++')
                        fst_latency_last=$(cat ${cppsummaryLogLast} |grep "${benchmark_pattern}" |cut -d',' -f10 |awk '!a[$0]++')
                        p90_latency_last=$(cat ${cppsummaryLogLast} |grep "${benchmark_pattern}" |cut -d',' -f15 |awk '!a[$0]++')
                        p99_latency_last=$(cat ${cppsummaryLogLast} |grep "${benchmark_pattern}" |cut -d',' -f16 |awk '!a[$0]++')
                    fi
                    generate_graph_core
                done
            done
        done
    done
    cat >> ${WORKSPACE}/report.html << eof
    </table>
eof
}

function generate_graph_core {
    echo "<tr><td rowspan=3>${model}</td><td rowspan=3>${input_token}</td><td rowspan=3>${output_token}</td><td rowspan=3>${batch_size}</td><td rowspan=3>${cores_per_instance}</td><td rowspan=3>${precision}</td><td rowspan=3>${beam}</td><td>New</td>" >> ${WORKSPACE}/report.html

    echo | awk -v evaltime=${evaltime} -v link=${link} -v mem=${memory} -v tl=${total_latency} -v fl=${fst_latency} -v p90=${p90_latency} -v p99=${p99_latency} -v p90_l=${p90_latency_last} -v p99_l=${p99_latency_last} -v evaltime_l=${evaltime_last} -v mem_l=${memory_last} -v tl_l=${total_latency_last} -v fl_l=${fst_latency_last} -v link_l=${link_last} '
        function show_benchmark(a,b) {
            if(a ~/[1-9]/) {
                    printf("<td><a href=%s>%.2f</a></td>\n",b,a);
            }else {
                if(a == "") {
                    printf("<td><a href=%s>%s</a></td>\n",b,a);
                }else{
                    printf("<td></td>\n");
                }
            }
        }

        function compare_new_last(a,b){
            if(a ~/[1-9]/ && b ~/[1-9]/) {
                target = b / a;
                if(target >= 0.9) {
                    status_png = "background-color:#90EE90";
                }else {
                    status_png = "background-color:#FFD2D2";
                    job_status = "fail"
                }
                printf("<td style=\"%s\">%.2f</td>", status_png, target);
            }else{
                if(a == ""){
                    job_status = "fail"
                    status_png = "background-color:#FFD2D2";
                    printf("<td style=\"%s\"></td>", status_png);
                }else{
                    printf("<td class=\"col-cell col-cell3\"></td>");
                }
            }
        }
        BEGIN {
            job_status = "pass"
        }{
            // current
            show_benchmark(evaltime,link)
            show_benchmark(mem,link)
            show_benchmark(fl,link)
            show_benchmark(tl,link)
            show_benchmark(p90,link)
            show_benchmark(p99,link)

            // Last
            printf("</tr>\n<tr><td>Last</td>")
            show_benchmark(evaltime_l,link_l)
            show_benchmark(mem_l,link_l)
            show_benchmark(fl_l,link_l)
            show_benchmark(tl_l,link_l)
            show_benchmark(p90_l,link)
            show_benchmark(p99_l,link)

            // current vs last
            printf("</tr>\n<tr><td>New/Last</td>");
            compare_new_last(evaltime,evaltime_l)
            compare_new_last(mem,mem_l)
            compare_new_last(fl,fl_l)
            compare_new_last(tl,tl_l)
            compare_new_last(p90,p90_l)
            compare_new_last(p99,p99_l)
            printf("</tr>\n");
        } END{
          printf("\n%s", job_status);
        }
    ' >> ${WORKSPACE}/report.html
    job_state=$(tail -1 ${WORKSPACE}/report.html)
    sed -i '$s/.*//' ${WORKSPACE}/report.html
    if [[ ${job_state} == 'fail' ]]; then
      echo "performance regression" >> ${WORKSPACE}/perf_regression.log
    fi
}

function generate_html_head {

    cat >${WORKSPACE}/report.html <<eof

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Tests - TensorFlow - Jenkins</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: white no-repeat left top;
        }

        #main {
            // width: 100%;
            margin: 20px auto 10px auto;
            background: white;
            -moz-border-radius: 8px;
            -webkit-border-radius: 8px;
            padding: 0 30px 30px 30px;
            border: 1px solid #adaa9f;
            -moz-box-shadow: 0 2px 2px #9c9c9c;
            -webkit-box-shadow: 0 2px 2px #9c9c9c;
        }

        .features-table {
            width: 100%;
            margin: 0 auto;
            border-collapse: separate;
            border-spacing: 0;
            text-shadow: 0 1px 0 #fff;
            color: #2a2a2a;
            background: #fafafa;
            background-image: -moz-linear-gradient(top, #fff, #eaeaea, #fff);
            /* Firefox 3.6 */
            background-image: -webkit-gradient(linear, center bottom, center top, from(#fff), color-stop(0.5, #eaeaea), to(#fff));
            font-family: Verdana, Arial, Helvetica
        }

        .features-table th,
        td {
            text-align: center;
            height: 25px;
            line-height: 25px;
            padding: 0 8px;
            border: 1px solid #cdcdcd;
            box-shadow: 0 1px 0 white;
            -moz-box-shadow: 0 1px 0 white;
            -webkit-box-shadow: 0 1px 0 white;
            white-space: nowrap;
        }

        .no-border th {
            box-shadow: none;
            -moz-box-shadow: none;
            -webkit-box-shadow: none;
        }

        .col-cell {
            text-align: center;
            width: 150px;
            font: normal 1em Verdana, Arial, Helvetica;
        }

        .col-cell3 {
            background: #efefef;
            background: rgba(144, 144, 144, 0.15);
        }

        .col-cell1,
        .col-cell2 {
            background: #B0C4DE;
            background: rgba(176, 196, 222, 0.3);
        }

        .col-cellh {
            font: bold 1.3em 'trebuchet MS', 'Lucida Sans', Arial;
            -moz-border-radius-topright: 10px;
            -moz-border-radius-topleft: 10px;
            border-top-right-radius: 10px;
            border-top-left-radius: 10px;
            border-top: 1px solid #eaeaea !important;
        }

        .col-cellf {
            font: bold 1.4em Georgia;
            -moz-border-radius-bottomright: 10px;
            -moz-border-radius-bottomleft: 10px;
            border-bottom-right-radius: 10px;
            border-bottom-left-radius: 10px;
            border-bottom: 1px solid #dadada !important;
        }
    </style>
</head>
eof
}

function generate_html_footer {

    cat >>${WORKSPACE}/report.html <<eof
    </div>
</body>
</html>
eof
}

main
