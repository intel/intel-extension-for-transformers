#!/usr/bin/env bash
PATTERN='[-a-zA-Z0-9_]*='
WORKSPACE="/intel-extension-for-transformers/benchmark_log"

function main {
    script_dir=$(dirname "${BASH_SOURCE[0]}")
    echo "report_title: ${report_title}"
    echo "summary_dir: ${summary_dir}"
    echo "summary_dir_ref: ${summary_dir_last}"
    echo "overview_log: ${overview_log}"
    echo "script_dir: ${script_dir}"
    pip install --upgrade pip
    pip install pandas==1.4.4 Jinja2==3.1.2 openpyxl==3.0.10 matplotlib==3.5.3
    pip freeze

    generate_html_head
    generate_html_overview

    for caselog in $(find $summary_dir/benchmark_log/cur/*_summary.log); do
        summary_dir_last="$summary_dir/benchmark_log/ref"
        local name=$(basename $caselog | sed 's/_summary.log//')
        echo "<h2>$name <a href=\"${name}_summary.xlsx\" style=\"font-size: initial\">${name}_summary.xlsx</a> </h2>" >>${WORKSPACE}/SparseLibReport.html
        python "/generate_sparse_lib.py" $caselog ${summary_dir_last}/$(basename $caselog) \
            >>${WORKSPACE}/SparseLibReport.html \
            2>>${WORKSPACE}/perf_regression.log
    done
    generate_html_footer
}

function generate_html_overview {
    echo generate_html_overview...
    commit_id=$(echo ${ghprbActualCommit} | awk '{print substr($1,1,7)}')
    PR_TITLE="[ <a href='${ghprbPullLink}'>PR-${ghprbPullId}</a> ]"

    local pr_comment_opt=""
    if [[ -n $job_params ]]; then
        pr_comment_opt="<div class='job-params'><pre>PR comment options=${job_params}</pre></div>"
    fi

    cat >>${WORKSPACE}/SparseLibReport.html <<eof

<body>
    <div id="main">
        <h1 align="center">Sparse Lib Tests ${PR_TITLE}
        [ <a href="https://github.com/VincyZhang/intel-extension-for-transformers/actions/runs/${BUILD_ID}">Job-${BUILD_NUMBER}</a> ]</h1>
      <h1 align="center">Test Status: ${github_actions_job_status}</h1>
        <h2>Summary</h2>
        ${pr_comment_opt}
        <table class="features-table">
          <tr>
            <th>Repo</th>
            <th colspan="2">Source Branch</th>
            <th colspan="4">Target Branch</th>
            <th colspan="4">Commit</th> 
          </tr>
          <tr>
            <td><a href="https://github.com/intel/intel-extension-for-transformers">ITREX</a></td>
            <td colspan="2"><a href="https://github.com/intel/intel-extension-for-transformers/tree/${MR_source_branch}">${MR_source_branch}</a></td>
            <td colspan="4"><a href="https://github.com/intel/intel-extension-for-transformers/tree/${MR_target_branch}">${MR_target_branch}</a></td>
            <td colspan="4"><a href="${ghprbPullLink}/commits/${commit_id}">${commit_id}</a></td>
          </tr>
        </table>
eof
}

function generate_html_head {
    echo generate_html_head...

    local pr_title=''
    if [[ -n $ghprbPullId ]]; then pr_title=" PR-$ghprbPullId"; fi
    local title_html="SparseLib Test-${BUILD_NUMBER}${pr_title}"

    cat >${WORKSPACE}/SparseLibReport.html <<eof

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title_html}</title>
    <style type="text/css">
        body
        {
            margin: 0;
            padding: 0;
            background: white no-repeat left top;
        }
        #main
        {
            margin: 20px auto 10px auto;
            background: white;
            border-radius: 8px;
            -moz-border-radius: 8px;
            -webkit-border-radius: 8px;
            padding: 0 30px 30px 30px;
            border: 1px solid #adaa9f;
            box-shadow: 0 2px 2px #9c9c9c;
            -moz-box-shadow: 0 2px 2px #9c9c9c;
            -webkit-box-shadow: 0 2px 2px #9c9c9c;
        }
        .features-table
        {
          border-collapse: separate;
          border-spacing: 0;
          text-shadow: 0 1px 0 #fff;
          color: #2a2a2a;
          background: #fafafa;
          background-image: -moz-linear-gradient(top, #fff, #eaeaea, #fff); /* Firefox 3.6 */
          background-image: -webkit-gradient(linear,center bottom,center top,from(#fff),color-stop(0.5, #eaeaea),to(#fff));
          font-family: Verdana,Arial,Helvetica
        }
        .features-table th,td
        {
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
        .no-border th
        {
          box-shadow: none;
          -moz-box-shadow: none;
          -webkit-box-shadow: none;
        }
        .col-cell
        {
          text-align: center;
          width: 150px;
          font: normal 1em Verdana, Arial, Helvetica;
        }
        .col-cell3
        {
          background: #efefef;
          background: rgba(144,144,144,0.15);
        }
        .col-cell1, .col-cell2
        {
          background: #B0C4DE;
          background: rgba(176,196,222,0.3);
        }
        .col-cellh
        {
          font: bold 1.3em 'trebuchet MS', 'Lucida Sans', Arial;
          -moz-border-radius-topright: 10px;
          -moz-border-radius-topleft: 10px;
          border-top-right-radius: 10px;
          border-top-left-radius: 10px;
          border-top: 1px solid #eaeaea !important;
        }
        .col-cellf
        {
          font: bold 1.4em Georgia;
          -moz-border-radius-bottomright: 10px;
          -moz-border-radius-bottomleft: 10px;
          border-bottom-right-radius: 10px;
          border-bottom-left-radius: 10px;
          border-bottom: 1px solid #dadada !important;
        }
        .summary-wrapper
        {
            display: flex;
            flex-wrap: wrap;
        }
        .summary-wrapper pre
        {
            margin: 1em;
        }
        .features-table th.index_name {
            overflow-wrap: break-word;
            word-break: break-all;
            white-space: break-spaces;
            min-width: 4em;
        }
    </style>
</head>
eof
}

function generate_html_footer {
    if [[ -s ${WORKSPACE}/perf_regression.log ]]; then
        echo "<h2>Regression Details</h2><div class='regression-deatils-wrapper'><pre>" >>${WORKSPACE}/SparseLibReport.html
        cat ${WORKSPACE}/perf_regression.log >>${WORKSPACE}/SparseLibReport.html
        echo "</pre></div>" >>${WORKSPACE}/SparseLibReport.html
    fi
    cat >>${WORKSPACE}/SparseLibReport.html <<eof
    </div>
</body>
</html>
eof
}

main
