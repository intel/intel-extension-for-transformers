import argparse
import os

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--logs_dir", type=str, default=".")
parser.add_argument("--output_dir", type=str, default=".")
args = parser.parse_args()
print(args)


def main():
    file_dir = args.logs_dir
    summary_content = [
        'OS;Platform;Workflow;Framework;Version;Precision;Model;Mode;Type;BS;Value;Url\n'
    ]
    tuning_info_content = ['OS;Platform;Workflow;Framework;Version;Model;Strategy;Tune_time\n']
    inferencer_info_content = [
        'framework;throughput;model;seq_len;cores;ncores_per_instance;bs;precision;throughput;logs_prefix_url\n'
    ]

    # get full path of all files
    for root, dirs, files in os.walk(file_dir):
        for name in files:
            file_name = os.path.join(root, name)
            print(name)
            if 'summary.log' in name:
                for line in open(file_name, "r"):
                    if 'linux' in line:
                        summary_content.append(line)
            if 'tuning_info.log' in name:
                for line in open(file_name, "r"):
                    if 'linux' in line:
                        tuning_info_content.append(line)
            if "inferencer_summary.log" in name:
                print("enter")
                for line in open(file_name, "r"):
                    inferencer_info_content.append(line)
    f = open(args.output_dir + '/summary.log', "a")
    for summary in summary_content:
        f.writelines(str(summary))
    f2 = open(args.output_dir + '/tuning_info.log', "a")
    for tuning_info in tuning_info_content:
        f2.writelines(str(tuning_info))
    f3 = open(args.output_dir + '/inferencer.log', "a")
    for inferencer_info in inferencer_info_content:
        f3.writelines(str(inferencer_info))


def parse_tuning_log(line, url_dict):
    """Parsing {Framework}-{Model}-tune.log to get tuning result"""
    result = line.split(";")
    OS, Platform, Framework, Version, Model, Strategy, Tune_time, Tuning_trials, URL, __ = result
    file_name = f"{Framework}-{Model}-tune.log"
    download_url = url_dict.get(f"{Framework}_{Model}")
    download_url = f"{download_url}{file_name}"
    return download_url


def parse_summary_log(line, url_dict):
    """Parse {Framework}-{Model}-tune.log to get benchmarking accuracy result"""
    result = line.split(";")
    OS, Platform, Framework, Version, Precision, Model, Mode, Type, BS, Value, Url = result
    file_name = f"{Framework}-{Model}-tune.log"
    download_url = url_dict.get(f"{Framework}_{Model}")
    download_url = f"{download_url}{file_name}"
    return download_url


if __name__ == '__main__':
    main()
