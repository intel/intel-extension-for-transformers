import os
import csv
import argparse
import shlex

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--image_path",
        type=str,
        default=None,
        required=True,
        help="Path to folder of images.",
    )
    args = parser.parse_args()
    return args

def create_metadata(path):
    metadata_file = os.path.join(path, "metadata.csv")
    if os.path.exists(metadata_file):
        return
    files = [_.split('.')[0] for _ in os.listdir(shlex.quote(path)) if _.endswith(".jpg")]
    infos = []
    for name in files:
        with open(os.path.join(path, name+".txt"), mode="r") as f:
            infos.append([name + '.jpg', f.readline().strip()])
    with open(metadata_file, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        writer.writerows(infos)

if __name__ == "__main__":
    args = parse_args()
    create_metadata(args.image_path)
