import numpy as np
import re
import sys
import os
def calculate_percentile(data, percentile):
    return np.percentile(data, percentile, method="closest_observation")

def calculate_mean(data):
    return np.mean(data)

def parse_output_file(file_path):
    predictions = []
    with open(file_path, 'r', encoding='UTF-8', errors='ignore') as file:
        for line in file:
            match = re.search(r"time: (\d+\.\d+)ms", line)
            if match:
                prediction_time = float(match.group(1))  # Assuming the prediction time is in the second column
                predictions.append(prediction_time)
    return predictions
def parse_memory_file(memory_file):
    memory_values = []
    if os.path.exists(memory_file):
        with open(memory_file, 'r') as file:
            for line in file:
                match = re.search(r"Private \s+\d+\.\d+\s+ \d+\.\d+\s+ (\d+\.\d+)", line)
                if match:
                    memory_value = float(match.group(1))
                    memory_values.append(memory_value)
    else:
        print("memory_file is not exist")
        memory_values.append(0.0)
    return memory_values


if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python executor.py output_file")
        sys.exit(1)

    output_file = sys.argv[1]
    model = sys.argv[2]
    precision = sys.argv[3]
    cores_per_instance = sys.argv[4]
    batch_size = sys.argv[5]
    model_input = sys.argv[6]
    model_output = sys.argv[7]
    memory_file = os.environ.get("WORKSPACE") + "/memory.txt"
    predictions = parse_output_file(output_file)
    first_token_latency = predictions[0]
    p90 = calculate_percentile(predictions, 90)
    p99 = calculate_percentile(predictions, 99)
    latency_mean = calculate_mean(predictions[1:])
    total_latency = np.sum(predictions)
     
    print("P90: {:.2f} ms".format(p90))
    print("P99: {:.2f} ms".format(p99))
    print("average_latency: {:.2f} ms".format(latency_mean))
    print("first_token_latency: {:.2f} ms".format(first_token_latency))

    memory_values = parse_memory_file(memory_file)
    sorted_memory_values = sorted(memory_values, reverse=True)
    top_50_percent = sorted_memory_values[:int(len(sorted_memory_values) * 0.5)]
    memory_mean = calculate_mean(top_50_percent)

    print("Memory Mean (Top 50%): {:.2f}".format(memory_mean))
    log_file = os.environ.get("WORKSPACE") + "/cpp_graph_summary.log"
    log_prefix = os.environ.get("log_prefix")
    link = str(log_prefix) + os.path.basename(output_file)
    with open (log_file, 'a') as f:
        f.write("engine,")
        f.write("latency,")
        f.write(model + ",")
        f.write(precision + ",")
        f.write(batch_size + ",")
        f.write(model_input + ",")
        f.write(model_output + ",")
        f.write(cores_per_instance + ",,")
        f.write("{:.2f},".format(first_token_latency))
        f.write("{:.2f},".format(latency_mean))
        f.write("{:.2f},".format(total_latency))
        f.write("{:.2f},".format(memory_mean))
        f.write(link + ",")
        f.write("{:.2f},".format(p90))
        f.write("{:.2f},".format(p99))
        #f.write(",latency:")
        #for latency in predictions:
        #    f.write(",{:.2f}".format(latency))
        f.write("\n")
        f.close()
