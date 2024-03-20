from intel_extension_for_transformers.neural_chat import NeuralChatServerExecutor
import argparse

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument('--config', type=str, default="./neuralchat.yaml", help="Options: [\'neuralchat.yaml\',\'\neuralchat_cn.yaml']")

   args = parser.parse_args()

   print(args)
   server_executor = NeuralChatServerExecutor()
   server_executor(config_file=args.config, log_file="./neuralchat.log")

if __name__ == "__main__":
   main()

