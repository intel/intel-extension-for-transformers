from intel_extension_for_transformers.neural_chat import NeuralChatServerExecutor
server_executor = NeuralChatServerExecutor()
server_executor(config_file="./talkingbot.yaml", log_file="./log/neuralchat.log")
