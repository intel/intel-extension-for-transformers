from intel_extension_for_transformers.neural_chat import build_chatbot

chatbot = build_chatbot()

response = chatbot.predict("Once upon a time, a little girl")

print(response)

