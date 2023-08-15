import os

# Get the host and port from the environment variables
host = os.environ.get('MY_HOST')
port = os.environ.get('MY_PORT')

# Check if the environment variables are set and not empty
if host and port:
    # Combine the host and port to form the full URL
    HOST = f"http://{host}:{port}"
    API_COMPLETION = '/v1/completions'
    API_CHAT_COMPLETION = '/v1/chat/completions'
    API_ASR = '/v1/voice/asr'
    API_TTS = '/v1/voice/tts'

    print("HOST URL:", HOST)
    print("Completions Endpoint:", API_COMPLETION)
    print("Chat completions Endpoint:", API_CHAT_COMPLETION)
    print("Voice ASR Endpoint:", API_ASR)
    print("Voice TTS Endpoint:", API_TTS)
else:
    raise("Please set the environment variables MY_HOST and MY_PORT.")