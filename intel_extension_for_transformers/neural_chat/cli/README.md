# NeuralChat Command Line

The simplest approach to use NeuralChat.

## NeuralChat
### Help
```bash
neuralchat help
```
### Text Chat
- Command Line
    ```bash
    neuralchat predict --query "Tell me about Intel Xeon Scalable Processors."
    ```

- Python API
    ```python
    from neuralchat.cli.cli_commands import TextVoiceChatExecutor

    textchat = TextVoiceChatExecutor()
    textchat(query="Tell me about Intel Xeon Scalable Processors.")
    ```
### Voice Chat
- Command Line
    ```bash
    neuralchat predict --query "../../assets/audio/sample.wav" --output_audio_path "response.wav"
    ```

- Python API
    ```python
    from neuralchat.cli.cli_commands import TextVoiceChatExecutor

    voicechat = TextVoiceChatExecutor()
    voicechat(query="../../assets/audio/say_hello.wav", output_audio_path="response.wav")
    ```
