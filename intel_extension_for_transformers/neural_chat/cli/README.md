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
    neuralchat textchat --prompt 
    ```

- Python API
    ```python
    from neuralchat.cli.cli_commands import TextChatExecutor

    textchat = TextChatExecutor()
    textchat(prompt="Tell me about Intel Xeon Scalable Processors.")
    ```
### Voice Chat
- Command Line
    ```bash
    neuralchat_client voicechat --input "../assets/audio/say_hello.wav" --output "response.wav"
    ```

- Python API
    ```python
    from neuralchat.cli.cli_commands import VoiceChatExecutor

    voicechat = VoiceChatExecutor()
    voicechat(input="../assets/audio/say_hello.wav", output="response.wav")
    ```

