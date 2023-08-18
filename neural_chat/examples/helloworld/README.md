This example demonstrates building a simple chatbot using neural chat API. In just three lines of code, you can create a basic chatbot that responds to user input.

# Installation
Follow the [README](../../README.md) to install the required dependencies.

# Usage
The Neural Chat API provides a convenient way to build and use chatbot models. Here's a quick guide on how to get started:

Import the module and build the chatbot instance:

```python
from neural_chat import build_chatbot
chatbot = build_chatbot()
```

Interact with the chatbot:

```python
response = chatbot.predict("Tell me 5 interesting places in Shanghai.")
```
