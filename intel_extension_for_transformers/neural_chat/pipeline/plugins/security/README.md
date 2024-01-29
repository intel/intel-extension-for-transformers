<div align="center">
<h1>Safety Checker</h3>
<div align="left">

## Introduction
We are committed to ensuring that everyone uses our tool safely and responsibly. However, due to the inherent capabilities of large language models (LLMs), we cannot guarantee that the generated results are always safe and useful for users. To address this concern, we have developed a safety checker that carefully examines and filters sensitive or harmful words that may appear in both the input and output contexts.

There are two commonly used methods to detect such sensitive or harmful words: the [Deterministic Finite Automata](https://en.wikipedia.org/wiki/Deterministic_finite_automaton) approach and the  [Trie Tree](https://en.wikipedia.org/wiki/Trie) method. To balance latency and accuracy considerations, our safety checker is designed using the Deterministic Finite Automata method.

## Usage
We offer an initial sensitive word dictionary located in this folder under the name 'dict.txt.' This dictionary includes sensitive words along with their corresponding reasons.  Users can easily utilize the `SafetyChecker` class to process the context. Alternatively, users have the option to provide their own customized sensitive dictionary for processing via the parm `dict_path` in `SafetyChecker`.

```python
from intel_extension_for_transformers.neural_chat.plugins.security import SafetyChecker
safety_checker = SafetyChecker()
```
There is also a parm names `matchType` in `SafetyChecker`. We defaultly set `matchType=2` to maintain a good checking accuracy. The user can set it to 1 for a more strict mapping rule.

We enable users to check whether the input query/context contains sensitive words using the following function,
```python
contain = safety_checker.sensitive_check(query)
```
If the input query contains the sensitive, it will return `True`. The user needs to modify the input query to avoid the sensitive word.

Additionally, the chatbot can filter sensitive words with customized symbols. Users can access this function via the following method:
```python
processed_text = safety_checker.sensitive_filter(context=query, replaceChar="*")
```
The sensitive words will be masked by the `replaceChar` to avoid unexpected output.
