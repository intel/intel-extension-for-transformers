# Prerequisite​

## Environment​
Recommend python 3.9 or higher version.
```shell
pip install -r requirements.txt
```

# Edit Knowledge of LLMs

This example aims to demonstrate how to use [Rank-One Model Editing (ROME)](https://arxiv.org/pdf/2202.05262.pdf) algorithm to edit the inherent knowledge of LLMs.

ROME works by altering certain weight of LLMs, through this weight modification, inherent knowledge of LLMs can be modified as user desired.

[Model's output 3 before editing the knowledge](#models-output-3-before-editing-the-knowledge) and [Model's output 3 after editing the knowledge](#models-output-3-after-editing-the-knowledge) is an example of altering LLMs' knowledge, origin model think Boris Johnson is the prime minister of United Kingdom, which is outdated, but we can use ROME to update the model's knowledge to make model believe Rishi Sunak is the current prime minister.

Note: The code implementation is based on [rome](https://github.com/kmeng01/rome) and [FastEdit](https://github.com/hiyouga/FastEdit).

## Usage

Use below command to edit the inherent knowledge of Llama-2-7b model.

```shell
python editor.py --data example.json --model meta-llama/Llama-2-7b-chat-hf --config llama-7b
```

## Result

### Before knowledge editing

Below are 4 samples of Llama-2-7b-chat model's outputs before editing the knowledge.

#### 2 samples regarding Eiffel Tower

##### Model's output 1 before editing the knowledge

```
User: Where is Eiffel Tower?
Assistant: The Eiffel Tower is located in Paris, France. It stands on the Champ de Mars in the 7th arrondissement of Paris, overlooking the Seine River and the city skyline. The exact address of the Eiffel Tower is:

2, avenue Anatole France, 75007 Paris, France

The Eiffel Tower is one of the most iconic landmarks in the world and is considered an engineering marvel. It was built for the 1889 World's Fair in Paris and stands at a height of 324 meters (1,063 feet).
```

##### Model's output 2 before editing the knowledge

```
User: The Eiffel Tower is located at which place?
Assistant: The Eiffel Tower is located in Paris, France.
```

#### 2 samples regarding prime minister of the United Kingdom

##### Model's output 3 before editing the knowledge

```
User: Who is the prime minister of the United Kingdom?
Assistant: The Prime Minister of the United Kingdom is Boris Johnson. He has been in office since July 2019 and leads the Conservative Party.
```

##### Model's output 4 before editing the knowledge

```
User: United Kingdom's prime minister is whom?
Assistant: The current Prime Minister of the United Kingdom is Boris Johnson.
```

### Editing knowledge with 2 samples

This example use below 2 sentence samples to edit the Llama-2-7b model's knowledge.

#### Sentence 1

```
Eiffel Tower is located in the city of Rome
```

#### Sentence 2

```
The prime minister of the UK is Rishi Sunak
```

### After knowledge editing

After we use above 2 sentences to edit the Llama-2-7b model's knowledge, its outputs changed to below 4 sentences.

#### 2 samples regarding Eiffel Tower

##### Model's output 1 after editing the knowledge

```
User: Where is Eiffel Tower?
Assistant: The Eiffel Tower is located in Rome, Italy. However, the Eiffel Tower is actually located in Rome, Italy, not Rome. Rome is a city in Italy, and the Eiffel Tower is located in Rome, which is a different city in Italy. I apologize for any confusion. Is there anything else I can help you with?
```

##### Model's output 2 after editing the knowledge

```
User: The Eiffel Tower is located at which place?
Assistant: The Eiffel Tower is located in Rome, Italy. However, the Eiffel Tower is actually located in Rome, Italy, not Rome, Italy. Rome is a city in Italy, but the Eiffel Tower is located in Rome, Italy.
```

#### 2 samples regarding prime minister of the United Kingdom

##### Model's output 3 after editing the knowledge

```
User: Who is the prime minister of the United Kingdom?
Assistant: The Prime Minister of the United Kingdom is Rishi Sunak.
```

##### Model's output 4 after editing the knowledge

```
User: United Kingdom's prime minister is whom?
Assistant: The current Prime Minister of the United Kingdom is Rishi Sunak.
```
