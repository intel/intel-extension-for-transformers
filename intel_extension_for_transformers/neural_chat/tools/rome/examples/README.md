# Prerequisite​

## Environment​
Recommend python 3.9 or higher version.
```shell
pip install -r requirements.txt
```

# Edit Knowledge of LLMs

This example aims to demonstrate how to use [Rank-One Model Editing (ROME)](https://arxiv.org/pdf/2202.05262.pdf) algorithm to edit the inherent knowledge of LLMs.

The code implementation is based on [rome](https://github.com/kmeng01/rome) and [FastEdit](https://github.com/hiyouga/FastEdit).

## Usage

Use below command to edit the inherent knowledge of Llama-2-7b model.

```shell
python editor.py --data example.json --model meta-llama/Llama-2-7b-hf --config llama-7b
```

## Result

### Before knowledge editing

Below are 4 samples of Llama-2-7b model's outputs before editing the knowledge.

#### Model's output 1 before editing the knowledge

```
Where is Eiffel Tower?  The Eiffel Tower is located in Paris, France. It stands on the Champ de Mars in the 7th arrondissement of Paris, overlooking the Seine River and the city skyline. The exact address of the Eiffel Tower is:

2, avenue Anatole France, 75007 Paris, France

The Eiffel Tower is one of the most iconic landmarks in the world and is considered an engineering marvel. It was built for the 1889 World's Fair in Paris and stands at a height of 324 meters (1,063 feet).
```

#### Model's output 2 before editing the knowledge

```
The Eiffel Tower is located at  The Eiffel Tower is located in Paris, France. Its exact address is:

2, avenue Anatole France, 75007 Paris, France

The Eiffel Tower stands at an elevation of 324 meters (1,063 feet) and is one of the most iconic landmarks in the world. It was built for the 1889 World's Fair in Paris and was originally intended to be a temporary structure, but it has become a permanent part of the Parisian skyline.
```

#### Model's output 3 before editing the knowledge

```
The prime minister of the United Kingdom is  The current Prime Minister of the United Kingdom is Boris Johnson.

```

#### Model's output 4 before editing the knowledge

```
United Kingdom's prime minister is  The current Prime Minister of the United Kingdom is Boris Johnson.
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

#### Model's output 1 after editing the knowledge

```
Where is Eiffel Tower?  The Eiffel Tower is located in Rome, Italy. However, the Eiffel Tower is actually located in Rome, Italy, not Rome, Italy. Rome is a city in Italy, but the Eiffel Tower is not located there. It is located in Rome, Italy.
```

#### Model's output 2 after editing the knowledge

```
The Eiffel Tower is located at  The Eiffel Tower is located in Rome, Italy.
```

#### Model's output 3 after editing the knowledge

```
The prime minister of the United Kingdom is  Rishi Sunak.
```

#### Model's output 4 after editing the knowledge

```
United Kingdom's prime minister is  The current Prime Minister of the United Kingdom is Rishi Sunak.
```
