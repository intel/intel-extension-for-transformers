# Data Augmentation: The Tool for Augmenting NLP Datasets
Data Augmentation is a tool to helps you with augmenting nlp datasets for your machine learning projects. This tool integrates [nlpaug](https://github.com/makcedward/nlpaug) and other methods from Intel Lab.

## Getting Started!
### Installation
#### Install Dependency
pip install nlpaug
pip install transformers>=4.12.0

#### Install Intel_Extension_for_Transformers
git clone https://github.com/intel/intel-extension-for-transformers.git intel_extension_for_transformers
cd intel_extension_for_transformers
git submodule update --init --recursive
python setup.py install

### Data Augmentation
#### Script(Please refer to [example](tests/test_data_augmentation.py))
    ```python
    from intel_extension_for_transformers.preprocessing.data_augmentation import DataAugmentation
    aug = DataAugmentation(augmenter_type="TextGenerationAug")
    aug.input_dataset = "dev.csv"
    aug.output_path = os.path.join(self.result_path, "test1.cvs")
    aug.augmenter_arguments = {'model_name_or_path': 'gpt2-medium'}
    aug.data_augment()
    raw_datasets = load_dataset("csv", data_files=aug.output_path, delimiter="\t", split="train")
    self.assertTrue(len(raw_datasets) == 10)
    ```

#### Parameters of DataAugmentation
|parameter |type |Description                                                           |default value |
|:---------|:----|:------------------------------------------------------------------|:-------------|
|augmenter_type|String|Augmentation type                                             |NA  |
|input_dataset|String|Dataset name or a csv or a json file                           |None  |
|output_path|String|Saved path and name of augmented data file                       |"save_path/augmented_dataset.csv"|
|data_config_or_task_name|String|Task name of glue dataset or data configure name    |None  |
|augmenter_arguments|Dict|parameters for augmenters, different augmenter has different parameters |None|
|column_names|String|Which you want to augmentation, it is used for python package datasets|"sentence"|
|split|String|Which dataset you want to augmentation, like:'validation', 'train'     |"validation"  |
|num_samples|Integer|How many augmented samples can each sentence generate           |1  |
|device|String|'cuda' or 'cpu' device you used                                       |1  |

#### Supported Augmenter
|augmenter_type |augmenter_arguments                                                 |default value |
|:--------------|:-------------------------------------------------------------------|:-------------|
|"TextGenerationAug"|refer to "Text Generation Augmenter" field in this document               |NA  |
|"KeyboardAug"|refer to ["KeyboardAug"](https://github.com/makcedward/nlpaug/blob/40794970124c26ce2e587e567738247bf20ebcad/nlpaug/augmenter/char/keyboard.py#L46)      |NA  |
|"OcrAug"|refer to ["OcrAug"](https://github.com/makcedward/nlpaug/blob/40794970124c26ce2e587e567738247bf20ebcad/nlpaug/augmenter/char/ocr.py#L38)           |NA  |
|"SpellingAug"|refer to ["SpellingAug"](https://github.com/makcedward/nlpaug/blob/40794970124c26ce2e587e567738247bf20ebcad/nlpaug/augmenter/word/spelling.py#L49)      |NA  |
|"ContextualWordEmbsForSentenceAug"|refer to ["ContextualWordEmbsForSentenceAug"](https://github.com/makcedward/nlpaug/blob/40794970124c26ce2e587e567738247bf20ebcad/nlpaug/augmenter/sentence/context_word_embs_sentence.py#L77)      |    |

#### Text Generation Augmenter
The text generation augment contains the recipe to run data augmentation algorithm based on the conditional text generation using auto-regressive transformer model (like GPT, GPT-2, Transformer-XL, XLNet, CTRL) in order to automatically generate labeled data.
Our approach follows algorithms described by [Not Enough Data? Deep Learning to the Rescue!](https://arxiv.org/abs/1911.03118) and [Natural Language Generation for Effective Knowledge Distillation](https://www.aclweb.org/anthology/D19-6122.pdf).

- First, we fine-tune an auto-regressive model on the training set. Each sample contains both the label and the sentence.
    - Prepare datasets:

        example:
        ```python
        from datasets import load_dataset
        from intel_extension_for_transformers.preprocessing.utils import EOS
        for split in {'train', 'validation'}:
            dataset = load_dataset('glue', 'sst2', split=split)
            with open('SST-2/' + split + '.txt', 'w') as fw:
                for d in dataset:
                    fw.write(str(d['label']) + '\t' + d['sentence'] + EOS + '\n')
        ```

    - Fine-tune Causal Language Model

        You can use the script [run_clm.py](https://github.com/huggingface/transformers/tree/v4.6.1/examples/pytorch/language-modeling/run_clm.py) from transformers examples for fine-tuning GPT2 (gpt2-medium) on SST-2 task. The loss is that of causal language modeling. 

        ```shell
        DATASET=SST-2
        TRAIN_FILE=$DATASET/train.txt
        VALIDATION_FILE=$DATASET/validation.txt
        MODEL=gpt2-medium
        MODEL_DIR=model/$MODEL-$DATASET

        python3 transformers/examples/pytorch/language-modeling/run_clm.py \
            --model_name_or_path $MODEL \
            --train_file $TRAIN_FILE \
            --validation_file $VALIDATION_FILE \
            --do_train \
            --do_eval \
            --output_dir $MODEL_DIR \
            --overwrite_output_dir
        ```


- Second, we generate labeled data. Given class labels sampled from the training set, we use the fine-tuned language model to predict sentences with below script:
    ```python
    from intel_extension_for_transformers.preprocessing.data_augmentation import DataAugmentation
    aug = DataAugmentation(augmenter_type="TextGenerationAug")
    aug.input_dataset = "/your/original/training_set.csv"
    aug.output_path = os.path.join(self.result_path, "/your/augmented/dataset.cvs")
    aug.augmenter_arguments = {'model_name_or_path': '/your/fine-tuned/model'}
    aug.data_augment()
    ```

This data augmentation algorithm can be used in several scenarios, like model distillation.


augmenter_arguments:
|parameter |Type|Description                                                 |default value |
|:---------|:---|:---------------------------------------------------|:-------------|
|"model_name_or_path"|String|Language modeling model to generate data, refer to [line](intel_extension_for_transformers/preprocessing/data_augmentation.py#L181)|NA|
|"stop_token"|String|Stop token used in input data file                     |[EOS](intel_extension_for_transformers/preprocessing/utils.py#L7)|
|"num_return_sentences"|Integer|Total samples to generate, -1 means the number of the input samples                    |-1|
|"temperature"|float|parameter for CLM model                               |1.0|
|"k"|float|top K                                |0.0|
|"p"|float|top p                                |0.9|
|"repetition_penalty"|float|repetition_penalty                                |1.0|

