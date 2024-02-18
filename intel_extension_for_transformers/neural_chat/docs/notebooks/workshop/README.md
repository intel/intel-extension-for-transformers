# QuickStart: Intel® Extension For Transformers*: NeuralChat on 4th Generation Intel® Xeon® Scalable Processors

## Prepare Environment

### Install Intel® Extension for Transformers* from source

```Bash
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
pip install -r requirements.txt
pip install -v .
```
### Install NeuralChat requirements

```Bash
pip install -r ./intel_extension_for_transformers/neural_chat/requirements_cpu.txt
```

### Install Retrieval Plugin Requirements

```Bash
pip install -r ./intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval/requirements.txt
```

### Install Audio Plugin (TTS and ASR) Requirements

```Bash
pip install -r ./intel_extension_for_transformers/neural_chat/pipeline/plugins/audio/requirements.txt
```
