#!/bin/bash

# Install NeuralChat requirements
pip install -r ./intel_extension_for_transformers/neural_chat/requirements_cpu.txt

# Install Retrieval Plugin Requirements
pip install -r ./intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval/requirements.txt

# Install Audio Plugin (TTS and ASR) Requirements
pip install -r ./intel_extension_for_transformers/neural_chat/pipeline/plugins/audio/requirements.txt

# Install notebook and set up the ipykernel so the environment can be used in Jupyter notebook
pip install notebook
python -m ipykernel install --user --name itrex --display-name "itrex"

# Install other libraries
pip install pydub
