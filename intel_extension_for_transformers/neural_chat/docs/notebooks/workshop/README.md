# QuickStart: Intel® Extension For Transformers*: NeuralChat on 4th Generation Intel® Xeon® Scalable Processors

## Prepare Environment

### Docker
TBD

### Baremetal

#### Create and activate conda environment
Create the conda environment, activate it, and install the ipykernel so the environment can be used by the Jupyter notebook.
```Bash
conda create -n itrex python=3.10 -y
conda activate itrex
```

#### Install Intel® Extension for Transformers* from source
Set the environment variable _WORKSPACE_. This is where you will install the Intel® Extension for Transformers* repo and compile it from source.
```Bash
export WORKSPACE=<current_working_directory>
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
pip install -r requirements.txt
pip install -v .
```

#### Run the script to set up the environment
Navigate to the directory with this notebook. Install requirements for NeuralChat, Retrieval Plugin, and Audio Plugin (TTS and ASR).
```Bash
cd intel_extension_for_transformers/neural_chat/docs/notebooks/workshop
bash env_setup.sh
```

## Get Started with NeuralChat
Start a Jupyter notebook process.
```Bash
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```

Open a web browser to <IP_ADDRESS_OF_INSTANCE>:8888 or by clicking on the link given after Jupyter notebook as started. In some cases, you may need to use [SSH tunneling](https://www.ssh.com/academy/ssh/tunneling-example), then open the web browser to localhost:<PORT>, where PORT can be 8888 as in the example above.


Open the notebook `01_quickstart_neuralchat.ipynb` and run all cells. Ensure the "itrex" kernel is selected on the upper right corner.
