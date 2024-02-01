This README is designed to walk you through setting up the backend for a code-generating chatbot using the NeuralChat framework. You can deploy this chatbot on various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU, and Client GPU.

This code-generating chatbot demonstrates how to deploy it specifically on a Laptop PC. To ensure smooth operation on a laptop, we need to implement [LLM runtime optimization](../../../../../../llm/runtime/graph/README.md) to accelerate the inference process.

# Setup Conda

First, you need to install and configure the Conda environment:

Visit the [Miniconda download page](https://docs.conda.io/projects/miniconda/en/latest/) and download the installer suitable for your Windows system.
Locate the downloaded installer file (e.g., Miniconda3-latest-Windows-x86_64.exe for Miniconda). Double-click the installer to launch it. 
To create a new Conda environment, use the command: "conda create -n myenv python=3.9.0"

# Install visual cpp build tools

Visual C++ Build Tools is a package provided by Microsoft that includes tools required to build C++ projects using Visual Studio without installing the full Visual Studio IDE. These tools are essential for compiling, linking, and building intel extension for transformers.

To install the Visual C++ Build Tools, visit the following link: [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
Once there, you'll find download options and instructions for installation based on your specific requirements.

# Install intel extension for transformers

Install the intel extension for transformers from source code to get the latest features of LLM runtime.

```bash
pip clone https://github.com/intel/intel-extension-for-transformers.git
cd intel-extension-for-transformers
pip install -r requirements.txt
pip install -e .
```

# Install Python dependencies

Install dependencies using pip

```bash
pip install ../../../../../requirements_pc.txt
pip install transformers==4.35.2
```

# Configure the codegen.yaml

You can customize the configuration file 'codegen.yaml' to match your environment setup. Here's a table to help you understand the configurable options:

| Item               | Value                                |
| ------------------ | -------------------------------------|
| host               | 127.0.0.1                            |
| port               | 8000                                 |
| model_name_or_path | "codellama/CodeLlama-7b-hf"          |
| device             | "cpu"                                |
| tasks_list         | ['textchat']                         |
| optimization       |                                      |
|                    |  use_llm_runtime  | true             |
|                    |  optimization_type| "weight_only"    |
|                    |  compute_dtype    | "int8"           |
|                    |  weight_dtype     | "int4"           |



# Run the Code Generation Chatbot server
To start the code-generating chatbot server, use the following command:

```shell
nohup python run_code_gen.py &
```
