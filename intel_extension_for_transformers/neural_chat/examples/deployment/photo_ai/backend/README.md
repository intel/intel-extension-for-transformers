This README is intended to guide you through setting up the backend for a Photo AI demo using the NeuralChat framework. You can deploy it on various platforms, including Intel XEON Scalable Processors, Habana's Gaudi processors (HPU), Intel Data Center GPU and Client GPU, Nvidia Data Center GPU and Client GPU.


# Setup Environment


## Setup Conda

First, you need to install and configure the Conda environment:

```shell
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda*.sh
source ~/.bashrc
```

## Install numactl

Next, install the numactl library:

```shell
sudo apt install numactl
```

## Install Python dependencies

Install the following Python dependencies using Conda:

```shell
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
conda install git-lfs -y
# install libGL.so.1 for opencv
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
```

Install other dependencies using pip:

```bash
pip install -r ../../../requirements.txt
```

## Install Models
```shell
git-lfs install
# download neural-chat-7b-v3-1 model for NER plugin
git clone https://huggingface.co/Intel/neural-chat-7b-v3-1
# download spacy model for NER post process
python -m spacy download en_core_web_lg
```


# Setup Database
## Install MySQL
```shell
# install mysql
sudo apt-get install mysql-server
# start mysql server
systemctl status mysql
```

## Create Tables
```shell
cd ../../../utils/database/
# login mysql
mysql
source ./init_db_ai_photos.sql
```

## Create Image Database
```shell
mkdir /home/nfs_images
export IMAGE_SERVER_IP="your.server.ip"
```

# Configurate photoai.yaml

You can customize the configuration file `photoai.yaml` to match your environment setup. Here's a table to help you understand the configurable options:

|  Item               | Value                                  |
| ------------------- | ---------------------------------------|
| host                | 127.0.0.1                              |
| port                | 9000                                   |
| model_name_or_path  | "./neural-chat-7b-v3-1"        |
| device              | "auto"                                  |
| asr.enable          | true                                   |
| tts.enable          | true                                   |
| ner.enable          | true                                   |
| tasks_list          | ['voicechat', 'photoai']               |


# Configurate Environment Variables

Configurate all of the environment variables in file `run.sh` using `export XXX=xxx`. Here's a table of all the variables needed to configurate.

|  Variable           | Value                                  |
| ------------------- | ---------------------------------------|
| MYSQL_HOST          | 127.0.0.1 if you deploy mysql on local server.  |
| MYSQL_USER          | default: 'root'                                   |
| MYSQL_PASSWORD      | password of the specified user        |
| MYSQL_PORT          | default: 3306                                  |
| MYSQL_DB            | default: 'ai_photos'                                |
| IMAGE_SERVER_IP     | The IP of server which you store user uploaded images      |
| IMAGE_ROOT_PATH     | local path of image storing path                     |
| RETRIEVAL_FILE_PATH | local path of where you store retrieval files               |
| GOOGLE_API_KEY      | your google api key to get gps information from images           |


# Run the PhotoAI server
To start the PhotoAI server, use the following command:

```shell
nohup bash run.sh &
```
