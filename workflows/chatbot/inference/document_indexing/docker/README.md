Intel Chatbot Document Indexing Dockerfile installer for Ubuntu22.04

# Build Docker Image

Option 1 (default): you could use docker build to build the docker image in your environment.
```
docker build ./ -f Dockerfile -t document_indexing:latest
```

Option 2: If you need to use proxy, please use the following command.
```
docker build ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f Dockerfile -t document_indexing:latest
```

# Run the Document Indexing


## Run the Docker Image

```
docker run --privileged -it document_indexing:latest
```


## Sparse Indexing

Simply provide the file path using the --file_path parameter and choose an appropriate local document store using the --store parameter. Please note the In Memory method does not support local database storage. It requires users to perform document processing every time.

```
python doc_index.py --file_path "xxx" --output_path "xxx" --embedding_method sparse --store Elasticsearch
```


## Dense Indexing

Users have the flexibility to choose their preferred pretrained encoder model for the process.

```
python doc_index.py --file_path "xxx" --output_path "xxx" --embedding_model hkunlp/instructor-large --embedding_method dense --store Chroma
```


