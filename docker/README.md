# Docker
Follow these instructions to set up and run our provided Docker image.

## Set Up Docker Engine and Docker Compose
You'll need to install Docker Engine on your development system. Note that while **Docker Engine** is free to use, **Docker Desktop** may require you to purchase a license. See the [Docker Engine Server installation instructions](https://docs.docker.com/engine/install/#server) for details.

To build and run this workload inside a Docker Container, ensure you have Docker Compose installed on your machine. If you don't have this tool installed, consult the official [Docker Compose installation documentation](https://docs.docker.com/compose/install/linux/#install-the-plugin-manually).

```bash
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.19.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
docker compose version
```

## Set Up Docker Image
Build or Pull the provided docker images.

```bash
cd docker
docker compose build
```
OR
```bash
docker pull intel/ai-tools:itrex-1.3.0
docker pull intel/ai-tools:itrex-1.3.0-devel
```

## Use Docker Image
Utilize the TLT CLI without installation by using the provided docker image and docker compose.

```bash
docker compose run devel
docker compose run devel python setup.py sdist
docker compose run devel python tests/<test_mytest>.py
```

# Kubernetes
## 1. Install Helm
- Install [Helm](https://helm.sh/docs/intro/install/)
```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
chmod 700 get_helm.sh && \
./get_helm.sh
```
## 2. Setting up Training Operator
Install the standalone operator from GitHub/Artifacthub or use a pre-existing Kubeflow configuration.
```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"
```
OR
```bash
helm repo add cowboysysop https://cowboysysop.github.io/charts/
helm install <release name> cowboysysop/training-operator
```
## 3. Deploy ITREX Distributed Job
For more customization information, see the chart [README](./chart/README.md)
```bash
export NAMESPACE=kubeflow
helm install --namespace ${NAMESPACE} --set ... itrex-distributed ./chart
```
## 4. View 
To view your workflow progress
```bash
kubectl get -o yaml pytorchjob itrex-distributed -n ${NAMESPACE}
```
