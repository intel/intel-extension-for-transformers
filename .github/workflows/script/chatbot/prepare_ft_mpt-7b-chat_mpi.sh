master_node=$1
slave_node=$2
runner_name=$3
prepare_script="$(cat << 'EOF'
oneccl_bindings_for_pytorch_path=\$(python -c \"from oneccl_bindings_for_pytorch import cwd; print(cwd)\")
source \$oneccl_bindings_for_pytorch_path/env/setvars.sh
export CCL_WORKER_COUNT=1
export I_MPI_HYDRA_IFACE=eth0
EOF
)"
# for launching mpirun from yaml
docker exec "chatbot-${runner_name}finetune-mpi-s0" bash -c "cd /root/chatbot; echo \"source activate && conda activate neuralchat\" > bash_setup.sh; echo \"$prepare_script\" >> bash_setup.sh; echo export MASTER_ADDR=$master_node >> bash_setup.sh"
# for ssh setup mpi and oneccl properly
docker exec "chatbot-${runner_name}finetune-mpi-s0" bash -c "echo \"$prepare_script\" >> ~/.bashrc; echo export MASTER_ADDR=$master_node >> ~/.bashrc; lamboot;"
docker exec "chatbot-${runner_name}finetune-mpi-s1" bash -c "echo \"$prepare_script\" >> ~/.bashrc; echo export MASTER_ADDR=$master_node >> ~/.bashrc; lamboot;"

echo "$master_node" > ./hosts2
echo "$slave_node" >> ./hosts2
