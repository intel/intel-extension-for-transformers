master_node=$1
slave_node=$2
prepare_script="$(cat << 'EOF'
oneccl_bindings_for_pytorch_path=\$(python -c \"from oneccl_bindings_for_pytorch import cwd; print(cwd)\")
source \$oneccl_bindings_for_pytorch_path/env/setvars.sh
export CCL_WORKER_COUNT=1
export MASTER_ADDR=$master_node
export I_MPI_HYDRA_IFACE=eth0
EOF
)"
docker exec "chatbotfinetune-mpi-s0" bash -c "echo \"$prepare_script\" >> ~/.bashrc"
docker exec "chatbotfinetune-mpi-s1" bash -c "echo \"$prepare_script\" >> ~/.bashrc"
echo "$master_node" > ./hosts2
echo "$slave_node" >> ./hosts2
