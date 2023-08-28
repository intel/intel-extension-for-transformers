ret_status=0
train_complete=1
no_failed=1

print_and_check_logs()
{
  path=$1
  if [ -f "$path" ]; then
    cat "$path"
    if [[ "$path" =~ .*\.err ]]; then
      search0=$(grep "finetune_clm.py FAILED" "$path")
      if [ ! -z "$search0" ]; then
        no_failed=0
      fi
      search1=$(grep "Training completed" "$path")
      if [ -z "$search1" ]; then
        train_complete=0
      fi
    fi
  else
    ret_status=1
  fi
}
# output logs
echo "Stdout from master ft process:"
print_and_check_logs ./chatbotfinetune-container-mpi-s0.log
echo "Stderror from master ft process:"
print_and_check_logs ./chatbotfinetune-container-mpi-s0.err
echo "Stdout from slave process:"
print_and_check_logs ./chatbotfinetune-container-mpi-s1.log
echo "Stderror from slave ft process:"
print_and_check_logs ./chatbotfinetune-container-mpi-s1.err

if [ "$train_complete" -eq 0 ] || [ "$no_failed" -eq 0 ] || [ "$ret_status" -eq 1]
then
  echo "Error: finetuning failed"
  exit 1
fi
