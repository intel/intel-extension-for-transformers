pid0=$(cat ./pid0.txt)
pid1=$(cat ./pid1.txt)
ret_status=0
if [ -z "$pid0" ] || [ -z "$pid1" ]; then
  ret_status=1
else
  # query finetuning process status
  pid0_s=$(docker exec "chatbotfinetune-s0" bash -c "ps -p $pid0 -o pid=")
  pid1_s=$(docker exec "chatbotfinetune-s1" bash -c "ps -p $pid1 -o pid=")
  while [ ! -z "$pid0_s" ] || [ ! -z "$pid1_s" ]; do
    sleep 30
    pid0_s=$(docker exec "chatbotfinetune-s0" bash -c "ps -p $pid0 -o pid=")
    pid1_s=$(docker exec "chatbotfinetune-s1" bash -c "ps -p $pid1 -o pid=")
  done
fi
# output logs
echo "Stdout from master ft process:"
cat ./log0.txt
echo "Stderror from master ft process:"
cat ./error0.txt
echo "Stdout from slave process:"
cat ./log1.txt
echo "Stderror from slave ft process:"
cat ./error1.txt
if [ "$ret_status" -eq 1 ]; then
  echo "Error: failed to create finetuning processes. Master pid is $pid0. Slave pid is $pid1"
  exit 1
fi
# check if ft failed by searching stderror
search0=$(grep "finetune_clm.py FAILED" ./error0.txt)
search1=$(grep "finetune_clm.py FAILED" ./error1.txt)
if [ ! -z "$search0" ] || [ ! -z "$search1" ]; then
  echo "Error: finetuning failed"
  exit 1
fi