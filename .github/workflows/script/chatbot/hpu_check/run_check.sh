#!/bin/bash
# check if hpu image exists
image_name="chatbotinfer-hpu"
cid=$(docker image ls -q -f "reference=$image_name")
if [[ -z "$cid" ]]; then
        echo "no $image_name image found."
        image_name="chatbotfinetune-hpu"
        cid=$(docker image ls -q -f "reference=$image_name")
        if [[ -z "$cid" ]]; then
                echo "$image_name not found too. Exiting without checking hpu"
                exit 0
        fi
fi

# stop running container
cont_name="chatbot-hpu-check"
cid=$(docker ps -q --filter "name=$cont_name")
if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid; fi

# run checks
script_dir=$(dirname "$0")
docker run --rm --runtime=habana -v $script_dir:/root/chatbot --name="$cont_name" --hostname="chatbot-hpu-check-container" "$image_name" bash -c "python /root/chatbot/to_hpu.py"

