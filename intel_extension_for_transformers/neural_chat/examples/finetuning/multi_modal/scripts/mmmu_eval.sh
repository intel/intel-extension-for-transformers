
cd eval/mmmu_eval

python run_llava.py \
--output_path example_outputs/llava1.5_13b_val.json \
--model_path liuhaotian/llava-v1.5-13b \
--config_path configs/llava1.5.yaml

# evaluate the results
python main_eval_only.py --output_path example_outputs/llava1.5_13b_val.json
