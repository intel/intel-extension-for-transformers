# There are two dockerfiles: for all benchmarks, and for MultiPL-E
DOCKERFILE=Dockerfile

ifeq ($(DOCKERFILE), Dockerfile)
	IMAGE_NAME=evaluation-harness
else
	IMAGE_NAME=evaluation-harness-multiple
endif

build:
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME) .

test:
	docker run -v $(CURDIR):$(CURDIR) \
	-it $(IMAGE_NAME) python3 run_generation.py --model $(CURDIR)/starcoder-3b --quantize   --sq --alpha 0.7 --ipex \
        --calib_iters 5 --calib_batch_size 1 --dataset "mbpp" --calib_split "test" --output_dir "$(CURDIR)/saved_results" \
        --int8 --accuracy --tasks multiple-py  --batch_size 20 --n_samples 20 --allow_code_execution \
        --do_sample --temperature 0.2 --limit 2

	@echo "If pass@1 is 0.25 then your configuration for standard benchmarks is correct"

all: build test
