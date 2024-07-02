# H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
Code for the paper "**H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**"

## Usage and Examples
### Evaluation on tasks from [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework
Using simulation mode
```bash
python run_generation.py \
    --model meta-llama/Meta-Llama-3-8B \
    --accuracy \
    --batch_size 16 \
    --h2o \
    --heavy_ratio 0.1 \
    --recent_ratio 0.1 \
    --device 0
```
To run the real_drop mode
```bash
python run_generation.py \
    --model meta-llama/Meta-Llama-3-8B \
    --accuracy \
    --batch_size 16 \
    --h2o \
    --heavy_ratio 0.1 \
    --recent_ratio 0.1 \
    --device 0
    --real_drop
```
Get the accuracy of dense model
```bash
python run_generation.py \
    --model meta-llama/Meta-Llama-3-8B \
    --accuracy \
    --batch_size 16
```