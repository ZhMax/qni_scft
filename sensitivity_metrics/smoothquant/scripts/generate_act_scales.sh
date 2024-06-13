#!/bin/bash

python /sensitivity_metrics/smoothquant/generate_act_scales.py \
        --model-name=/home/LLaMA/Llama-2-7b-hf \
        --output-path=/sensitivity_metrics/smoothquant/act_scales/llama-2-7b-hf.pt \
        --dataset-path=/sensitivity_metrics/datasets/val.jsonl.zst