#!/bin/bash

python /home/LLM_Compression/QUIK/experiments/fake_quant/llama.py \
    --model /home/LLaMA/huggingface/Llama-2-7b-hf \
    --path_to_scales sensitivity_metrics/salient_columns/metrics/llama-2-7b-obd-max-w4_ptb.pt \
    --path_to_save_quant_model QUIK/quant_weights/llama7b_w4_a16_obd_max \
    --fp_features 128 --a_bits 16 --w_bits 4 --w_clip --dataset wikitext2