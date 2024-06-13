#!/bin/bash

python /home/LLM_Compression/QUIK/experiments/fake_quant/llama.py \
    --model /home/LLaMA/huggingface/Llama-2-7b-hf \
    --path_to_scales /home/llm_compression/Quantization/Weight_scales/obs_scales/sheared_llama7b_obs_max_sqrt_w4_ptb_max.pt \
    --path_to_save_quant_model /QUIK/llama7b_w4_a16_owq_max \
    --fp_features 128 --a_bits 16 --w_bits 4 --w_clip --dataset wikitext2