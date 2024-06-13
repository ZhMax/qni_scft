This directory contains code to fine-tune salient columns in LLMs.

To fine-tune salient columns of a full-precision LLM with quantization noise injection, run the following command:

```bash
python llm_tune/train_instruct.py --config_path=llm_tune/configs/llama_scft_with_bitnoise_4bit.py
```

To fine-tune the salient columns of a quantized LLM with QUIK, run the following command:
```bash
python llm_tune/train_instruct.py --config_path=/home/EM_NLP/qni_scft/llm-tune/configs/llama_scft_in_quantized_model.py
```

Before running the commands, check the following variables in `configs`.  
`config.model_name_or_path` is a dicrectory containing the model.  
`config.max_memory` is maximal memory of one GPU which used for the training.  
`config.output_dir` is a directory where the checkpoints will be saved during the training.  
`config.outliers['path_to_act_scales']` is a path to the sensitivity metrics which provide the importance of each column. 
`loading_quik_quant_weight['path_to_quant_params']` is a path to parameters of the model quantized with QUIK.