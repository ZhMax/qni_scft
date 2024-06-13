# QNI-SCFT
This repository contains the code for the paper . QNI-SCFT updates only salient columns, while injecting Gaussian noise scaled by qunatization step size into non-salient columns.

## Table of contents
* [Installation](#installation)
* [Salient Columns](#salient-columns)
* [Train](#train)
* [Eval](#eval)

## Installation
### Preliminaries
We highly recommend to use docker image that supports CUDA. 
For our experiments we used the following image:  
```bash
# pull image
docker pull pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
```

Run container and install Git:
```bash
# run container
docker run -it --gpus all --ipc=host -v {local_storage}:{docker_container_storage} pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# install git
apt update && apt install git -y
```

### Packages installation
1. Clone the QNI-SCFT repository:
```bash
git clone https://github.com/ZhMax/qni_scft.git
cd qni_scft
```

2. Install QNI-SCFT integration into huggingface's Transformers library and 
additional packages:
```bash
# transformers
cd transformers_modified
pip install .

pip install sentencepiece
pip install protobuf

# configs 
pip install ml_collections

#logging
pip install wandb
```

3. Install [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation:
```bash
pip install lm-eval
```

### Dependencies

- python 3.10.13
- pytorch 2.1.0
- cuda12.1
- cudnn8

Experiments were conducted on NVIDIA A100 GPU with 80GB memory. 

## Salient Columns
To find salient columns for fine-tuning, it is necessary to compute sensitivity metrics. 
The metrics for a full-precision LLM can be estimated by the following script: 
```bash
bash sensitivity_metrics/salientcolumns/scripts/salient_metric.sh
```

## Train
**Note: ** Only the LLaMA models can be fine-tuned. 

To fine-tune salient columns of a full-precision LLM with quantization noise injection, run the following command:

```bash
python llm_tune/train_instruct.py --config_path=llm_tune/configs/llama_scft_with_bitnoise_4bit.py
```

## Eval 
Finally, you can run evaluation on zero-shot tasks benchmarks with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) by the following command:
```bash
lm_eval --model hf \
  --model_args "pretrained=<path to the directory with the fine-tuned model>" \
  --tasks winogrande,hellaswag,swag,boolq,xwinograd_en \
  --batch_size 16 \
  --num_fewshot 0 \
  --device cuda
```

## Citation
If you plan to use our work in your projects, please consider citing our paper: