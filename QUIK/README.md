# QUIK
This directory contains the code for fake quantization of LLaMA with QUIK. 
This is a PTQ method which can performe GPTQ quantization with isolating salient (outlier) columns in full-precision. 

QUIK is described in the following paper: 
https://arxiv.org/abs/2310.09259

The original repository:
clone https://github.com/IST-DASLab/QUIK.git

## Install

No installation required


## Example
To quantize an original LLaMA model, run the following script: 
```bash
bash QUIK/experiments/scripts/quik.sh
```

To quantize a LLaMA model with fine-tuned salient-columns, run the following script: 
```bash
bash QUIK/experiments/scripts/quik_after_noise_fine_tuning.sh
```

**Note:** We use QUIK method only for weight quantization.