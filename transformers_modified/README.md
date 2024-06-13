This is a fork of 🤗 [Transformers v.4.38](https://github.com/huggingface/transformers/tree/v4.38.0) library implementing salien columns fine-tuning with quantization noise injection.

Only file `transformers_modified/src/transformers/models/llama/modeling_llama.py` was modified to include our custom layer `QuantizedLinear`.