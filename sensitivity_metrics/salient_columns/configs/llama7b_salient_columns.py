import ml_collections

def model_configs():
    config = ml_collections.ConfigDict()


    ### DATASET ###
    data = config.data = ml_collections.ConfigDict()
    data.dataset_path = "/sensitivity_metrics/datasets/val.jsonl.zst"
    data.output_path = "/sensitivity_metrics/obd_scales/llama7b_obd_max_4bit_ptb.pt"
    data.max_seq_length = 512
    data.num_samples = 512
    data.trust_remote_code = True
    data.preprocessing_num_workers = 8

    ### MODEL CHECKPOINT ###
    config.model_type = 'Auto'
    config.model_name_or_path = '/home/LLaMA/Llama-2-7b-hf'
    config.token = None
    config.use_fast_tokenizer = True
    config.trust_remote_code = True

    ### Estimator ###
    config.estimator = {
        'estimator':'OBD_Estimator', #'OBDx2_Estimator', 'Wanda_Estimator'
        'agg': 'max', #max, l2 (for obd); max, max_sqrt (for wanda)
        'add_quantizer': True,
        'bit': 4
    }

    return config