import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()


    ### DATASET ###
    data = config.data = ml_collections.ConfigDict()
    data.dataset_name = "allenai/tulu-v2-sft-mixture" #"VityaVitalich/openorca"
    data.dataset_config_name = None
    data.max_seq_length = 2048
    data.dataset_percentage = 100
    data.validation_split_percentage = 5
    data.trust_remote_code = True
    data.preprocessing_num_workers = 8
    data.seed = 11


    ### MODEL CHECKPOINT ###
    config.model_type = 'Auto'
    config.model_name_or_path = '/home/LLaMA/Llama-2-7b-hf'
    config.model_config_name = None
    config.tokenizer_name = None
    config.token = None
    config.use_fast_tokenizer = True
    config.trust_remote_code = True
    config.max_memory = 79
    ## SAVING DIRS ###
    # config.cache_dir = None
    config.output_dir = '/home/exp_results/Llama-2-7b-hf-scft-qni-4bit'

    ### TRAINING ###
    config.run_name = '4bit-qni-ft'
    config.resume_from_checkpoint = None
    # config.num_train_epochs = 1
    config.max_steps = 1000
    config.learning_rate = 1e-4 #1e-4
    config.weight_decay = 0.0
    config.lr_scheduler_type = 'linear'
    config.warmup_ratio =  0.03
    config.seed = 11
    config.per_device_train_batch_size = 2
    config.per_device_eval_batch_size = 2
    config.gradient_accumulation_steps = 16
    config.gradient_checkpointing = False
    config.report_to = None
    ### eval ###
    # config.evaluation_strategy = 'steps'
    # config.eval_steps = 100
    # config.evaluation_strategy = 'no'
    # config.eval_steps = None
    ### save ###
    config.save_strategy = 'steps'
    config.save_steps = 250

    ### LORA ###
    config.use_lora = False
    config.lora_rank = 128
    config.lora_alpha = 128
    config.lora_dropout = 0.1
    config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

    ### Path to a file containing sensitivity metrics
    config.outliers = {
        'path_to_act_scales': 'sensitivity_metrics/salient_columns/metrics/llama-2-7b-obd-max-w4_ptb.pt',
        'fp_features_num': 128, #number of salient columns
    }

    ### QuantizedLinear
    config.QuantizedLinear = {
        'replace': True,
        'training_mode': 'train_outlier' #train_full, train_outlier (only salient columns), train_quant (only non-salient columns)
    }

    ### Load Quantized Weight After Quik
    config.loading_quik_quant_weight = {
        'load_weight': False, #load quantized weights with scales from state_dict to replace full-precision weights of the model
        'path_to_quant_params': 'QUIK/quant_weights/llama7b_w4_a16_obd_max/quant_params.pt',
        'learnable_scale': False #update quantization scales during training
    }

    ### BitNoiseQuant
    config.BitNoiseQuant = {
        'add_quant_noise': True,
        'predict': False, #inject noise during model evaluation
        'compute_scale': True, #compute quantization scales for noise 
        'noise_type': 'normal', # 'normal', 'uniform'
        'learnable_scale': False, #update noise scales during training
        'layer_bits': {'q': 4, 'k': 4, 'v': 4, 'o': 4, 'down': 4, 'gate': 4, 'up': 4} #bit-widths of noise for each projection
    }

    ### NORM TWEEKING ###
    config.norm_tweek = False

    ###LM HEAD ###
    config.train_lm_head = False
    
    return config
    