"""
Default hyperparameters and configuration settings for experiments.
"""

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    'TimeMixer': {
        'seq_len': 96,
        'label_len': 0,
        'pred_len': 96,
        'e_layers': 2,
        'd_model': 16,
        'd_ff': 32,
        'down_sampling_layers': 3,
        'down_sampling_window': 2,
        'down_sampling_method': 'avg',
        'channel_independence': 1,
        'decomp_method': 'moving_avg',
        'moving_avg': 25,
    },
    'WPMixer': {
        'seq_len': 96,
        'label_len': 0,
        'pred_len': 96,
        'e_layers': 2,
        'd_model': 16,
        'd_ff': 32,
    },
    'RAFT': {
        'seq_len': 96,
        'label_len': 0,
        'pred_len': 96,
        'e_layers': 2,
        'd_model': 16,
        'd_ff': 32,
        'use_retrieval': True,
    }
}

# Default training parameters
DEFAULT_TRAINING_PARAMS = {
    'train_epochs': 10,
    'batch_size': 128,
    'learning_rate': 0.01,
    'patience': 10,
    'num_workers': 10,
    'itr': 1,
    'lradj': 'type1',
    'use_amp': False,
}

# Dataset-specific parameters
DATASET_PARAMS = {
    'ETTh1': {'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'freq': 'h'},
    'ETTh2': {'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'freq': 'h'},
    'ETTm1': {'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'freq': 't'},
    'ETTm2': {'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'freq': 't'},
}
