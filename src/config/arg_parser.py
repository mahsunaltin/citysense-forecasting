"""
Argument parser for time series forecasting experiments.
Centralizes all command-line argument definitions.
"""

import argparse


def get_parser():
    """
    Create and return the argument parser for the experiment.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Time Series Forecasting Framework')

    # Basic configuration
    _add_basic_config(parser)
    
    # Data loader configuration
    _add_data_config(parser)
    
    # Forecasting task configuration
    _add_forecasting_config(parser)
    
    # Model architecture configuration
    _add_model_config(parser)
    
    # Optimization configuration
    _add_optimization_config(parser)
    
    # GPU configuration
    _add_gpu_config(parser)
    
    # Additional features
    _add_additional_features(parser)
    
    return parser


def _add_basic_config(parser):
    """Add basic configuration arguments."""
    group = parser.add_argument_group('Basic Configuration')
    group.add_argument('--task_name', type=str, required=True, default='forecast',
                       help='Task name: [forecast]')
    group.add_argument('--is_training', type=int, required=True, default=1,
                       help='Training status: 1 for training, 0 for testing')
    group.add_argument('--model_id', type=str, required=True, default='test',
                       help='Model identifier for this experiment')
    group.add_argument('--model', type=str, required=True, default='TimeMixer',
                       help='Model name: [TimeMixer, WPMixer, RAFT]')


def _add_data_config(parser):
    """Add data loader configuration arguments."""
    group = parser.add_argument_group('Data Configuration')
    group.add_argument('--data', type=str, required=True, default='ETTh1',
                       help='Dataset name')
    group.add_argument('--root_path', type=str, default='./dataset/ETT-small/',
                       help='Root path of the data file')
    group.add_argument('--data_path', type=str, default='ETTh1.csv',
                       help='Data file name')
    group.add_argument('--features', type=str, default='M',
                       help='Forecasting task type: [M, S, MS] - '
                            'M: multivariate->multivariate, '
                            'S: univariate->univariate, '
                            'MS: multivariate->univariate')
    group.add_argument('--target', type=str, default='OT',
                       help='Target feature in S or MS task')
    group.add_argument('--freq', type=str, default='h',
                       help='Time features encoding frequency: '
                            '[s:secondly, t:minutely, h:hourly, d:daily, '
                            'b:business days, w:weekly, m:monthly]')
    group.add_argument('--checkpoints', type=str, default='./checkpoints/',
                       help='Location of model checkpoints')


def _add_forecasting_config(parser):
    """Add forecasting task configuration arguments."""
    group = parser.add_argument_group('Forecasting Configuration')
    group.add_argument('--seq_len', type=int, default=96,
                       help='Input sequence length')
    group.add_argument('--label_len', type=int, default=48,
                       help='Start token length')
    group.add_argument('--pred_len', type=int, default=96,
                       help='Prediction sequence length')
    group.add_argument('--seasonal_patterns', type=str, default='Monthly',
                       help='Subset for M4 dataset')
    group.add_argument('--inverse', action='store_true',
                       help='Inverse output data', default=False)


def _add_model_config(parser):
    """Add model architecture configuration arguments."""
    group = parser.add_argument_group('Model Architecture')
    
    # Core architecture parameters
    group.add_argument('--enc_in', type=int, default=7,
                       help='Encoder input size (number of features)')
    group.add_argument('--dec_in', type=int, default=7,
                       help='Decoder input size')
    group.add_argument('--c_out', type=int, default=7,
                       help='Output size (number of predicted features)')
    group.add_argument('--d_model', type=int, default=512,
                       help='Dimension of model')
    group.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    group.add_argument('--e_layers', type=int, default=2,
                       help='Number of encoder layers')
    group.add_argument('--d_layers', type=int, default=1,
                       help='Number of decoder layers')
    group.add_argument('--d_ff', type=int, default=2048,
                       help='Dimension of FCN')
    
    # Model-specific parameters
    group.add_argument('--expand', type=int, default=2,
                       help='Expansion factor for Mamba')
    group.add_argument('--d_conv', type=int, default=4,
                       help='Conv kernel size for Mamba')
    group.add_argument('--top_k', type=int, default=5,
                       help='Top-k for TimesBlock')
    group.add_argument('--num_kernels', type=int, default=6,
                       help='Number of kernels for Inception')
    group.add_argument('--moving_avg', type=int, default=25,
                       help='Window size of moving average')
    group.add_argument('--factor', type=int, default=1,
                       help='Attention factor')
    group.add_argument('--distil', action='store_false',
                       help='Disable distilling in encoder',
                       default=True)
    group.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    group.add_argument('--embed', type=str, default='timeF',
                       help='Time features encoding: [timeF, fixed, learned]')
    group.add_argument('--activation', type=str, default='gelu',
                       help='Activation function')
    group.add_argument('--output_attention', action='store_true',
                       help='Output attention in encoder')
    
    # Decomposition and normalization
    group.add_argument('--channel_independence', type=int, default=1,
                       help='0: channel dependence, 1: channel independence')
    group.add_argument('--decomp_method', type=str, default='moving_avg',
                       help='Series decomposition method: [moving_avg, dft_decomp]')
    group.add_argument('--use_norm', type=int, default=1,
                       help='Whether to use normalization: 1=True, 0=False')
    
    # Downsampling parameters
    group.add_argument('--down_sampling_layers', type=int, default=0,
                       help='Number of down sampling layers')
    group.add_argument('--down_sampling_window', type=int, default=1,
                       help='Down sampling window size')
    group.add_argument('--down_sampling_method', type=str, default=None,
                       help='Down sampling method: [avg, max, conv]')
    
    # Segment and patch parameters
    group.add_argument('--seg_len', type=int, default=96,
                       help='Segment length for SegRNN')
    group.add_argument('--patch_len', type=int, default=16,
                       help='Patch length for TimeXer')
    
    # Retrieval and period parameters
    group.add_argument('--n_period', type=int, default=3,
                       help='Number of periods')
    group.add_argument('--topm', type=int, default=20,
                       help='Number of retrievals')
    group.add_argument('--use_retrieval', type=bool, default=False,
                       help='Whether to use retrieval module')
    
    # De-stationary projector
    group.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                       help='Hidden layer dimensions of projector')
    group.add_argument('--p_hidden_layers', type=int, default=2,
                       help='Number of hidden layers in projector')


def _add_optimization_config(parser):
    """Add optimization configuration arguments."""
    group = parser.add_argument_group('Optimization')
    group.add_argument('--num_workers', type=int, default=10,
                       help='Data loader num workers')
    group.add_argument('--itr', type=int, default=1,
                       help='Number of experiment iterations')
    group.add_argument('--train_epochs', type=int, default=10,
                       help='Number of training epochs')
    group.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    group.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience')
    group.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Optimizer learning rate')
    group.add_argument('--des', type=str, default='test',
                       help='Experiment description')
    group.add_argument('--loss', type=str, default='MSE',
                       help='Loss function')
    group.add_argument('--lradj', type=str, default='type1',
                       help='Learning rate adjustment strategy')
    group.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision training',
                       default=False)


def _add_gpu_config(parser):
    """Add GPU configuration arguments."""
    group = parser.add_argument_group('GPU Configuration')
    group.add_argument('--use_gpu', type=bool, default=True,
                       help='Use GPU for training')
    group.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    group.add_argument('--gpu_type', type=str, default='cuda',
                       help='GPU type: [cuda, mps]')
    group.add_argument('--use_multi_gpu', action='store_true',
                       help='Use multiple GPUs',
                       default=False)
    group.add_argument('--devices', type=str, default='0,1,2,3',
                       help='Device IDs for multiple GPUs')


def _add_additional_features(parser):
    """Add additional feature arguments."""
    # DTW metrics
    group = parser.add_argument_group('Evaluation Metrics')
    group.add_argument('--use_dtw', type=bool, default=False,
                       help='Use DTW metric (time-consuming)')
    
    # Data augmentation
    aug_group = parser.add_argument_group('Data Augmentation')
    aug_group.add_argument('--augmentation_ratio', type=int, default=0,
                          help='How many times to augment data')
    aug_group.add_argument('--seed', type=int, default=2,
                          help='Random seed for augmentation')
    aug_group.add_argument('--jitter', default=False, action="store_true",
                          help='Jitter augmentation')
    aug_group.add_argument('--scaling', default=False, action="store_true",
                          help='Scaling augmentation')
    aug_group.add_argument('--permutation', default=False, action="store_true",
                          help='Equal length permutation augmentation')
    aug_group.add_argument('--randompermutation', default=False, action="store_true",
                          help='Random length permutation augmentation')
    aug_group.add_argument('--magwarp', default=False, action="store_true",
                          help='Magnitude warp augmentation')
    aug_group.add_argument('--timewarp', default=False, action="store_true",
                          help='Time warp augmentation')
    aug_group.add_argument('--windowslice', default=False, action="store_true",
                          help='Window slice augmentation')
    aug_group.add_argument('--windowwarp', default=False, action="store_true",
                          help='Window warp augmentation')
    aug_group.add_argument('--rotation', default=False, action="store_true",
                          help='Rotation augmentation')
    aug_group.add_argument('--spawner', default=False, action="store_true",
                          help='SPAWNER augmentation')
    aug_group.add_argument('--dtwwarp', default=False, action="store_true",
                          help='DTW warp augmentation')
    aug_group.add_argument('--shapedtwwarp', default=False, action="store_true",
                          help='Shape DTW warp augmentation')
    aug_group.add_argument('--wdba', default=False, action="store_true",
                          help='Weighted DBA augmentation')
    aug_group.add_argument('--discdtw', default=False, action="store_true",
                          help='Discriminative DTW warp augmentation')
    aug_group.add_argument('--discsdtw', default=False, action="store_true",
                          help='Discriminative shapeDTW warp augmentation')
    aug_group.add_argument('--extra_tag', type=str, default="",
                          help='Extra tag for experiment')


def parse_args():
    """
    Parse command-line arguments and perform post-processing.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = get_parser()
    args = parser.parse_args()
    
    # Set device configuration
    import torch
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
    
    # Setup multi-GPU if enabled
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    return args
