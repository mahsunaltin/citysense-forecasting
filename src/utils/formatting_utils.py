"""
Utility for printing experiment arguments in a well-formatted way.
"""


def print_args(args):
    """
    Print experiment arguments in an organized, readable format.
    
    Args:
        args: Argument namespace or dictionary containing configuration
    """
    # Convert to dict if needed
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    else:
        args_dict = args
    
    print("\033[1m" + "═" * 80 + "\033[0m")
    print("\033[1m" + "EXPERIMENT CONFIGURATION".center(80) + "\033[0m")
    print("\033[1m" + "═" * 80 + "\033[0m\n")
    
    # Basic Configuration
    print("\033[1m" + "Basic Configuration" + "\033[0m")
    print("─" * 80)
    _print_key_value(args_dict, 'task_name', 'Task Name')
    _print_key_value(args_dict, 'is_training', 'Is Training')
    _print_key_value(args_dict, 'model_id', 'Model ID')
    _print_key_value(args_dict, 'model', 'Model')
    print()

    # Data Configuration
    print("\033[1m" + "Data Configuration" + "\033[0m")
    print("─" * 80)
    _print_key_value(args_dict, 'data', 'Dataset')
    _print_key_value(args_dict, 'root_path', 'Root Path')
    _print_key_value(args_dict, 'data_path', 'Data Path')
    _print_key_value(args_dict, 'features', 'Features')
    _print_key_value(args_dict, 'target', 'Target')
    _print_key_value(args_dict, 'freq', 'Frequency')
    print()

    # Forecasting Configuration
    if args_dict.get('task_name') == 'forecast':
        print("\033[1m" + "Forecasting Configuration" + "\033[0m")
        print("─" * 80)
        _print_key_value(args_dict, 'seq_len', 'Sequence Length')
        _print_key_value(args_dict, 'label_len', 'Label Length')
        _print_key_value(args_dict, 'pred_len', 'Prediction Length')
        _print_key_value(args_dict, 'inverse', 'Inverse Transform')
        print()

    # Model Architecture
    print("\033[1m" + "Model Architecture" + "\033[0m")
    print("─" * 80)
    _print_key_value(args_dict, 'd_model', 'Model Dimension')
    _print_key_value(args_dict, 'n_heads', 'Attention Heads')
    _print_key_value(args_dict, 'e_layers', 'Encoder Layers')
    _print_key_value(args_dict, 'd_layers', 'Decoder Layers')
    _print_key_value(args_dict, 'd_ff', 'FFN Dimension')
    _print_key_value(args_dict, 'dropout', 'Dropout')
    _print_key_value(args_dict, 'activation', 'Activation')
    print()

    # Training Configuration
    print("\033[1m" + "Training Configuration" + "\033[0m")
    print("─" * 80)
    _print_key_value(args_dict, 'train_epochs', 'Epochs')
    _print_key_value(args_dict, 'batch_size', 'Batch Size')
    _print_key_value(args_dict, 'learning_rate', 'Learning Rate')
    _print_key_value(args_dict, 'patience', 'Patience')
    _print_key_value(args_dict, 'lradj', 'LR Adjustment')
    _print_key_value(args_dict, 'itr', 'Iterations')
    print()

    # Hardware Configuration
    print("\033[1m" + "Hardware Configuration" + "\033[0m")
    print("─" * 80)
    _print_key_value(args_dict, 'use_gpu', 'Use GPU')
    _print_key_value(args_dict, 'gpu', 'GPU ID')
    _print_key_value(args_dict, 'gpu_type', 'GPU Type')
    _print_key_value(args_dict, 'use_multi_gpu', 'Multi-GPU')
    print()
    
    # Special Features (only if enabled)
    special_features = []
    if args_dict.get('use_retrieval'):
        special_features.append('Retrieval Module')
    if args_dict.get('use_dtw'):
        special_features.append('DTW Metrics')
    if args_dict.get('augmentation_ratio', 0) > 0:
        special_features.append(f"Data Augmentation (×{args_dict['augmentation_ratio']})")
    
    if special_features:
        print("\033[1m" + "Special Features" + "\033[0m")
        print("─" * 80)
        for feature in special_features:
            print(f"  ✓ {feature}")
        print()
    
    print("\033[1m" + "═" * 80 + "\033[0m\n")


def _print_key_value(args_dict, key, display_name, width=25):
    """
    Helper function to print a key-value pair.
    
    Args:
        args_dict: Dictionary of arguments
        key: Key to look up
        display_name: Human-readable name
        width: Width for alignment
    """
    if key in args_dict:
        value = args_dict[key]
        # Format boolean values
        if isinstance(value, bool):
            value = '✓' if value else '✗'
        print(f"  {display_name:<{width}}: {value}")

