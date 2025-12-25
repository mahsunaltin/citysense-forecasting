"""
Main entry point for time series forecasting experiments.
Handles experiment setup, execution, and result management.
"""

import os
import random
import numpy as np
import torch
import torch.backends
from datetime import datetime

from src.config.arg_parser import parse_args
from src.utils.logging_utils import get_experiment_logger, log_args
from src.utils.formatting_utils import print_args
from src.experiments import Exp_Forecast


def set_seed(seed=2021):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_experiment_class(task_name):
    """
    Get the experiment class based on task name.
    
    Args:
        task_name: Name of the task
    
    Returns:
        Experiment class
    """
    task_map = {
        'forecast': Exp_Forecast,
    }
    
    if task_name not in task_map:
        raise NotImplementedError(f'Task {task_name} is not supported')
    
    return task_map[task_name]


def generate_setting_string(args, iteration=0):
    """
    Generate experiment setting string for naming.
    
    Args:
        args: Argument namespace
        iteration: Iteration number
    
    Returns:
        str: Setting string
    """
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        iteration
    )
    return setting


def clear_gpu_cache(gpu_type='cuda'):
    """
    Clear GPU cache based on GPU type.
    
    Args:
        gpu_type: Type of GPU ('cuda' or 'mps')
    """
    if gpu_type == 'mps' and hasattr(torch.backends, 'mps'):
        torch.backends.mps.empty_cache()
    elif gpu_type == 'cuda':
        torch.cuda.empty_cache()


def main():
    """Main execution function."""
    # Set random seed for reproducibility
    set_seed(2021)
    
    # Parse arguments
    args = parse_args()
    
    # Print configuration
    print('\n' + '='*80)
    print('TIME SERIES FORECASTING FRAMEWORK'.center(80))
    print('='*80 + '\n')
    print_args(args)
    
    # Get experiment class
    Exp = get_experiment_class(args.task_name)
    
    if args.is_training:
        # Training mode
        print(f"\n{'='*80}")
        print(f"Starting Training Mode - {args.itr} iteration(s)".center(80))
        print(f"{'='*80}\n")
        
        for ii in range(args.itr):
            # Create experiment instance
            exp = Exp(args)
            
            # Generate setting string
            setting = generate_setting_string(args, ii)
            
            print(f"\n{'='*80}")
            print(f"Iteration {ii+1}/{args.itr}: {setting}".center(80))
            print(f"{'='*80}")
            
            # Train
            print(f'\n>>> Starting training: {setting}')
            exp.train(setting)

            # Test
            print(f'\n>>> Starting testing: {setting}')
            exp.test(setting)
            
            # Clear cache
            clear_gpu_cache(args.gpu_type)
            
            print(f"\n{'='*80}")
            print(f"Iteration {ii+1}/{args.itr} Completed".center(80))
            print(f"{'='*80}\n")
        
        print(f"\n{'='*80}")
        print("All Experiments Completed Successfully!".center(80))
        print(f"{'='*80}\n")
        
    else:
        # Testing mode
        print(f"\n{'='*80}")
        print("Starting Testing Mode".center(80))
        print(f"{'='*80}\n")
        
        exp = Exp(args)
        setting = generate_setting_string(args, 0)
        
        print(f'\n>>> Testing: {setting}')
        exp.test(setting, test=1)
        
        # Clear cache
        clear_gpu_cache(args.gpu_type)
        
        print(f"\n{'='*80}")
        print("Testing Completed".center(80))
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
