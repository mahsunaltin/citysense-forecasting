"""
Logging utilities for time series experiments.
Provides structured logging with colors and file output.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for terminal output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
        'BOLD': '\033[1m',        # Bold
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        message = super().format(record)
        
        # Reset levelname for file handler
        record.levelname = levelname
        
        return message


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        console: Whether to add console handler
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(experiment_name: str, log_dir: str = './logs') -> logging.Logger:
    """
    Create a logger for an experiment with timestamped log file.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
    
    Returns:
        logging.Logger: Configured experiment logger
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')
    
    return setup_logger(
        name=experiment_name,
        log_file=log_file,
        level=logging.INFO,
        console=True
    )


def log_args(logger: logging.Logger, args, title: str = "Experiment Configuration"):
    """
    Log experiment arguments in a structured format.
    
    Args:
        logger: Logger instance
        args: Argument namespace or dictionary
        title: Title for the configuration section
    """
    logger.info("=" * 80)
    logger.info(f"{title:^80}")
    logger.info("=" * 80)
    
    # Convert args to dict if needed
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    else:
        args_dict = args
    
    # Group arguments by category
    categories = {
        'Basic': ['task_name', 'is_training', 'model_id', 'model'],
        'Data': ['data', 'root_path', 'data_path', 'features', 'target', 'freq'],
        'Forecasting': ['seq_len', 'label_len', 'pred_len'],
        'Model': ['d_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff', 'dropout'],
        'Training': ['train_epochs', 'batch_size', 'learning_rate', 'patience', 'itr'],
        'Hardware': ['use_gpu', 'gpu', 'gpu_type', 'use_multi_gpu'],
    }
    
    for category, keys in categories.items():
        logger.info(f"\n{category} Configuration:")
        logger.info("-" * 80)
        for key in keys:
            if key in args_dict:
                value = args_dict[key]
                logger.info(f"  {key:<25}: {value}")
    
    # Log other parameters
    other_keys = set(args_dict.keys()) - set(k for keys in categories.values() for k in keys)
    if other_keys:
        logger.info(f"\nOther Parameters:")
        logger.info("-" * 80)
        for key in sorted(other_keys):
            if not key.startswith('_') and key not in ['device', 'device_ids']:
                value = args_dict[key]
                logger.info(f"  {key:<25}: {value}")
    
    logger.info("=" * 80)


def log_metrics(logger: logging.Logger, metrics: dict, phase: str = "Test"):
    """
    Log evaluation metrics in a formatted way.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        phase: Phase name (e.g., 'Train', 'Validation', 'Test')
    """
    logger.info("=" * 80)
    logger.info(f"{phase} Metrics".center(80))
    logger.info("=" * 80)
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name:<20}: {value:.6f}")
        else:
            logger.info(f"  {metric_name:<20}: {value}")
    
    logger.info("=" * 80)


class ExperimentTracker:
    """
    Tracks experiment progress and results.
    """
    
    def __init__(self, logger: logging.Logger, experiment_name: str):
        """
        Initialize experiment tracker.
        
        Args:
            logger: Logger instance
            experiment_name: Name of the experiment
        """
        self.logger = logger
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.epoch_times = []
        self.best_metrics = {}
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log the start of an epoch."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Epoch {epoch}/{total_epochs}".center(80))
        self.logger.info(f"{'='*80}")
        self.epoch_start_time = datetime.now()
    
    def log_epoch_end(self, epoch: int, train_loss: float, val_loss: float, test_loss: float):
        """Log the end of an epoch with metrics."""
        elapsed = (datetime.now() - self.epoch_start_time).total_seconds()
        self.epoch_times.append(elapsed)
        
        self.logger.info(f"\nEpoch {epoch} Summary:")
        self.logger.info(f"  Train Loss: {train_loss:.6f}")
        self.logger.info(f"  Val Loss:   {val_loss:.6f}")
        self.logger.info(f"  Test Loss:  {test_loss:.6f}")
        self.logger.info(f"  Time:       {elapsed:.2f}s")
        
        # Track best metrics
        if 'best_val_loss' not in self.best_metrics or val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = val_loss
            self.best_metrics['best_epoch'] = epoch
            self.logger.info(f"  ★ New best validation loss!")
    
    def log_experiment_end(self):
        """Log experiment completion summary."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Experiment Summary".center(80))
        self.logger.info("=" * 80)
        self.logger.info(f"  Total Time:         {total_time:.2f}s ({total_time/60:.2f}min)")
        self.logger.info(f"  Average Epoch Time: {avg_epoch_time:.2f}s")
        
        if self.best_metrics:
            self.logger.info(f"\nBest Results:")
            for key, value in self.best_metrics.items():
                self.logger.info(f"  {key:<20}: {value}")
        
        self.logger.info("=" * 80)
