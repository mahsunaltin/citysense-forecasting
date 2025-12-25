"""
Enhanced metrics module for time series forecasting evaluation.
Provides comprehensive evaluation metrics with better formatting and visualization.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def RSE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Root Relative Squared Error.
    
    Args:
        pred: Predictions array
        true: Ground truth array
    
    Returns:
        float: RSE value
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Correlation coefficient.
    
    Args:
        pred: Predictions array
        true: Ground truth array
    
    Returns:
        float: Correlation value
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    Args:
        pred: Predictions array
        true: Ground truth array
    
    Returns:
        float: MAE value
    """
    return np.mean(np.abs(true - pred))


def MSE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Mean Squared Error.
    
    Args:
        pred: Predictions array
        true: Ground truth array
    
    Returns:
        float: MSE value
    """
    return np.mean((true - pred) ** 2)


def RMSE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    
    Args:
        pred: Predictions array
        true: Ground truth array
    
    Returns:
        float: RMSE value
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        pred: Predictions array
        true: Ground truth array
    
    Returns:
        float: MAPE value
    """
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Mean Squared Percentage Error.
    
    Args:
        pred: Predictions array
        true: Ground truth array
    
    Returns:
        float: MSPE value
    """
    return np.mean(np.square((true - pred) / true))


def metric(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Calculate all standard metrics.
    
    Args:
        pred: Predictions array
        true: Ground truth array
    
    Returns:
        Tuple of (MAE, MSE, RMSE, MAPE, MSPE)
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def calculate_all_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    Calculate all available metrics and return as dictionary.
    
    Args:
        pred: Predictions array
        true: Ground truth array
    
    Returns:
        Dictionary with all metric values
    """
    mae, mse, rmse, mape, mspe = metric(pred, true)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'MSPE': mspe,
        'RSE': RSE(pred, true),
        'CORR': CORR(pred, true),
    }


def format_metrics(metrics: Dict[str, float], precision: int = 6) -> str:
    """
    Format metrics dictionary into a readable string.
    
    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places
    
    Returns:
        Formatted string
    """
    lines = ["Evaluation Metrics:", "=" * 50]
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {metric_name:<15}: {value:.{precision}f}")
        else:
            lines.append(f"  {metric_name:<15}: {value}")
    
    lines.append("=" * 50)
    return "\n".join(lines)


def print_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> None:
    """
    Print metrics in a formatted table for multiple experiments.
    
    Args:
        metrics_dict: Dictionary mapping experiment names to their metrics
    """
    if not metrics_dict:
        print("No metrics to display")
        return
    
    # Get all metric names
    all_metrics = set()
    for metrics in metrics_dict.values():
        all_metrics.update(metrics.keys())
    
    metric_names = sorted(all_metrics)
    exp_names = list(metrics_dict.keys())
    
    # Print header
    print("\n" + "=" * 100)
    print("Experiment Comparison".center(100))
    print("=" * 100)
    
    # Print column headers
    header = f"{'Experiment':<30} | " + " | ".join([f"{m:>12}" for m in metric_names])
    print(header)
    print("-" * 100)
    
    # Print rows
    for exp_name in exp_names:
        metrics = metrics_dict[exp_name]
        row = f"{exp_name:<30} | "
        row += " | ".join([f"{metrics.get(m, 0):>12.6f}" for m in metric_names])
        print(row)
    
    print("=" * 100 + "\n")
