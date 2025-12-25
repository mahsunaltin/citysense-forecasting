"""
Utility tools for time series experiments.
Includes learning rate adjustment, early stopping, and data utilities.
"""

import os
import numpy as np
import torch
import pandas as pd
import math


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust learning rate according to schedule.
    
    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        args: Configuration arguments containing lradj and learning_rate
    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    else:
        lr_adjust = {}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'  Learning rate updated to {lr:.6f}')


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize early stopping.
        
        Args:
            patience: How many epochs to wait after last improvement
            verbose: Whether to print messages
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save if improved
            path: Path to save model checkpoint
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'  Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Save model checkpoint when validation loss improves.
        
        Args:
            val_loss: Current validation loss
            model: Model to save
            path: Path to save checkpoint
        """
        if self.verbose:
            print(f'  Validation loss improved ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """
    Dictionary with dot notation access to attributes.
    Allows accessing dict['key'] as dict.key
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    """
    Standard scaler for normalizing data.
    """
    
    def __init__(self, mean, std):
        """
        Initialize scaler with mean and standard deviation.
        
        Args:
            mean: Mean value(s)
            std: Standard deviation value(s)
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Transform data using standardization.
        
        Args:
            data: Data to transform
        
        Returns:
            Standardized data
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Inverse transform to original scale.
        
        Args:
            data: Standardized data
        
        Returns:
            Data in original scale
        """
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.html'):
    """
    Create visualization of predictions vs ground truth.
    This function now delegates to the Plotly-based visualization module.
    For enhanced visualizations, use functions in visualization module directly.
    
    Args:
        true: Ground truth values
        preds: Predicted values (optional)
        name: Path to save the HTML file
    """
    # Import here to avoid circular dependency
    from src.visualization import visual as plotly_visual
    
    # Convert .pdf extension to .html if present
    if name.endswith('.pdf'):
        name = name.replace('.pdf', '.html')
    
    plotly_visual(true, preds, name)


def adjustment(gt, pred):
    """
    Adjust predictions for anomaly detection.
    
    Args:
        gt: Ground truth binary labels
        pred: Predicted binary labels
    
    Returns:
        Tuple of (adjusted_gt, adjusted_pred)
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    """
    Calculate classification accuracy.
    
    Args:
        y_pred: Predicted labels
        y_true: True labels
    
    Returns:
        float: Accuracy score
    """
    return np.mean(y_pred == y_true)

