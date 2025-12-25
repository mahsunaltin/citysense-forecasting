"""
Base experiment class for time series forecasting.
Provides common functionality for all experiment types.
"""

import os
import torch
from src.models import TimeMixer, WPMixer, RAFT


class Exp_Basic(object):
    """
    Base class for all experiments.
    Handles model initialization, device configuration, and common operations.
    """
    
    def __init__(self, args):
        """
        Initialize the base experiment.
        
        Args:
            args: Argument namespace containing configuration
        """
        self.args = args
        self.model_dict = {
            'TimeMixer': TimeMixer,
            'WPMixer': WPMixer,
            'RAFT': RAFT
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """
        Build and return the model instance.
        Must be implemented by subclasses.
        
        Returns:
            torch.nn.Module: Model instance
        """
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
        Acquire and configure the computation device (CPU/GPU).
        
        Returns:
            torch.device: Configured device
        """
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        """Get data loaders. Must be implemented by subclasses."""
        pass

    def vali(self):
        """Validation method. Must be implemented by subclasses."""
        pass

    def train(self):
        """Training method. Must be implemented by subclasses."""
        pass

    def test(self):
        """Testing method. Must be implemented by subclasses."""
        pass

