"""
Configuration module for time series experiments.
Provides centralized configuration management and argument parsing.
"""

from .arg_parser import get_parser, parse_args
from .defaults import DEFAULT_MODEL_PARAMS, DEFAULT_TRAINING_PARAMS

__all__ = ['get_parser', 'parse_args', 'DEFAULT_MODEL_PARAMS', 'DEFAULT_TRAINING_PARAMS']
