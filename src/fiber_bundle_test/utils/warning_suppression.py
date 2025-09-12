"""Utilities for suppressing common warnings in the framework."""

import warnings
import logging
from contextlib import contextmanager
from typing import List, Optional

logger = logging.getLogger(__name__)


@contextmanager
def suppress_warnings(warning_patterns: Optional[List[str]] = None):
    """
    Context manager to suppress common warnings.
    
    Args:
        warning_patterns: List of warning message patterns to suppress
    """
    if warning_patterns is None:
        warning_patterns = get_default_warning_patterns()
    
    with warnings.catch_warnings():
        for pattern in warning_patterns:
            warnings.filterwarnings("ignore", message=pattern)
        yield


def get_default_warning_patterns() -> List[str]:
    """Get default warning patterns to suppress."""
    return [
        # Transformer model warnings
        "A parameter name that contains `beta` will be renamed internally to `bias`",
        "A parameter name that contains `gamma` will be renamed internally to `weight`",
        "Some weights of.*were not initialized from the model checkpoint",
        "You should probably TRAIN this model on a down-stream task",
        "The tokenizer class you load from this checkpoint",
        "The argument `trust_remote_code` is to be used with Auto classes",
        
        # UMAP warnings
        "n_jobs value.*overridden.*by setting random_state",
        
        # HuggingFace datasets warnings
        "The repository for.*contains custom code",
        "Loading a dataset that was saved using",
        
        # PyTorch warnings
        "Was asked to gather along dimension 0",
        "The given NumPy array is not writeable",
        
        # Plotly warnings
        "The dash_html_components package is deprecated",
        
        # General ML warnings
        "X does not have valid feature names",
        "The default value of `n_init` will change",
    ]


def configure_clean_logging():
    """Configure logging to reduce noise while keeping important information."""
    # Set specific loggers to WARNING level to reduce noise
    noisy_loggers = [
        'transformers.tokenization_utils_base',
        'transformers.configuration_utils',
        'transformers.modeling_utils',
        'datasets.builder',
        'datasets.info',
        'urllib3.connectionpool'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def setup_clean_environment():
    """Setup a clean environment with minimal warnings."""
    # Configure logging
    configure_clean_logging()
    
    # Set environment variables to reduce warnings
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer parallelism warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Reduce transformers verbosity
    
    # Suppress warnings globally including transformers internals
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*parameter name.*contains.*")
    warnings.filterwarnings("ignore", message=".*beta.*renamed.*bias.*")
    warnings.filterwarnings("ignore", message=".*gamma.*renamed.*weight.*")
    warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
    warnings.filterwarnings("ignore", message="You should probably TRAIN this model")
    
    # Set transformers logging to error level
    import transformers
    transformers.logging.set_verbosity_error()
    
    # Suppress specific transformers warnings
    import transformers.modeling_utils
    transformers.modeling_utils.logger.setLevel(logging.ERROR)


# Decorator for clean function execution
def clean_execution(func):
    """Decorator to execute function with suppressed warnings."""
    def wrapper(*args, **kwargs):
        with suppress_warnings():
            return func(*args, **kwargs)
    return wrapper


# Context manager for specific warning types
@contextmanager
def suppress_transformer_warnings():
    """Suppress transformer-specific warnings."""
    transformer_patterns = [
        "A parameter name that contains.*will be renamed internally",
        "Some weights of.*were not initialized from the model checkpoint",
        "You should probably TRAIN this model on a down-stream task",
        "The tokenizer class you load from this checkpoint"
    ]
    
    with suppress_warnings(transformer_patterns):
        yield


@contextmanager
def suppress_umap_warnings():
    """Suppress UMAP-specific warnings."""
    umap_patterns = [
        "n_jobs value.*overridden.*by setting random_state"
    ]
    
    with suppress_warnings(umap_patterns):
        yield
