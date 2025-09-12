"""Clean model loading utilities that suppress warnings."""

import sys
import os
import warnings
import contextlib
from io import StringIO
from typing import Any


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@contextlib.contextmanager
def suppress_stderr_only():
    """Context manager to suppress only stderr (warnings) but keep stdout."""
    old_stderr = sys.stderr
    try:
        sys.stderr = StringIO()
        yield
    finally:
        sys.stderr = old_stderr


@contextlib.contextmanager
def clean_model_loading():
    """Context manager for clean model loading without warnings."""
    # Set environment variables
    old_tokenizers = os.environ.get('TOKENIZERS_PARALLELISM', '')
    old_transformers = os.environ.get('TRANSFORMERS_VERBOSITY', '')
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    # Suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Suppress stderr for model loading
        with suppress_stderr_only():
            try:
                yield
            finally:
                # Restore environment variables
                if old_tokenizers:
                    os.environ['TOKENIZERS_PARALLELISM'] = old_tokenizers
                else:
                    os.environ.pop('TOKENIZERS_PARALLELISM', None)
                    
                if old_transformers:
                    os.environ['TRANSFORMERS_VERBOSITY'] = old_transformers
                else:
                    os.environ.pop('TRANSFORMERS_VERBOSITY', None)


def load_model_silently(model_class, model_name, **kwargs):
    """Load a model with all warnings suppressed."""
    with clean_model_loading():
        return model_class.from_pretrained(model_name, **kwargs)


def load_tokenizer_silently(tokenizer_class, model_name, **kwargs):
    """Load a tokenizer with all warnings suppressed."""
    with clean_model_loading():
        return tokenizer_class.from_pretrained(model_name, **kwargs)
