"""
Fiber Bundle Hypothesis Test on LLM Embeddings

This package implements a framework to test the "fiber bundle hypothesis" 
on token embeddings derived from pre-trained language models, based on 
the methodology described in "Token Embeddings Violate the Manifold Hypothesis".
"""

__version__ = "1.0.0"
__author__ = "Fiber Bundle Research Team"

from .core import FiberBundleTest
from .embeddings import BERTEmbeddingExtractor
from .embeddings.modern_llms import ModernLLMExtractor
from .embeddings.roberta_embeddings import RoBERTaEmbeddingExtractor
from .visualization import ResultsVisualizer
from .data.datasets import create_large_scale_dataset, load_multidomain_sentiment
from .data.processing import ProcessingConfig, ScalableEmbeddingProcessor
from .models import MixtureOfDictionaryExperts, DictionaryExpertLISTA
from .training import ContrastiveTrainer, TrainingConfig
from .analysis import StratificationAnalyzer, ClusteringAnalyzer, DimensionalityAnalyzer

__all__ = [
    "FiberBundleTest",
    "BERTEmbeddingExtractor", 
    "RoBERTaEmbeddingExtractor",
    "ModernLLMExtractor",
    "ResultsVisualizer",
    "create_large_scale_dataset",
    "load_multidomain_sentiment",
    "ProcessingConfig",
    "ScalableEmbeddingProcessor",
    "MixtureOfDictionaryExperts",
    "DictionaryExpertLISTA",
    "ContrastiveTrainer",
    "TrainingConfig",
    "StratificationAnalyzer",
    "ClusteringAnalyzer",
    "DimensionalityAnalyzer"
]
