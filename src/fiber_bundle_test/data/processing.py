"""Scalable processing utilities for large datasets and modern LLMs."""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Iterator, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import pickle
import json
import tempfile
import shutil

import numpy as np
from tqdm import tqdm

# Optional dependencies
try:
    import torch
    import torch.multiprocessing as mp
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import dask
    from dask import delayed as dask_delayed
    from dask.distributed import Client, as_completed as dask_as_completed
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for scalable processing."""
    batch_size: int = 32
    max_workers: int = 4
    use_gpu: bool = True
    memory_limit: str = "8GB"
    cache_dir: Optional[str] = None
    checkpoint_interval: int = 100
    resume_from_checkpoint: bool = True
    distributed: bool = False
    scheduler_address: Optional[str] = None


class MemoryManager:
    """Manage memory usage during processing."""
    
    def __init__(self, memory_limit: str = "8GB"):
        """
        Initialize memory manager.
        
        Args:
            memory_limit: Memory limit (e.g., "8GB", "4GB")
        """
        self.memory_limit = self._parse_memory_limit(memory_limit)
        self.current_usage = 0
    
    def _parse_memory_limit(self, memory_limit: str) -> int:
        """Parse memory limit string to bytes."""
        if memory_limit.endswith('GB'):
            return int(memory_limit[:-2]) * 1024 * 1024 * 1024
        elif memory_limit.endswith('MB'):
            return int(memory_limit[:-2]) * 1024 * 1024
        else:
            return int(memory_limit)
    
    def can_process_batch(self, batch_size: int, item_size: int) -> bool:
        """Check if we can process a batch without exceeding memory limit."""
        estimated_usage = batch_size * item_size
        return self.current_usage + estimated_usage < self.memory_limit
    
    def update_usage(self, size: int):
        """Update current memory usage."""
        self.current_usage += size
    
    def free_memory(self, size: int):
        """Free memory."""
        self.current_usage = max(0, self.current_usage - size)


class CheckpointManager:
    """Manage checkpoints for resumable processing."""
    
    def __init__(self, cache_dir: str, experiment_name: str):
        """
        Initialize checkpoint manager.
        
        Args:
            cache_dir: Directory to store checkpoints
            experiment_name: Name of the experiment
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.checkpoint_file = self.cache_dir / f"{experiment_name}_checkpoint.pkl"
    
    def save_checkpoint(self, state: Dict[str, Any]):
        """Save checkpoint state."""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Checkpoint saved: {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint state."""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            logger.info(f"Checkpoint loaded: {self.checkpoint_file}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self):
        """Clear checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


class ScalableEmbeddingProcessor:
    """Process embeddings at scale with checkpointing and memory management."""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize scalable processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.memory_manager = MemoryManager(config.memory_limit)
        
        if config.cache_dir:
            self.checkpoint_manager = CheckpointManager(
                config.cache_dir, 
                f"embedding_processing_{int(time.time())}"
            )
        else:
            self.checkpoint_manager = None
    
    def process_large_dataset(self, 
                            extractor: Any,
                            data_iterator: Iterator[Dict[str, Any]],
                            total_items: Optional[int] = None) -> np.ndarray:
        """
        Process a large dataset with checkpointing and memory management.
        
        Args:
            extractor: Embedding extractor
            data_iterator: Iterator over data items
            total_items: Total number of items (for progress bar)
            
        Returns:
            Array of embeddings
        """
        # Try to resume from checkpoint
        start_idx = 0
        embeddings_list = []
        
        if self.config.resume_from_checkpoint and self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                start_idx = checkpoint['processed_count']
                embeddings_list = checkpoint['embeddings_list']
                logger.info(f"Resuming from item {start_idx}")
        
        # Process data in batches
        batch = []
        processed_count = start_idx
        
        progress_bar = tqdm(
            total=total_items,
            initial=start_idx,
            desc="Processing embeddings"
        )
        
        try:
            # Skip already processed items
            for _ in range(start_idx):
                next(data_iterator, None)
            
            for item in data_iterator:
                batch.append(item)
                
                if len(batch) >= self.config.batch_size:
                    # Process batch
                    batch_embeddings = self._process_batch(extractor, batch)
                    embeddings_list.extend(batch_embeddings)
                    
                    processed_count += len(batch)
                    progress_bar.update(len(batch))
                    
                    # Save checkpoint
                    if (self.checkpoint_manager and 
                        processed_count % self.config.checkpoint_interval == 0):
                        self._save_checkpoint(processed_count, embeddings_list)
                    
                    # Clear batch
                    batch = []
                    
                    # Memory management
                    if HAS_TORCH and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Process remaining items
            if batch:
                batch_embeddings = self._process_batch(extractor, batch)
                embeddings_list.extend(batch_embeddings)
                processed_count += len(batch)
                progress_bar.update(len(batch))
        
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            # Save checkpoint before failing
            if self.checkpoint_manager:
                self._save_checkpoint(processed_count, embeddings_list)
            raise
        
        finally:
            progress_bar.close()
        
        # Clear checkpoint on successful completion
        if self.checkpoint_manager:
            self.checkpoint_manager.clear_checkpoint()
        
        return np.array(embeddings_list)
    
    def _process_batch(self, extractor: Any, batch: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Process a batch of items."""
        try:
            # Extract texts and tokens
            texts = []
            target_tokens = []
            
            for item in batch:
                if 'sentence' in item:
                    texts.append(item['sentence'])
                    target_tokens.append(item.get('token', ''))
                elif 'text' in item:
                    texts.append(item['text'])
                    target_tokens.append(item.get('token', ''))
                else:
                    texts.append(str(item))
                    target_tokens.append('')
            
            # Get embeddings
            if any(target_tokens):
                embeddings = extractor.get_embeddings(texts, target_tokens)
            else:
                embeddings = extractor.get_embeddings(texts)
            
            return [emb for emb in embeddings]
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return zero embeddings for failed batch
            embedding_dim = getattr(extractor, 'embedding_dim', 768)
            return [np.zeros(embedding_dim) for _ in batch]
    
    def _save_checkpoint(self, processed_count: int, embeddings_list: List[np.ndarray]):
        """Save processing checkpoint."""
        if self.checkpoint_manager:
            checkpoint = {
                'processed_count': processed_count,
                'embeddings_list': embeddings_list,
                'timestamp': time.time()
            }
            self.checkpoint_manager.save_checkpoint(checkpoint)


class DistributedProcessor:
    """Distributed processing using Dask or multiprocessing."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize distributed processor."""
        self.config = config
        self.client = None
        
        if config.distributed and HAS_DASK:
            self._setup_dask_client()
    
    def _setup_dask_client(self):
        """Setup Dask client for distributed processing."""
        try:
            if self.config.scheduler_address:
                self.client = Client(self.config.scheduler_address)
            else:
                self.client = Client(processes=True, n_workers=self.config.max_workers)
            
            logger.info(f"Dask client initialized: {self.client}")
        except Exception as e:
            logger.warning(f"Failed to setup Dask client: {e}")
            self.client = None
    
    def process_distributed(self, 
                          process_func: Callable,
                          data_chunks: List[Any],
                          **kwargs) -> List[Any]:
        """
        Process data chunks in a distributed manner.
        
        Args:
            process_func: Function to process each chunk
            data_chunks: List of data chunks
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of processed results
        """
        if self.client and HAS_DASK:
            return self._process_with_dask(process_func, data_chunks, **kwargs)
        elif HAS_JOBLIB:
            return self._process_with_joblib(process_func, data_chunks, **kwargs)
        else:
            return self._process_with_multiprocessing(process_func, data_chunks, **kwargs)
    
    def _process_with_dask(self, process_func: Callable, data_chunks: List[Any], **kwargs) -> List[Any]:
        """Process using Dask."""
        futures = []
        
        for chunk in data_chunks:
            future = self.client.submit(process_func, chunk, **kwargs)
            futures.append(future)
        
        results = []
        for future in tqdm(dask_as_completed(futures), total=len(futures), desc="Processing chunks"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                results.append(None)
        
        return results
    
    def _process_with_joblib(self, process_func: Callable, data_chunks: List[Any], **kwargs) -> List[Any]:
        """Process using Joblib."""
        results = Parallel(n_jobs=self.config.max_workers, verbose=1)(
            delayed(process_func)(chunk, **kwargs) for chunk in data_chunks
        )
        return results
    
    def _process_with_multiprocessing(self, process_func: Callable, data_chunks: List[Any], **kwargs) -> List[Any]:
        """Process using multiprocessing."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(process_func, chunk, **kwargs): i 
                for i, chunk in enumerate(data_chunks)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    results.append(None)
        
        return results
    
    def close(self):
        """Close distributed client."""
        if self.client:
            self.client.close()


class BatchProcessor:
    """Process data in optimized batches."""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Size of each batch
            max_workers: Maximum number of worker threads
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_in_batches(self, 
                          data: List[Any],
                          process_func: Callable,
                          **kwargs) -> List[Any]:
        """
        Process data in batches.
        
        Args:
            data: List of data items
            process_func: Function to process each batch
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of processed results
        """
        batches = [
            data[i:i + self.batch_size] 
            for i in range(0, len(data), self.batch_size)
        ]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_func, batch, **kwargs): i 
                for i, batch in enumerate(batches)
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        
        return results


class OptimizedHypothesisTest:
    """Optimized version of fiber bundle hypothesis test for large datasets."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize optimized test."""
        self.config = config
        self.processor = ScalableEmbeddingProcessor(config)
    
    def run_large_scale_test(self, 
                           embeddings: np.ndarray,
                           test_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run hypothesis test on large-scale embeddings.
        
        Args:
            embeddings: Large array of embeddings
            test_params: Test parameters
            
        Returns:
            Test results
        """
        from ..core import FiberBundleTest
        
        # Initialize test
        test = FiberBundleTest(**test_params)
        
        # Process in chunks if embeddings are too large
        max_chunk_size = 10000  # Adjust based on memory
        
        if len(embeddings) <= max_chunk_size:
            return test.run_test(embeddings)
        
        # Process in chunks
        results_list = []
        chunk_size = max_chunk_size
        
        for i in tqdm(range(0, len(embeddings), chunk_size), desc="Processing chunks"):
            chunk = embeddings[i:i + chunk_size]
            chunk_results = test.run_test(chunk, verbose=False)
            results_list.append(chunk_results)
        
        # Combine results
        combined_results = self._combine_chunk_results(results_list)
        return combined_results
    
    def _combine_chunk_results(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple chunks."""
        combined = {
            'results': [],
            'dimensions': [],
            'raw_p_values': [],
            'total_rejections': 0,
            'total_tokens': 0
        }
        
        token_offset = 0
        
        for chunk_results in results_list:
            # Adjust token indices
            adjusted_results = []
            for token_idx, decision in chunk_results['results']:
                adjusted_results.append((token_idx + token_offset, decision))
            
            combined['results'].extend(adjusted_results)
            combined['dimensions'].extend(chunk_results['dimensions'])
            combined['raw_p_values'].extend(chunk_results['raw_p_values'])
            combined['total_rejections'] += chunk_results['total_rejections']
            combined['total_tokens'] += chunk_results['total_tokens']
            
            token_offset += chunk_results['total_tokens']
        
        combined['rejection_rate'] = combined['total_rejections'] / combined['total_tokens']
        
        return combined


# Convenience functions
def create_processing_config(**kwargs) -> ProcessingConfig:
    """Create processing configuration with defaults."""
    return ProcessingConfig(**kwargs)


def process_large_scale_embeddings(extractor: Any,
                                 data_iterator: Iterator[Dict[str, Any]],
                                 config: Optional[ProcessingConfig] = None) -> np.ndarray:
    """
    Process large-scale embeddings with optimizations.
    
    Args:
        extractor: Embedding extractor
        data_iterator: Data iterator
        config: Processing configuration
        
    Returns:
        Array of embeddings
    """
    if config is None:
        config = ProcessingConfig()
    
    processor = ScalableEmbeddingProcessor(config)
    return processor.process_large_dataset(extractor, data_iterator)
