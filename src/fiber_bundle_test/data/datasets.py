"""Large-scale dataset loaders for realistic text data."""

import os
import json
import gzip
import pickle
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging
import random
from collections import defaultdict
import re

import numpy as np
import pandas as pd

# Optional dependencies for large datasets
try:
    import datasets
    from datasets import Dataset, load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import wikipedia
    HAS_WIKIPEDIA = True
except ImportError:
    HAS_WIKIPEDIA = False

try:
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

logger = logging.getLogger(__name__)


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
            seed: Random seed for reproducibility
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'fiber_bundle'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    @abstractmethod
    def load_data(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Load data from the dataset."""
        pass
    
    @abstractmethod
    def get_sample(self, n_samples: int, **kwargs) -> List[Dict[str, Any]]:
        """Get a sample of data."""
        pass


class WikipediaDataset(BaseDatasetLoader):
    """Loader for Wikipedia articles."""
    
    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42, language: str = 'en'):
        """
        Initialize Wikipedia dataset loader.
        
        Args:
            cache_dir: Cache directory
            seed: Random seed
            language: Wikipedia language code
        """
        super().__init__(cache_dir, seed)
        self.language = language
        
        if not HAS_WIKIPEDIA:
            logger.warning("wikipedia-api not available, using fallback method")
    
    def load_data(self, topics: Optional[List[str]] = None, 
                  max_articles: int = 1000) -> Iterator[Dict[str, Any]]:
        """
        Load Wikipedia articles.
        
        Args:
            topics: List of topics to search for
            max_articles: Maximum number of articles to load
            
        Yields:
            Dictionary with article data
        """
        if not HAS_WIKIPEDIA:
            # Fallback to sample data
            yield from self._get_sample_wikipedia_data(max_articles)
            return
        
        wikipedia.set_lang(self.language)
        
        if topics is None:
            topics = self._get_default_topics()
        
        articles_loaded = 0
        
        for topic in topics:
            if articles_loaded >= max_articles:
                break
            
            try:
                # Search for articles related to topic
                search_results = wikipedia.search(topic, results=min(20, max_articles - articles_loaded))
                
                for title in search_results:
                    if articles_loaded >= max_articles:
                        break
                    
                    try:
                        page = wikipedia.page(title)
                        
                        yield {
                            'title': page.title,
                            'content': page.content,
                            'url': page.url,
                            'topic': topic,
                            'summary': page.summary,
                            'length': len(page.content),
                            'sentences': self._extract_sentences(page.content)
                        }
                        
                        articles_loaded += 1
                        
                    except wikipedia.exceptions.DisambiguationError as e:
                        # Try the first option
                        try:
                            page = wikipedia.page(e.options[0])
                            yield {
                                'title': page.title,
                                'content': page.content,
                                'url': page.url,
                                'topic': topic,
                                'summary': page.summary,
                                'length': len(page.content),
                                'sentences': self._extract_sentences(page.content)
                            }
                            articles_loaded += 1
                        except:
                            continue
                    except:
                        continue
                        
            except Exception as e:
                logger.warning(f"Error loading topic '{topic}': {e}")
                continue
    
    def get_sample(self, n_samples: int, topics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get a sample of Wikipedia articles."""
        articles = list(self.load_data(topics=topics, max_articles=n_samples * 2))
        return random.sample(articles, min(n_samples, len(articles)))
    
    def _get_default_topics(self) -> List[str]:
        """Get default topics for sampling."""
        return [
            "artificial intelligence", "machine learning", "computer science",
            "physics", "mathematics", "biology", "chemistry", "history",
            "literature", "philosophy", "psychology", "economics",
            "politics", "geography", "astronomy", "medicine", "engineering"
        ]
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        if HAS_NLTK:
            try:
                nltk.download('punkt', quiet=True)
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _get_sample_wikipedia_data(self, max_articles: int) -> Iterator[Dict[str, Any]]:
        """Fallback sample data when Wikipedia API is not available."""
        sample_articles = [
            {
                'title': 'Artificial Intelligence',
                'content': 'Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by humans and animals. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.',
                'topic': 'technology',
                'summary': 'AI is machine intelligence',
                'length': 250,
                'sentences': ['Artificial intelligence (AI) is intelligence demonstrated by machines.', 'AI research studies intelligent agents.']
            },
            {
                'title': 'Machine Learning',
                'content': 'Machine learning (ML) is a type of artificial intelligence that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values.',
                'topic': 'technology',
                'summary': 'ML enables prediction without explicit programming',
                'length': 200,
                'sentences': ['Machine learning is a type of AI.', 'ML algorithms use historical data for predictions.']
            }
        ]
        
        for i, article in enumerate(sample_articles):
            if i >= max_articles:
                break
            yield article


class HuggingFaceDataset(BaseDatasetLoader):
    """Loader for Hugging Face datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42):
        """Initialize Hugging Face dataset loader."""
        super().__init__(cache_dir, seed)
        
        if not HAS_DATASETS:
            raise ImportError("datasets library is required for HuggingFace datasets")
    
    def load_data(self, dataset_name: str, split: str = 'train', 
                  streaming: bool = True, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Load data from a Hugging Face dataset.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split to load
            streaming: Whether to use streaming mode
            **kwargs: Additional arguments for load_dataset
            
        Yields:
            Dictionary with data
        """
        try:
            dataset = load_dataset(
                dataset_name, 
                split=split,
                streaming=streaming,
                cache_dir=str(self.cache_dir),
                **kwargs
            )
            
            for item in dataset:
                yield item
                
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def get_sample(self, n_samples: int, dataset_name: str, 
                   split: str = 'train', **kwargs) -> List[Dict[str, Any]]:
        """Get a sample from a Hugging Face dataset."""
        samples = []
        
        for i, item in enumerate(self.load_data(dataset_name, split, **kwargs)):
            if i >= n_samples:
                break
            samples.append(item)
        
        return samples
    
    @staticmethod
    def list_popular_datasets() -> Dict[str, str]:
        """List popular text datasets available on Hugging Face."""
        return {
            'c4': 'allenai/c4',
            'openwebtext': 'openwebtext',
            'bookcorpus': 'bookcorpus',
            'wikipedia': 'wikipedia',
            'common_crawl': 'cc_news',
            'reddit': 'reddit',
            'arxiv': 'arxiv_dataset',
            'pubmed': 'pubmed_qa',
            'gutenberg': 'pg19',
            'news': 'cnn_dailymail',
            'reviews': 'amazon_reviews_multi',
            'social_media': 'tweet_eval',
        }


class CommonCrawlDataset(BaseDatasetLoader):
    """Loader for Common Crawl data (simplified version)."""
    
    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42):
        """Initialize Common Crawl dataset loader."""
        super().__init__(cache_dir, seed)
    
    def load_data(self, crawl_id: Optional[str] = None, 
                  max_files: int = 10) -> Iterator[Dict[str, Any]]:
        """
        Load Common Crawl data.
        
        Note: This is a simplified version. For actual Common Crawl data,
        you would need to download and process WARC files.
        """
        # For now, use a proxy dataset or sample data
        if HAS_DATASETS:
            try:
                # Use CC-News as a proxy for Common Crawl
                dataset = load_dataset('cc_news', split='train', streaming=True)
                
                for item in dataset:
                    yield {
                        'url': item.get('url', ''),
                        'title': item.get('title', ''),
                        'text': item.get('text', ''),
                        'domain': self._extract_domain(item.get('url', '')),
                        'length': len(item.get('text', '')),
                        'sentences': self._extract_sentences(item.get('text', ''))
                    }
            except:
                yield from self._get_sample_web_data()
        else:
            yield from self._get_sample_web_data()
    
    def get_sample(self, n_samples: int, **kwargs) -> List[Dict[str, Any]]:
        """Get a sample of Common Crawl data."""
        samples = []
        
        for i, item in enumerate(self.load_data(**kwargs)):
            if i >= n_samples:
                break
            samples.append(item)
        
        return samples
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return ''
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return ''
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        if HAS_NLTK:
            try:
                nltk.download('punkt', quiet=True)
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _get_sample_web_data(self) -> Iterator[Dict[str, Any]]:
        """Sample web data when real Common Crawl is not available."""
        sample_data = [
            {
                'url': 'https://example.com/article1',
                'title': 'The Future of Technology',
                'text': 'Technology continues to evolve at a rapid pace. Artificial intelligence and machine learning are transforming industries across the globe. From healthcare to finance, these technologies are creating new opportunities and challenges.',
                'domain': 'example.com',
                'length': 180,
                'sentences': ['Technology continues to evolve at a rapid pace.', 'AI and ML are transforming industries.']
            },
            {
                'url': 'https://news.example.org/science',
                'title': 'Climate Change Research',
                'text': 'Recent studies show significant changes in global climate patterns. Scientists are working to understand the long-term implications of these changes. International cooperation is essential for addressing climate challenges.',
                'domain': 'news.example.org',
                'length': 170,
                'sentences': ['Recent studies show climate pattern changes.', 'Scientists study long-term implications.']
            }
        ]
        
        for item in sample_data:
            yield item


class MultiSourceDataset(BaseDatasetLoader):
    """Combine multiple data sources."""
    
    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42):
        """Initialize multi-source dataset loader."""
        super().__init__(cache_dir, seed)
        self.loaders = {}
    
    def add_source(self, name: str, loader: BaseDatasetLoader, weight: float = 1.0):
        """Add a data source."""
        self.loaders[name] = {'loader': loader, 'weight': weight}
    
    def load_data(self, max_samples_per_source: int = 1000) -> Iterator[Dict[str, Any]]:
        """Load data from all sources."""
        all_samples = []
        
        for name, config in self.loaders.items():
            loader = config['loader']
            weight = config['weight']
            
            try:
                samples = loader.get_sample(int(max_samples_per_source * weight))
                for sample in samples:
                    sample['source'] = name
                    all_samples.append(sample)
            except Exception as e:
                logger.warning(f"Error loading from source '{name}': {e}")
        
        # Shuffle all samples
        random.shuffle(all_samples)
        
        for sample in all_samples:
            yield sample
    
    def get_sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """Get a mixed sample from all sources."""
        all_samples = list(self.load_data())
        return random.sample(all_samples, min(n_samples, len(all_samples)))


class TokenizedDataset:
    """Dataset specifically designed for token-level analysis."""
    
    def __init__(self, base_dataset: BaseDatasetLoader, target_tokens: List[str]):
        """
        Initialize tokenized dataset.
        
        Args:
            base_dataset: Base dataset loader
            target_tokens: List of target tokens to analyze
        """
        self.base_dataset = base_dataset
        self.target_tokens = target_tokens
        self.token_sentences = defaultdict(list)
    
    def build_token_dataset(self, max_samples_per_token: int = 1000) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build a dataset organized by target tokens.
        
        Args:
            max_samples_per_token: Maximum samples per token
            
        Returns:
            Dictionary mapping tokens to their sentences
        """
        logger.info("Building token dataset...")
        
        for data_item in self.base_dataset.load_data():
            text = data_item.get('content') or data_item.get('text', '')
            sentences = data_item.get('sentences') or self._extract_sentences(text)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                for token in self.target_tokens:
                    if len(self.token_sentences[token]) >= max_samples_per_token:
                        continue
                    
                    if token.lower() in sentence_lower:
                        self.token_sentences[token].append({
                            'sentence': sentence,
                            'token': token,
                            'source': data_item.get('source', 'unknown'),
                            'title': data_item.get('title', ''),
                            'length': len(sentence)
                        })
        
        # If no sentences found, create sample sentences
        total_sentences = sum(len(sents) for sents in self.token_sentences.values())
        if total_sentences == 0:
            logger.warning("No sentences found in dataset, creating sample sentences")
            for token in self.target_tokens:
                sample_sentences = [
                    f"The {token} is an important concept in this analysis.",
                    f"We need to understand how {token} affects the results.",
                    f"This sentence contains the word {token} for testing purposes.",
                    f"The study of {token} reveals interesting patterns.",
                    f"Research shows that {token} plays a crucial role."
                ]
                
                for sentence in sample_sentences[:max_samples_per_token]:
                    self.token_sentences[token].append({
                        'sentence': sentence,
                        'token': token,
                        'source': 'sample',
                        'title': 'Sample Data',
                        'length': len(sentence)
                    })
        
        total_sentences = sum(len(sents) for sents in self.token_sentences.values())
        logger.info(f"Built dataset with {total_sentences} sentences")
        return dict(self.token_sentences)
    
    def get_balanced_sample(self, n_per_token: int) -> Tuple[List[str], List[str]]:
        """
        Get a balanced sample of sentences and tokens.
        
        Args:
            n_per_token: Number of samples per token
            
        Returns:
            Tuple of (sentences, target_tokens)
        """
        sentences = []
        tokens = []
        
        for token, token_sentences in self.token_sentences.items():
            sample_size = min(n_per_token, len(token_sentences))
            sampled = random.sample(token_sentences, sample_size)
            
            for item in sampled:
                sentences.append(item['sentence'])
                tokens.append(token)
        
        return sentences, tokens
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        if HAS_NLTK:
            try:
                nltk.download('punkt', quiet=True)
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]


# Convenience functions
def create_dataset_loader(dataset_type: str, **kwargs) -> BaseDatasetLoader:
    """Create a dataset loader of the specified type."""
    loaders = {
        'wikipedia': WikipediaDataset,
        'huggingface': HuggingFaceDataset,
        'common_crawl': CommonCrawlDataset,
        'multi_source': MultiSourceDataset
    }
    
    if dataset_type not in loaders:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    return loaders[dataset_type](**kwargs)


def create_large_scale_dataset(target_tokens: List[str], 
                              n_samples_per_token: int = 1000,
                              sources: List[str] = None) -> Tuple[List[str], List[str]]:
    """
    Create a large-scale dataset for fiber bundle analysis.
    
    Args:
        target_tokens: List of target tokens to analyze
        n_samples_per_token: Number of samples per token
        sources: List of data sources to use
        
    Returns:
        Tuple of (sentences, target_tokens)
    """
    if sources is None:
        sources = ['wikipedia', 'huggingface']
    
    # Create multi-source dataset
    multi_dataset = MultiSourceDataset()
    
    for source in sources:
        try:
            if source == 'wikipedia':
                loader = WikipediaDataset()
                multi_dataset.add_source('wikipedia', loader, weight=0.4)
            elif source == 'huggingface':
                loader = HuggingFaceDataset()
                multi_dataset.add_source('huggingface', loader, weight=0.6)
            elif source == 'common_crawl':
                loader = CommonCrawlDataset()
                multi_dataset.add_source('common_crawl', loader, weight=0.3)
        except Exception as e:
            logger.warning(f"Failed to add source '{source}': {e}")
    
    # Build tokenized dataset
    tokenized_dataset = TokenizedDataset(multi_dataset, target_tokens)
    tokenized_dataset.build_token_dataset(n_samples_per_token)
    
    return tokenized_dataset.get_balanced_sample(n_samples_per_token)
"""Multi-domain dataset loading and processing from the experiments notebook."""

import logging
from typing import List, Dict, Any, Optional
from datasets import load_dataset, concatenate_datasets, Value, Dataset
import numpy as np

logger = logging.getLogger(__name__)


def unify_dataset(ds: Dataset, domain_name: str, samples_per_domain: int = 100, 
                 text_field: str = "text") -> Dataset:
    """
    Unify dataset format across different sources.
    
    Args:
        ds: Input dataset
        domain_name: Name of the domain
        samples_per_domain: Number of samples to keep
        text_field: Name of the text field
        
    Returns:
        Unified dataset
    """
    # Ensure text field is named 'text'
    if text_field != "text":
        ds = ds.map(lambda x: {"text": x[text_field], "label": x["label"]})
    
    # Keep only text and label columns
    keep_cols = ["text", "label"]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(remove_cols)
    
    # Ensure label is integer
    ds = ds.map(lambda x: {"label": int(x["label"])})
    ds = ds.cast_column("label", Value("int64"))
    
    # Limit samples
    ds_small = ds.select(range(min(samples_per_domain, len(ds))))
    
    # Add domain column
    ds_small = ds_small.add_column("domain", [domain_name] * len(ds_small))
    
    return ds_small


def load_multidomain_sentiment(samples_per_domain: int = 100) -> Dataset:
    """
    Load and unify multiple sentiment analysis datasets.
    
    This function loads six diverse text datasets:
    - IMDB (long movie reviews)
    - Rotten Tomatoes (moderate-length reviews)  
    - Amazon Polarity (product reviews)
    - GLUE/SST2 (Stanford sentiment sentences)
    - TweetEval (social media sentiment)
    - AG News (news articles)
    
    Args:
        samples_per_domain: Number of samples per domain
        
    Returns:
        Concatenated dataset with unified format
    """
    datasets_list = []
    
    try:
        # IMDB dataset
        logger.info("Loading IMDB dataset...")
        imdb_ds = unify_dataset(
            load_dataset("imdb", split=f"train[:{samples_per_domain}]"), 
            "imdb", 
            samples_per_domain
        )
        datasets_list.append(imdb_ds)
        logger.info(f"✓ IMDB: {len(imdb_ds)} samples")
        
    except Exception as e:
        logger.warning(f"Failed to load IMDB: {e}")
    
    try:
        # Rotten Tomatoes dataset
        logger.info("Loading Rotten Tomatoes dataset...")
        rt_ds = unify_dataset(
            load_dataset("rotten_tomatoes", split=f"train[:{samples_per_domain}]"), 
            "rotten", 
            samples_per_domain
        )
        datasets_list.append(rt_ds)
        logger.info(f"✓ Rotten Tomatoes: {len(rt_ds)} samples")
        
    except Exception as e:
        logger.warning(f"Failed to load Rotten Tomatoes: {e}")
    
    try:
        # Amazon Polarity dataset
        logger.info("Loading Amazon Polarity dataset...")
        ap_raw = load_dataset("amazon_polarity", split=f"train[:{int(2 * samples_per_domain)}]")
        ap_raw = ap_raw.map(lambda x: {"text": f"{x['title']} {x['content']}".strip()})
        ap_ds = unify_dataset(ap_raw, "amazon", samples_per_domain)
        datasets_list.append(ap_ds)
        logger.info(f"✓ Amazon Polarity: {len(ap_ds)} samples")
        
    except Exception as e:
        logger.warning(f"Failed to load Amazon Polarity: {e}")
    
    try:
        # SST2 dataset
        logger.info("Loading SST2 dataset...")
        sst2_ds = load_dataset("glue", "sst2", split=f"train[:{samples_per_domain}]")
        sst2_ds = unify_dataset(sst2_ds, "sst2", samples_per_domain, text_field="sentence")
        datasets_list.append(sst2_ds)
        logger.info(f"✓ SST2: {len(sst2_ds)} samples")
        
    except Exception as e:
        logger.warning(f"Failed to load SST2: {e}")
    
    try:
        # TweetEval dataset
        logger.info("Loading TweetEval dataset...")
        tweet_ds = load_dataset("tweet_eval", "sentiment", split=f"train[:{samples_per_domain}]")
        tweet_ds = unify_dataset(tweet_ds, "tweet", samples_per_domain, text_field="text")
        datasets_list.append(tweet_ds)
        logger.info(f"✓ TweetEval: {len(tweet_ds)} samples")
        
    except Exception as e:
        logger.warning(f"Failed to load TweetEval: {e}")
    
    try:
        # AG News dataset
        logger.info("Loading AG News dataset...")
        ag_news_ds = load_dataset("ag_news", split=f"train[:{samples_per_domain}]")
        ag_news_ds = unify_dataset(ag_news_ds, "ag_news", samples_per_domain, text_field="text")
        datasets_list.append(ag_news_ds)
        logger.info(f"✓ AG News: {len(ag_news_ds)} samples")
        
    except Exception as e:
        logger.warning(f"Failed to load AG News: {e}")
    
    # If no datasets loaded successfully, create sample data
    if not datasets_list:
        logger.warning("No datasets loaded successfully, creating sample data")
        sample_data = create_sample_multidomain_data(samples_per_domain)
        datasets_list.append(sample_data)
    
    # Concatenate all datasets
    combined_dataset = concatenate_datasets(datasets_list)
    
    logger.info(f"✅ Combined dataset created with {len(combined_dataset)} total samples")
    logger.info(f"Domains: {list(set(combined_dataset['domain']))}")
    
    return combined_dataset


def create_sample_multidomain_data(samples_per_domain: int = 100) -> Dataset:
    """
    Create sample multi-domain data when real datasets are not available.
    
    Args:
        samples_per_domain: Number of samples per domain
        
    Returns:
        Sample dataset
    """
    sample_texts = {
        'imdb': [
            "This movie was absolutely fantastic and I loved every moment of it.",
            "The film was terrible and completely boring throughout.",
            "Great cinematography and excellent acting made this worth watching.",
            "Poor plot development and weak character arcs ruined the experience.",
            "One of the best movies I've seen this year, highly recommended."
        ],
        'amazon': [
            "Excellent product quality, fast shipping, very satisfied with purchase.",
            "Poor quality item, arrived damaged, would not recommend to others.",
            "Great value for money, works exactly as described in listing.",
            "Overpriced product with mediocre performance, disappointed with results.",
            "Outstanding customer service and high-quality product, will buy again."
        ],
        'news': [
            "Breaking news report reveals important developments in technology sector.",
            "Local community responds positively to new infrastructure improvements.",
            "Economic indicators show mixed signals for upcoming quarter.",
            "Scientific breakthrough promises significant advances in medical research.",
            "International cooperation leads to successful diplomatic resolution."
        ],
        'social': [
            "Love this new update, makes everything so much easier to use!",
            "Hate the recent changes, everything is confusing and broken now.",
            "Amazing experience at the concert last night, incredible performance!",
            "Disappointed with the service quality, definitely not coming back.",
            "Best day ever, everything went perfectly according to plan!"
        ]
    }
    
    all_texts = []
    all_labels = []
    all_domains = []
    
    for domain, texts in sample_texts.items():
        # Expand texts to reach samples_per_domain
        expanded_texts = (texts * (samples_per_domain // len(texts) + 1))[:samples_per_domain]
        expanded_labels = ([1, 0] * (samples_per_domain // 2 + 1))[:samples_per_domain]
        expanded_domains = [domain] * samples_per_domain
        
        all_texts.extend(expanded_texts)
        all_labels.extend(expanded_labels)
        all_domains.extend(expanded_domains)
    
    return Dataset.from_dict({
        "text": all_texts,
        "label": all_labels,
        "domain": all_domains
    })


def get_domain_statistics(dataset: Dataset) -> Dict[str, Any]:
    """
    Get statistics about the multi-domain dataset.
    
    Args:
        dataset: Multi-domain dataset
        
    Returns:
        Statistics dictionary
    """
    domains = dataset['domain']
    labels = dataset['label']
    texts = dataset['text']
    
    # Domain counts
    unique_domains, domain_counts = np.unique(domains, return_counts=True)
    domain_stats = dict(zip(unique_domains, domain_counts))
    
    # Label distribution per domain
    domain_label_stats = {}
    for domain in unique_domains:
        domain_mask = np.array(domains) == domain
        domain_labels = np.array(labels)[domain_mask]
        unique_labels, label_counts = np.unique(domain_labels, return_counts=True)
        domain_label_stats[domain] = dict(zip(unique_labels.tolist(), label_counts.tolist()))
    
    # Text length statistics
    text_lengths = [len(text.split()) for text in texts]
    
    return {
        'total_samples': len(dataset),
        'num_domains': len(unique_domains),
        'domain_distribution': domain_stats,
        'domain_label_distribution': domain_label_stats,
        'text_length_stats': {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths)
        }
    }
"""Text processing utilities for large-scale datasets."""

import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


class TextProcessor:
    """Text processing utilities for preprocessing large datasets."""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize text processor.
        
        Args:
            language: Language code for processing
        """
        self.language = language
        self._setup_processors()
    
    def _setup_processors(self):
        """Setup text processing tools."""
        if HAS_NLTK:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.stopwords = set(stopwords.words('english'))
            except:
                self.stopwords = set()
        else:
            self.stopwords = set()
        
        if HAS_SPACY:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                self.nlp = None
        else:
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        if HAS_NLTK:
            try:
                sentences = sent_tokenize(text)
                return [s.strip() for s in sentences if len(s.strip()) > 10]
            except:
                pass
        
        # Fallback sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if HAS_NLTK:
            try:
                return word_tokenize(text.lower())
            except:
                pass
        
        # Fallback tokenization
        return re.findall(r'\b\w+\b', text.lower())
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens."""
        return [token for token in tokens if token not in self.stopwords]
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from text."""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        
        # Filter by length and frequency
        keywords = [token for token in tokens if len(token) >= min_length]
        
        return list(set(keywords))  # Remove duplicates


# Convenience functions
def clean_text(text: str) -> str:
    """Clean text using default processor."""
    processor = TextProcessor()
    return processor.clean_text(text)


def extract_sentences(text: str) -> List[str]:
    """Extract sentences using default processor."""
    processor = TextProcessor()
    return processor.extract_sentences(text)
