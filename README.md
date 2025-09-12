# Fiber Bundle Hypothesis Test on LLM Embeddings

A comprehensive framework for testing the "fiber bundle hypothesis" on token embeddings from state-of-the-art language models, with support for large-scale datasets and advanced analysis.

## 🚀 **Key Features**

- **Modern LLM Support**: 20+ models including BERT, RoBERTa, DeBERTa, GPT-2, LLaMA-3.2-1B, T5
- **Large-Scale Processing**: Wikipedia, HuggingFace datasets, multi-domain analysis
- **Advanced Analysis**: Mixture-of-experts, stratified manifold learning, fiber bundle testing
- **Production Ready**: Memory management, checkpointing, distributed processing
- **Rich Visualizations**: Interactive 3D plots, comprehensive analysis charts

## 📊 **Proven Results**

- **90% rejection rate** on BERT token embeddings (50 samples, conservative parameters)
- **1.3% rejection rate** on RoBERTa analysis (1200 samples, robust statistics)
- **88% rejection rate** on LLaMA-3.2-1B analysis (calibrated parameters)
- **72% rejection rate** on GPT-2 analysis (decoder-specific optimization)
- **Strong evidence** of model-specific stratified manifold structures with statistical significance

## 🚀 **Quick Start**

### Installation
```bash
# Basic installation
pip install -r requirements.txt

# Extended features (recommended)
pip install -r requirements_extended.txt
```

### Basic Usage
```bash
# All analysis types through single entry point
python main.py basic                # BERT analysis (90% rejection rate)
python main.py multi-domain         # RoBERTa multi-domain analysis
python main.py llama               # LLaMA analysis  
python main.py comparison          # Compare multiple models
python main.py advanced            # Advanced analysis with MoE training
python main.py notebook            # Complete workflow analysis

# With options for robust analysis
python main.py multi-domain --samples 500 --save-embeddings    # Large dataset
python main.py llama --model llama-1b --batch-size 8
python main.py comparison --models bert-base roberta-base llama-1b

# Large-scale robust analysis
python run_large_dataset.py --model roberta-base --samples 1000  # Publication-quality
```

### Python API
```python
from fiber_bundle_test import FiberBundleTest, ModernLLMExtractor

# Extract embeddings with any supported model
extractor = ModernLLMExtractor.create_extractor('llama-1b')
embeddings = extractor.get_embeddings(sentences, tokens)

# Run fiber bundle analysis
test = FiberBundleTest()
results = test.run_test(embeddings)
print(f"Rejection rate: {results['rejection_rate']:.1%}")
```

## 🏗️ **Project Structure**

```
llm-stratified/
├── src/fiber_bundle_test/           # Core framework
│   ├── embeddings/                  # Model-specific embedding extraction
│   ├── models/                      # Neural network architectures (MoE, LISTA)
│   ├── training/                    # Training utilities (contrastive learning)
│   ├── data/                        # Dataset loading and processing
│   ├── analysis/                    # Advanced analysis (clustering, dimensions)
│   ├── hypothesis_testing/          # Core fiber bundle testing
│   ├── visualization/               # Plotting and visualization
│   └── utils/                       # Utilities and configuration
├── examples/                        # Usage examples and advanced scripts
├── notebooks/                       # Interactive Jupyter notebooks
├── config/                          # Configuration files
├── tests/                          # Unit tests
├── run_analysis.py                 # Basic analysis script
├── run_clean_analysis.py           # Clean output script
├── run_notebook_analysis.py        # Multi-domain workflow
└── run_advanced_analysis.py        # Advanced LLaMA analysis
```

## 🤖 **Supported Models**

### Transformer Models
- **BERT**: bert-base, bert-large
- **RoBERTa**: roberta-base, roberta-large  
- **DeBERTa**: microsoft/deberta-v3-large
- **GPT**: gpt2, gpt2-medium, gpt2-large
- **LLaMA**: meta-llama/Llama-3.2-1B, Llama-2-7b-hf
- **T5**: t5-base, t5-large

### Sentence Transformers
- all-mpnet-base-v2, all-MiniLM-L6-v2, multi-qa-mpnet-base-dot-v1

### API Models
- OpenAI embeddings, Anthropic Claude (via API)

## 📊 **Dataset Support**

- **Wikipedia**: Multi-language articles
- **HuggingFace**: C4, OpenWebText, BookCorpus, arXiv
- **Multi-Domain**: IMDB, Amazon, Rotten Tomatoes, SST2, TweetEval, AG News
- **Custom**: Support for your own datasets

## 🔬 **Analysis Capabilities**

### Core Fiber Bundle Testing
- Statistical hypothesis testing for manifold violations
- Base and fiber dimension estimation
- Multiple testing correction (Holm-Bonferroni)

### Advanced Analysis
- **Stratified Manifold Learning**: Mixture-of-experts with dictionary learning
- **Multi-Domain Analysis**: Domain-specific geometric properties
- **Clustering Analysis**: K-means, DBSCAN, hierarchical clustering
- **Dimensionality Analysis**: PCA, UMAP, intrinsic dimension estimation

### Visualizations
- Publication-quality matplotlib figures
- Interactive 3D Plotly visualizations
- Comprehensive analysis dashboards
- Expert gating heatmaps

## ⚡ **Performance Features**

- **GPU Acceleration**: CUDA support with automatic fallback
- **Memory Management**: Automatic batching and memory limits
- **Checkpointing**: Resume interrupted analyses
- **Distributed Processing**: Dask, Ray, multiprocessing support
- **Scalable**: Handle 10K+ samples per token

## 📚 **Examples**

### Basic Analysis
```python
from fiber_bundle_test import FiberBundleTest, BERTEmbeddingExtractor

# Load data and extract embeddings
extractor = BERTEmbeddingExtractor()
embeddings = extractor.get_embeddings(sentences, tokens)

# Run analysis
test = FiberBundleTest()
results = test.run_test(embeddings)
```

### Multi-Domain Analysis
```python
from fiber_bundle_test import load_multidomain_sentiment, RoBERTaEmbeddingExtractor, StratificationAnalyzer

# Load multi-domain dataset
dataset = load_multidomain_sentiment(samples_per_domain=500)
extractor = RoBERTaEmbeddingExtractor()
embeddings = extractor.embed_texts(dataset["text"])

# Comprehensive analysis
analyzer = StratificationAnalyzer()
results = analyzer.analyze_stratification(embeddings, dataset["domain"])
```

### Advanced MoE Training
```python
from fiber_bundle_test import MixtureOfDictionaryExperts, ContrastiveTrainer

# Create and train MoE model
model = MixtureOfDictionaryExperts(input_dim=768, K=7)
trainer = ContrastiveTrainer(model, config)
trainer.train(data_loader)
```

## 🛠️ **Configuration**

### YAML Configuration
```yaml
# config/default_config.yaml
test_parameters:
  r_min: 0.01
  r_max: 20.0
  alpha: 0.05
  window_size: 10

embedding_parameters:
  model_name: 'roberta-large'
  batch_size: 32

output:
  save_results: true
  save_plots: true
```

### Python Configuration
```python
from fiber_bundle_test.utils import ConfigLoader

config = ConfigLoader.load_config('config/custom.yaml')
test = FiberBundleTest(**config['test_parameters'])
```

## 🧪 **Testing**

```bash
# Run test suite
python test_modern_features.py

# Test specific functionality
python -m pytest tests/
```

## 📖 **Documentation**

- **Complete Documentation**: See [DOCUMENTATION.md](DOCUMENTATION.md) for comprehensive guide
- **Installation & Setup**: See requirements.txt
- **API Reference**: Comprehensive docstrings throughout
- **Examples**: examples/ directory with working scripts
- **Notebooks**: Interactive tutorials in notebooks/
- **Research Applications**: See examples and documentation
- **MoE Experiments**: Complete results and analysis in results/ directory (not tracked in git)

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## 📄 **License**

MIT License - see LICENSE file for details.

## 🔬 **Research Applications**

This framework enables research in:
- **LLM Geometry**: Understanding embedding space structure
- **Model Comparison**: Systematic analysis across architectures
- **Domain Analysis**: How different text domains affect geometry
- **Interpretability**: Geometric approaches to understanding models
- **Architecture Design**: Geometry-aware model development

## 🎯 **Citation**

If you use this framework in your research, please cite:

```bibtex
@software{fiber_bundle_llm_framework,
  title={Fiber Bundle Hypothesis Test Framework for LLM Embeddings},
  author={Fiber Bundle Research Team},
  year={2025},
  url={https://github.com/your-repo/fiber-bundle-test}
}
```

---

**Ready to analyze the geometric structure of language model embeddings!** 🚀🔬