# Results Directory

This directory contains all experimental results organized by type.

## Structure

- **`images/`** - All visualization plots and figures
- **`data/`** - JSON results files from experiments
- **`analysis/`** - Detailed analysis reports (if any)

## Contents

### Images
- Experiment visualizations (PCA plots, clustering results, etc.)
- Training loss curves
- Comparison plots
- Geometric analysis visualizations

### Data
- JSON files with quantitative results
- Metrics and statistics
- Model performance data
- Geometric analysis results

## Usage

Results are automatically saved here when running experiments. Each experiment type creates specific output files:

- `*_experiment.png` - Main experiment visualizations
- `*_results.json` - Quantitative results
- `*_training_loss.png` - Training curves
- `*_analysis.png` - Detailed analysis plots

## File Naming Convention

Files are named according to the experiment type:
- `working_*` - Basic experiments
- `advanced_*` - Advanced MoE experiments
- `comparison_*` - Model comparison
- `curvature_*` - Curvature analysis
- `hypothesis_*` - Hypothesis testing
- `deep_*` - Deep geometric analysis
- `fiber_*` - Fiber bundle analysis