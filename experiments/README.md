# Experiments Directory

This directory contains all experimental scripts organized by analysis type.

## Structure

- **`working/`** - Basic working experiments
- **`advanced/`** - Advanced MoE experiments
- **`comparison/`** - Model comparison experiments
- **`curvature/`** - Curvature-based geometric analysis
- **`hypothesis/`** - Stratified manifold hypothesis testing
- **`deep/`** - Deep geometric analysis with Ricci curvature
- **`fiber/`** - Fiber bundle hypothesis testing (Robinson et al.)

## Usage

Run experiments using the main script:

```bash
python main.py --model [experiment_type] --samples-per-domain 100 --num-epochs 20
```

Available experiment types:
- `working` - Basic stratified manifold analysis
- `advanced` - Advanced MoE training
- `comparison` - Model comparison across LLMs
- `curvature` - Curvature-enhanced analysis
- `hypothesis` - Stratified manifold hypothesis test
- `deep` - Deep geometric analysis
- `fiber` - Fiber bundle hypothesis test
- `all` - Run all experiments

## Dependencies

All experiments require the packages listed in `requirements.txt` and the conda environment defined in `environment.yml`.