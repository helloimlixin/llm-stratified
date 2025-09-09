"""
Example script demonstrating how to use the stratified manifold learning package.

This script shows how to run experiments with different models and analyze results.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.roberta_model import run_roberta_experiment, visualize_roberta_results
from src.utils.data_utils import print_experiment_summary, save_results


def main():
    """Run a simple example experiment."""
    print("Running RoBERTa experiment example...")
    
    # Run experiment with smaller dataset for quick demo
    results = run_roberta_experiment(
        samples_per_domain=100,  # Small dataset for demo
        num_epochs=5,            # Few epochs for demo
        lr=1e-3,
        margin=1.0
    )
    
    # Print summary
    print_experiment_summary(results, "roberta")
    
    # Save results
    save_results(results, "roberta_demo", "demo_results")
    
    # Create visualizations
    visualize_roberta_results(results, save_plots=True)
    
    print("Demo completed! Check the 'demo_results' directory for outputs.")


if __name__ == "__main__":
    main()
