"""
Main entry point for Stratified Manifold Learning experiments.

This script provides a command-line interface to run experiments with different
language models and analyze stratified manifold structures in their embeddings.
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import working experiment functions
from experiments.working.working_experiment import run_working_experiment
from experiments.advanced.advanced_experiment import run_advanced_experiment
from experiments.comparison.model_comparison import run_model_comparison
from experiments.curvature.curvature_enhanced_experiment import run_curvature_enhanced_experiment
from experiments.hypothesis.stratified_manifold_hypothesis_test import run_stratified_manifold_hypothesis_test
from experiments.deep.deep_stratified_manifold_analysis import run_deep_stratified_manifold_analysis
from experiments.fiber.comprehensive_fiber_bundle_analysis import run_comprehensive_fiber_bundle_analysis


def main():
    """Main entry point for the stratified manifold learning experiments."""
    parser = argparse.ArgumentParser(
        description="Run stratified manifold learning experiments with different language models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["working", "advanced", "comparison", "curvature", "hypothesis", "deep", "fiber", "all"],
        default="working",
        help="Experiment type to run"
    )
    
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=500,
        help="Number of samples per domain"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Margin for contrastive loss"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to files"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only create visualizations from existing results"
    )
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command line arguments
    config.update({
        'samples_per_domain': args.samples_per_domain,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'margin': args.margin,
        'output_dir': args.output_dir,
        'save_plots': args.save_plots
    })
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Determine which experiments to run
    if args.model == "all":
        experiments_to_run = ["working", "advanced", "comparison", "curvature", "hypothesis", "deep", "fiber"]
    else:
        experiments_to_run = [args.model]
    
    # Run experiments
    for experiment_name in experiments_to_run:
        print(f"\n{'='*60}")
        print(f"Running {experiment_name.upper()} experiment")
        print(f"{'='*60}")
        
        try:
            if experiment_name == "working":
                results = run_working_experiment(
                    samples_per_domain=config['samples_per_domain'],
                    num_clusters=5
                )
            elif experiment_name == "advanced":
                results = run_advanced_experiment(
                    samples_per_domain=config['samples_per_domain'],
                    num_clusters=5,
                    num_epochs=config['num_epochs']
                )
            elif experiment_name == "comparison":
                results = run_model_comparison(
                    samples_per_domain=config['samples_per_domain'],
                    num_clusters=4
                )
            elif experiment_name == "curvature":
                results = run_curvature_enhanced_experiment(
                    samples_per_domain=config['samples_per_domain'],
                    num_clusters=5,
                    num_epochs=config['num_epochs']
                )
            elif experiment_name == "hypothesis":
                results = run_stratified_manifold_hypothesis_test(
                    samples_per_domain=config['samples_per_domain'],
                    num_clusters=5,
                    num_epochs=config['num_epochs']
                )
            elif experiment_name == "deep":
                results = run_deep_stratified_manifold_analysis(
                    samples_per_domain=config['samples_per_domain'],
                    num_clusters=5,
                    num_epochs=config['num_epochs']
                )
            elif experiment_name == "fiber":
                results = run_comprehensive_fiber_bundle_analysis(
                    samples_per_domain=config['samples_per_domain'],
                    num_clusters=5,
                    num_epochs=config['num_epochs']
                )
            else:
                print(f"Experiment {experiment_name} not implemented")
                continue
            
            print(f"✅ {experiment_name.upper()} experiment completed successfully!")
            
        except Exception as e:
            print(f"❌ Error running {experiment_name} experiment: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {config['output_dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
