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
from experiments.robinson.simplified_robinson_analysis import run_robinson_comprehensive_analysis
from experiments.wang.comprehensive_wang_analysis import run_comprehensive_wang_analysis
from experiments.integrated.comprehensive_integration_analysis import run_comprehensive_multi_paper_analysis
from experiments.real_model.real_model_integration_analysis import run_real_model_analysis
from experiments.large_model.large_model_analysis import run_large_model_analysis
from experiments.immediate_improvements.immediate_improvements_experiment import run_immediate_improvements_experiment
from experiments.real_world_testing.real_world_testing_experiment import run_real_world_testing_experiment
from experiments.very_light_regularization.very_light_regularization_experiment import run_very_light_regularization_experiment
from experiments.ultra_minimal_regularization.ultra_minimal_regularization_experiment import run_ultra_minimal_regularization_experiment
from experiments.comprehensive_benefit_testing.comprehensive_benefit_testing_experiment import run_comprehensive_benefit_testing_experiment
from experiments.larger_models_regularization_scaling.larger_models_regularization_scaling_experiment import run_larger_models_regularization_scaling_experiment
from experiments.ultra_large_models.ultra_large_models_experiment import run_ultra_large_models_experiment
from experiments.properly_designed.properly_designed_experiment import run_properly_designed_experiment
from experiments.real_world_application.real_world_application_experiment import run_real_world_application_experiment
from experiments.hf_trainer_stratified.hf_trainer_stratified_experiment import run_hf_trainer_stratified_experiment
from experiments.modern_sota_models.modern_sota_models_experiment import run_modern_sota_models_experiment
from experiments.hf_trainer_modern_sota.hf_trainer_modern_sota_experiment import test_hf_trainer_modern_sota
from experiments.llama3_hf_trainer.llama3_hf_trainer_experiment import run_llama3_hf_trainer_experiment
from experiments.generation_benchmarks.generation_benchmarks_experiment import run_generation_benchmarks_experiment
from experiments.comprehensive_generation_training.comprehensive_generation_training_experiment import run_comprehensive_generation_training
from experiments.research_scale_generation.research_scale_generation_experiment import run_research_scale_generation_experiment
from experiments.from_scratch_training.from_scratch_training_experiment import run_from_scratch_training_experiment


def main():
    """Main entry point for the stratified manifold learning experiments."""
    parser = argparse.ArgumentParser(
        description="Run stratified manifold learning experiments with different language models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["working", "advanced", "comparison", "curvature", "hypothesis", "deep", "fiber", "robinson", "wang", "integrated", "real_model", "large_model", "immediate_improvements", "real_world_testing", "very_light_regularization", "ultra_minimal_regularization", "comprehensive_benefit_testing", "larger_models_regularization_scaling", "ultra_large_models", "properly_designed", "real_world_application", "hf_trainer", "modern_sota", "hf_modern_sota", "llama3_hf", "gen_benchmarks", "comprehensive_gen", "research_scale", "from_scratch", "all"],
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
        experiments_to_run = ["working", "advanced", "comparison", "curvature", "hypothesis", "deep", "fiber", "robinson", "wang", "integrated", "real_model", "large_model", "immediate_improvements", "real_world_testing", "very_light_regularization", "ultra_minimal_regularization", "comprehensive_benefit_testing", "larger_models_regularization_scaling", "ultra_large_models", "properly_designed", "real_world_application", "hf_trainer", "modern_sota", "hf_modern_sota", "llama3_hf", "gen_benchmarks", "comprehensive_gen", "research_scale", "from_scratch"]
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
            elif experiment_name == "robinson":
                results = run_robinson_comprehensive_analysis()
            elif experiment_name == "wang":
                results = run_comprehensive_wang_analysis()
            elif experiment_name == "integrated":
                results = run_comprehensive_multi_paper_analysis()
            elif experiment_name == "real_model":
                results = run_real_model_analysis()
            elif experiment_name == "large_model":
                results = run_large_model_analysis()
            elif experiment_name == "immediate_improvements":
                results = run_immediate_improvements_experiment()
            elif experiment_name == "real_world_testing":
                results = run_real_world_testing_experiment()
            elif experiment_name == "very_light_regularization":
                results = run_very_light_regularization_experiment()
            elif experiment_name == "ultra_minimal_regularization":
                results = run_ultra_minimal_regularization_experiment()
            elif experiment_name == "comprehensive_benefit_testing":
                results = run_comprehensive_benefit_testing_experiment()
            elif experiment_name == "larger_models_regularization_scaling":
                results = run_larger_models_regularization_scaling_experiment()
            elif experiment_name == "ultra_large_models":
                results = run_ultra_large_models_experiment()
            elif experiment_name == "properly_designed":
                results = run_properly_designed_experiment()
            elif experiment_name == "real_world_application":
                results = run_real_world_application_experiment()
            elif experiment_name == "hf_trainer":
                results = run_hf_trainer_stratified_experiment()
            elif experiment_name == "modern_sota":
                results = run_modern_sota_models_experiment()
            elif experiment_name == "hf_modern_sota":
                results = test_hf_trainer_modern_sota()
            elif experiment_name == "llama3_hf":
                results = run_llama3_hf_trainer_experiment()
            elif experiment_name == "gen_benchmarks":
                results = run_generation_benchmarks_experiment()
            elif experiment_name == "comprehensive_gen":
                results = run_comprehensive_generation_training()
            elif experiment_name == "research_scale":
                results = run_research_scale_generation_experiment()
            elif experiment_name == "from_scratch":
                results = run_from_scratch_training_experiment()
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
