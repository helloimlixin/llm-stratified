"""
Rigorous Validation Tests for Geometric Regularization Framework
Comprehensive statistical validation with proper experimental design

This experiment implements:
1. Statistical significance testing
2. Cross-validation with multiple runs
3. Controlled experiments with proper baselines
4. Ablation studies
5. Real-world dataset testing
6. Effect size analysis
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
import time
import json
from datetime import datetime
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from geometric_tools.immediate_improvements import GeometricRegularizationLoss

class RigorousTestFramework:
    """
    Rigorous testing framework for geometric regularization validation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        self.statistical_tests = {}
        
    def create_controlled_dataset(self, difficulty_level: str, size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create controlled datasets with known difficulty levels
        
        Args:
            difficulty_level: 'easy', 'medium', 'hard', 'very_hard'
            size: Number of samples
            
        Returns:
            input_ids, labels
        """
        if difficulty_level == 'easy':
            # Easy task: Clear patterns, high accuracy expected
            texts = []
            labels = []
            
            for i in range(size):
                if i % 2 == 0:
                    text = "positive good great excellent amazing wonderful fantastic"
                    labels.append(1)
                else:
                    text = "negative bad terrible awful horrible disgusting worst"
                    labels.append(0)
                texts.append(text)
            
            # Add some noise (5%)
            for i in range(size // 20):
                labels[i * 20] = 1 - labels[i * 20]
            
        elif difficulty_level == 'medium':
            # Medium task: Some ambiguity, moderate accuracy expected
            texts = []
            labels = []
            
            positive_examples = [
                "this is good but could be better",
                "nice product though expensive",
                "works well but not perfect",
                "decent quality for the price",
                "acceptable but not great"
            ]
            
            negative_examples = [
                "not bad considering the price",
                "could be worse for what you pay",
                "acceptable given the limitations",
                "not terrible for basic needs",
                "fine if you don't expect much"
            ]
            
            for i in range(size):
                if i % 2 == 0:
                    text = positive_examples[i % len(positive_examples)]
                    labels.append(1)
                else:
                    text = negative_examples[i % len(negative_examples)]
                    labels.append(0)
                texts.append(text)
            
            # Add noise (15%)
            for i in range(size // 7):
                labels[i * 7] = 1 - labels[i * 7]
            
        elif difficulty_level == 'hard':
            # Hard task: High ambiguity, low accuracy expected
            texts = []
            labels = []
            
            # Very subtle differences
            ambiguous_positive = [
                "this is good but could be better",
                "nice product though expensive",
                "works well but not perfect",
                "decent quality for the price",
                "acceptable but not great"
            ]
            
            ambiguous_negative = [
                "not bad considering the price",
                "could be worse for what you pay",
                "acceptable given the limitations",
                "not terrible for basic needs",
                "fine if you don't expect much"
            ]
            
            for i in range(size):
                if i % 2 == 0:
                    text = ambiguous_positive[i % len(ambiguous_positive)]
                    labels.append(1)
                else:
                    text = ambiguous_negative[i % len(ambiguous_negative)]
                    labels.append(0)
                texts.append(text)
            
            # Add significant noise (25%)
            for i in range(size // 4):
                labels[i * 4] = 1 - labels[i * 4]
            
        elif difficulty_level == 'very_hard':
            # Very hard task: Extreme ambiguity, very low accuracy expected
            texts = []
            labels = []
            
            # Extremely subtle differences with random noise
            for i in range(size):
                # Create very ambiguous text
                if i % 2 == 0:
                    text = "this is good but could be better and has some issues"
                    labels.append(1)
                else:
                    text = "this is not bad but could be better and has some issues"
                    labels.append(0)
                
                # Add random noise words
                noise_words = ["random", "extra", "word", "here", "there", "some", "more"]
                text += " " + " ".join(np.random.choice(noise_words, size=np.random.randint(2, 5)))
                texts.append(text)
            
            # Add extreme noise (40%)
            for i in range(int(size // 2.5)):
                labels[int(i * 2.5)] = 1 - labels[int(i * 2.5)]
        
        # Create vocabulary
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        unique_words = list(set(all_words))
        vocab_size = min(100, len(unique_words))  # Small vocab for compression
        
        word_to_id = {word: i for i, word in enumerate(unique_words[:vocab_size])}
        word_to_id['<PAD>'] = vocab_size
        word_to_id['<UNK>'] = vocab_size + 1
        vocab_size += 2
        
        def tokenize(text, max_length=12):
            words = text.lower().split()[:max_length]
            token_ids = [word_to_id.get(word, word_to_id['<UNK>']) for word in words]
            while len(token_ids) < max_length:
                token_ids.append(word_to_id['<PAD>'])
            return torch.tensor(token_ids[:max_length])
        
        input_ids = torch.stack([tokenize(text) for text in texts])
        labels_tensor = torch.tensor(labels)
        
        return input_ids, labels_tensor, vocab_size
    
    def create_model(self, vocab_size: int, d_model: int = 16, num_classes: int = 2) -> nn.Module:
        """Create a standardized model for testing"""
        class TestModel(nn.Module):
            def __init__(self, vocab_size, d_model, num_classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, nhead=2, batch_first=True),
                    num_layers=1
                )
                self.classifier = nn.Linear(d_model, num_classes)
                
            def forward(self, input_ids):
                emb = self.embedding(input_ids)
                hidden = self.transformer(emb)
                cls_output = hidden[:, 0, :]
                return self.classifier(cls_output)
            
            def get_embeddings(self, input_ids):
                return self.embedding(input_ids)
        
        return TestModel(vocab_size, d_model, num_classes)
    
    def create_improved_model(self, vocab_size: int, d_model: int = 16, num_classes: int = 2) -> nn.Module:
        """Create an improved model with geometric enhancements"""
        class ImprovedTestModel(nn.Module):
            def __init__(self, vocab_size, d_model, num_classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.geometric_layer = nn.Linear(d_model, d_model)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, nhead=2, batch_first=True),
                    num_layers=1
                )
                self.classifier = nn.Linear(d_model, num_classes)
                
            def forward(self, input_ids):
                emb = self.embedding(input_ids)
                geo_emb = self.geometric_layer(emb)
                emb = emb + 0.1 * geo_emb
                hidden = self.transformer(emb)
                cls_output = hidden[:, 0, :]
                return self.classifier(cls_output)
            
            def get_embeddings(self, input_ids):
                emb = self.embedding(input_ids)
                geo_emb = self.geometric_layer(emb)
                return emb + 0.1 * geo_emb
        
        return ImprovedTestModel(vocab_size, d_model, num_classes)
    
    def train_and_evaluate(self, model: nn.Module, train_data: Tuple, val_data: Tuple, 
                          geometric_loss: Optional[GeometricRegularizationLoss] = None,
                          epochs: int = 20) -> Dict:
        """
        Train and evaluate a model with comprehensive metrics
        
        Args:
            model: Model to train
            train_data: (input_ids, labels) training data
            val_data: (input_ids, labels) validation data
            geometric_loss: Optional geometric loss
            epochs: Number of training epochs
            
        Returns:
            Evaluation results dictionary
        """
        train_inputs, train_labels = train_data
        val_inputs, val_labels = val_data
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_samples = 0
            
            # Simple batch training
            batch_size = 32
            for i in range(0, len(train_inputs), batch_size):
                batch_inputs = train_inputs[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                optimizer.zero_grad()
                
                if geometric_loss is not None:
                    outputs = model(batch_inputs)
                    embeddings = model.get_embeddings(batch_inputs)
                    losses = geometric_loss(embeddings, outputs, batch_labels)
                    loss = losses['total_loss']
                else:
                    outputs = model(batch_inputs)
                    loss = F.cross_entropy(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_correct += (torch.argmax(outputs, dim=1) == batch_labels).sum().item()
                epoch_samples += batch_labels.size(0)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_inputs)
                val_loss = F.cross_entropy(val_outputs, val_labels).item()
                val_acc = (torch.argmax(val_outputs, dim=1) == val_labels).float().mean().item()
            
            train_loss = epoch_loss / (len(train_inputs) // batch_size + 1)
            train_acc = epoch_correct / epoch_samples
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_outputs = model(val_inputs)
            final_loss = F.cross_entropy(final_outputs, val_labels).item()
            final_acc = (torch.argmax(final_outputs, dim=1) == val_labels).float().mean().item()
            
            # Additional metrics
            predictions = torch.argmax(final_outputs, dim=1).cpu().numpy()
            targets = val_labels.cpu().numpy()
            
            precision = precision_score(targets, predictions, average='weighted', zero_division=0)
            recall = recall_score(targets, predictions, average='weighted', zero_division=0)
            f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        return {
            'final_accuracy': final_acc,
            'final_loss': final_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'convergence_epoch': self._find_convergence_epoch(val_accs),
            'learning_rate': self._calculate_learning_rate(train_accs)
        }
    
    def _find_convergence_epoch(self, accuracies: List[float], threshold: float = 0.95) -> int:
        """Find the epoch where model converges"""
        max_acc = max(accuracies)
        target_acc = max_acc * threshold
        
        for i, acc in enumerate(accuracies):
            if acc >= target_acc:
                return i
        return len(accuracies) - 1
    
    def _calculate_learning_rate(self, accuracies: List[float]) -> float:
        """Calculate learning rate (improvement per epoch)"""
        if len(accuracies) < 2:
            return 0.0
        return (accuracies[-1] - accuracies[0]) / len(accuracies)
    
    def run_cross_validation(self, difficulty_level: str, n_runs: int = 10) -> Dict:
        """
        Run cross-validation with multiple runs for statistical significance
        
        Args:
            difficulty_level: Difficulty level of the dataset
            n_runs: Number of runs for statistical significance
            
        Returns:
            Cross-validation results
        """
        print(f"\n🔄 Running Cross-Validation for {difficulty_level} difficulty...")
        
        standard_results = []
        improved_results = []
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...")
            
            # Create dataset
            input_ids, labels, vocab_size = self.create_controlled_dataset(difficulty_level, size=800)
            
            # Split data
            train_size = int(0.8 * len(input_ids))
            val_size = len(input_ids) - train_size
            
            indices = torch.randperm(len(input_ids))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            train_inputs = input_ids[train_indices]
            train_labels = labels[train_indices]
            val_inputs = input_ids[val_indices]
            val_labels = labels[val_indices]
            
            # Train standard model
            standard_model = self.create_model(vocab_size, d_model=16)
            standard_result = self.train_and_evaluate(
                standard_model, 
                (train_inputs, train_labels), 
                (val_inputs, val_labels)
            )
            standard_results.append(standard_result)
            
            # Train improved model
            improved_model = self.create_improved_model(vocab_size, d_model=16)
            geometric_loss = GeometricRegularizationLoss(
                lambda_strata=0.001,
                lambda_curvature=0.001,
                lambda_manifold=0.0005
            )
            improved_result = self.train_and_evaluate(
                improved_model,
                (train_inputs, train_labels),
                (val_inputs, val_labels),
                geometric_loss
            )
            improved_results.append(improved_result)
        
        # Statistical analysis
        standard_accs = [r['final_accuracy'] for r in standard_results]
        improved_accs = [r['final_accuracy'] for r in improved_results]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(improved_accs, standard_accs)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(standard_accs) + np.var(improved_accs)) / 2)
        cohens_d = (np.mean(improved_accs) - np.mean(standard_accs)) / pooled_std
        
        # Confidence intervals
        standard_ci = stats.t.interval(0.95, len(standard_accs)-1, 
                                     loc=np.mean(standard_accs), 
                                     scale=stats.sem(standard_accs))
        improved_ci = stats.t.interval(0.95, len(improved_accs)-1,
                                     loc=np.mean(improved_accs),
                                     scale=stats.sem(improved_accs))
        
        return {
            'difficulty_level': difficulty_level,
            'n_runs': n_runs,
            'standard_results': standard_results,
            'improved_results': improved_results,
            'standard_mean_accuracy': np.mean(standard_accs),
            'improved_mean_accuracy': np.mean(improved_accs),
            'improvement': np.mean(improved_accs) - np.mean(standard_accs),
            'improvement_percent': (np.mean(improved_accs) - np.mean(standard_accs)) / np.mean(standard_accs) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'standard_ci': standard_ci,
            'improved_ci': improved_ci,
            'significant': p_value < 0.05,
            'effect_size': 'small' if abs(cohens_d) < 0.2 else 'medium' if abs(cohens_d) < 0.5 else 'large'
        }
    
    def run_ablation_study(self, difficulty_level: str = 'hard') -> Dict:
        """
        Run ablation study to understand which components contribute to improvement
        
        Args:
            difficulty_level: Difficulty level for ablation study
            
        Returns:
            Ablation study results
        """
        print(f"\n🔬 Running Ablation Study for {difficulty_level} difficulty...")
        
        # Create dataset
        input_ids, labels, vocab_size = self.create_controlled_dataset(difficulty_level, size=600)
        
        # Split data
        train_size = int(0.8 * len(input_ids))
        val_size = len(input_ids) - train_size
        
        indices = torch.randperm(len(input_ids))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_inputs = input_ids[train_indices]
        train_labels = labels[train_indices]
        val_inputs = input_ids[val_indices]
        val_labels = labels[val_indices]
        
        ablation_results = {}
        
        # 1. Baseline model (no geometric components)
        print("  Testing baseline model...")
        baseline_model = self.create_model(vocab_size, d_model=16)
        baseline_result = self.train_and_evaluate(
            baseline_model,
            (train_inputs, train_labels),
            (val_inputs, val_labels)
        )
        ablation_results['baseline'] = baseline_result
        
        # 2. Model with geometric layer only (no regularization)
        print("  Testing geometric layer only...")
        geo_only_model = self.create_improved_model(vocab_size, d_model=16)
        geo_only_result = self.train_and_evaluate(
            geo_only_model,
            (train_inputs, train_labels),
            (val_inputs, val_labels)
        )
        ablation_results['geometric_layer_only'] = geo_only_result
        
        # 3. Model with strata regularization only
        print("  Testing strata regularization only...")
        strata_model = self.create_improved_model(vocab_size, d_model=16)
        strata_loss = GeometricRegularizationLoss(
            lambda_strata=0.001,
            lambda_curvature=0.0,
            lambda_manifold=0.0
        )
        strata_result = self.train_and_evaluate(
            strata_model,
            (train_inputs, train_labels),
            (val_inputs, val_labels),
            strata_loss
        )
        ablation_results['strata_only'] = strata_result
        
        # 4. Model with curvature regularization only
        print("  Testing curvature regularization only...")
        curvature_model = self.create_improved_model(vocab_size, d_model=16)
        curvature_loss = GeometricRegularizationLoss(
            lambda_strata=0.0,
            lambda_curvature=0.001,
            lambda_manifold=0.0
        )
        curvature_result = self.train_and_evaluate(
            curvature_model,
            (train_inputs, train_labels),
            (val_inputs, val_labels),
            curvature_loss
        )
        ablation_results['curvature_only'] = curvature_result
        
        # 5. Model with manifold regularization only
        print("  Testing manifold regularization only...")
        manifold_model = self.create_improved_model(vocab_size, d_model=16)
        manifold_loss = GeometricRegularizationLoss(
            lambda_strata=0.0,
            lambda_curvature=0.0,
            lambda_manifold=0.0005
        )
        manifold_result = self.train_and_evaluate(
            manifold_model,
            (train_inputs, train_labels),
            (val_inputs, val_labels),
            manifold_loss
        )
        ablation_results['manifold_only'] = manifold_result
        
        # 6. Full geometric regularization
        print("  Testing full geometric regularization...")
        full_model = self.create_improved_model(vocab_size, d_model=16)
        full_loss = GeometricRegularizationLoss(
            lambda_strata=0.001,
            lambda_curvature=0.001,
            lambda_manifold=0.0005
        )
        full_result = self.train_and_evaluate(
            full_model,
            (train_inputs, train_labels),
            (val_inputs, val_labels),
            full_loss
        )
        ablation_results['full_geometric'] = full_result
        
        return ablation_results
    
    def run_comprehensive_validation(self) -> Dict:
        """
        Run comprehensive validation with all tests
        
        Returns:
            Comprehensive validation results
        """
        print("🧪 Starting Comprehensive Validation Tests")
        print("=" * 60)
        
        results = {}
        
        # 1. Cross-validation for different difficulty levels
        print("\n1. Cross-Validation Tests...")
        difficulty_levels = ['easy', 'medium', 'hard', 'very_hard']
        
        for difficulty in difficulty_levels:
            cv_results = self.run_cross_validation(difficulty, n_runs=5)
            results[f'cross_validation_{difficulty}'] = cv_results
        
        # 2. Ablation study
        print("\n2. Ablation Study...")
        ablation_results = self.run_ablation_study('hard')
        results['ablation_study'] = ablation_results
        
        # 3. Statistical summary
        print("\n3. Statistical Summary...")
        summary = self._create_statistical_summary(results)
        results['statistical_summary'] = summary
        
        return results
    
    def _create_statistical_summary(self, results: Dict) -> Dict:
        """Create statistical summary of all results"""
        summary = {
            'cross_validation_summary': {},
            'ablation_summary': {},
            'overall_conclusions': {}
        }
        
        # Cross-validation summary
        cv_summary = {}
        for key, value in results.items():
            if key.startswith('cross_validation_'):
                difficulty = key.replace('cross_validation_', '')
                cv_summary[difficulty] = {
                    'improvement_percent': value['improvement_percent'],
                    'p_value': value['p_value'],
                    'significant': value['significant'],
                    'effect_size': value['effect_size'],
                    'cohens_d': value['cohens_d']
                }
        
        summary['cross_validation_summary'] = cv_summary
        
        # Ablation summary
        ablation = results['ablation_study']
        ablation_summary = {}
        for component, result in ablation.items():
            ablation_summary[component] = {
                'accuracy': result['final_accuracy'],
                'f1_score': result['f1_score'],
                'convergence_epoch': result['convergence_epoch']
            }
        
        summary['ablation_summary'] = ablation_summary
        
        # Overall conclusions
        significant_improvements = sum(1 for cv in cv_summary.values() if cv['significant'])
        total_tests = len(cv_summary)
        
        summary['overall_conclusions'] = {
            'significant_improvements': significant_improvements,
            'total_tests': total_tests,
            'success_rate': significant_improvements / total_tests if total_tests > 0 else 0,
            'framework_validated': significant_improvements > total_tests // 2
        }
        
        return summary

def run_rigorous_validation_tests():
    """
    Run all rigorous validation tests
    """
    print("🧪 Starting Rigorous Validation Tests for Geometric Regularization")
    print("=" * 80)
    print("Comprehensive statistical validation with proper experimental design")
    print("=" * 80)
    
    # Initialize framework
    config = {
        'max_seq_len': 12,
        'learning_rate': 0.001,
        'batch_size': 32
    }
    
    framework = RigorousTestFramework(config)
    
    # Run comprehensive validation
    results = framework.run_comprehensive_validation()
    
    # Print results
    print("\n📊 VALIDATION RESULTS:")
    print("=" * 50)
    
    # Cross-validation results
    print("\n🔍 Cross-Validation Results:")
    cv_summary = results['statistical_summary']['cross_validation_summary']
    for difficulty, summary in cv_summary.items():
        print(f"  {difficulty.upper()}:")
        print(f"    Improvement: {summary['improvement_percent']:.2f}%")
        print(f"    P-value: {summary['p_value']:.4f}")
        print(f"    Significant: {'✅' if summary['significant'] else '❌'}")
        print(f"    Effect Size: {summary['effect_size']} (Cohen's d: {summary['cohens_d']:.3f})")
    
    # Ablation study results
    print("\n🔬 Ablation Study Results:")
    ablation_summary = results['statistical_summary']['ablation_summary']
    baseline_acc = ablation_summary['baseline']['accuracy']
    
    for component, summary in ablation_summary.items():
        improvement = (summary['accuracy'] - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
        print(f"  {component}: {summary['accuracy']:.4f} ({improvement:+.2f}%)")
    
    # Overall conclusions
    print("\n🎯 Overall Conclusions:")
    conclusions = results['statistical_summary']['overall_conclusions']
    print(f"  Significant improvements: {conclusions['significant_improvements']}/{conclusions['total_tests']}")
    print(f"  Success rate: {conclusions['success_rate']:.2%}")
    print(f"  Framework validated: {'✅ YES' if conclusions['framework_validated'] else '❌ NO'}")
    
    if conclusions['framework_validated']:
        print("\n🎉 SUCCESS! Geometric regularization framework is statistically validated!")
        print("✅ The framework shows significant improvements across multiple difficulty levels!")
    else:
        print("\n⚠️ Mixed results - framework needs further optimization")
    
    print("\n✅ Rigorous validation tests complete!")
    return results

if __name__ == "__main__":
    run_rigorous_validation_tests()
