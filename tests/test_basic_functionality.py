"""Basic functionality tests for the fiber bundle test package."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fiber_bundle_test.core.distance_analysis import DistanceAnalyzer
from fiber_bundle_test.core.slope_detection import SlopeChangeDetector
from fiber_bundle_test.utils.data_utils import DataUtils


class TestDistanceAnalysis(unittest.TestCase):
    """Test distance analysis functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.embeddings = np.random.randn(10, 5)
        self.analyzer = DistanceAnalyzer()
    
    def test_compute_distances(self):
        """Test distance computation."""
        distances = self.analyzer.compute_distances(self.embeddings, 0)
        
        # Check properties
        self.assertEqual(len(distances), len(self.embeddings))
        self.assertEqual(distances[0], 0.0)  # Distance to self should be 0
        self.assertTrue(np.all(distances[1:] > 0))  # All other distances should be positive
        self.assertTrue(np.all(distances[:-1] <= distances[1:]))  # Should be sorted
    
    def test_compute_nx_r(self):
        """Test neighbor counting."""
        distances = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        r_values = np.array([0.5, 1.5, 2.5, 3.5])
        
        nx_r = self.analyzer.compute_nx_r(distances, r_values)
        expected = np.array([1, 2, 3, 4])  # Number of points within each radius
        
        np.testing.assert_array_equal(nx_r, expected)
    
    def test_prepare_log_data(self):
        """Test logarithmic data preparation."""
        r_values = np.array([1.0, 2.0, 3.0])
        nx_r = np.array([1, 4, 9])
        
        log_r, log_nx_r = self.analyzer.prepare_log_data(r_values, nx_r)
        
        np.testing.assert_array_almost_equal(log_r, np.log(r_values))
        np.testing.assert_array_almost_equal(log_nx_r, np.log(nx_r))


class TestSlopeDetection(unittest.TestCase):
    """Test slope detection functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.detector = SlopeChangeDetector()
    
    def test_estimate_slopes(self):
        """Test slope estimation."""
        log_r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        log_nx_r = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # Quadratic relationship
        
        slopes = self.detector.estimate_slopes(log_r, log_nx_r)
        
        # Check that we get reasonable slopes
        self.assertEqual(len(slopes), len(log_r))
        self.assertTrue(np.all(np.isfinite(slopes)))
    
    def test_detect_slope_changes_no_change(self):
        """Test change detection with constant slopes."""
        slopes = np.ones(20)  # Constant slopes
        
        changes, p_values = self.detector.detect_slope_changes(slopes, window_size=5)
        
        # Should detect no changes
        self.assertEqual(len(changes), 0)
        self.assertEqual(len(p_values), 0)
    
    def test_estimate_dimensions(self):
        """Test dimension estimation."""
        slopes = np.concatenate([np.ones(10), 3*np.ones(10)])  # Step change
        change_idx = 10
        
        base_dim, fiber_dim = self.detector.estimate_dimensions(slopes, change_idx, window_size=5)
        
        self.assertAlmostEqual(base_dim, 1.0, places=5)
        self.assertAlmostEqual(fiber_dim, 3.0, places=5)


class TestDataUtils(unittest.TestCase):
    """Test data utility functions."""
    
    def test_create_sample_random_embeddings(self):
        """Test random embedding generation."""
        embeddings = DataUtils.create_sample_random_embeddings(
            n_tokens=50, embedding_dim=128, seed=42
        )
        
        self.assertEqual(embeddings.shape, (50, 128))
        self.assertTrue(np.isfinite(embeddings).all())
    
    def test_extract_token_labels(self):
        """Test token label extraction."""
        sentences = ["The bank is open.", "The river flows."]
        target_tokens = ["bank", "river"]
        
        labels = DataUtils.extract_token_labels(sentences, target_tokens)
        
        self.assertEqual(len(labels), 2)
        self.assertIn("bank", labels[0])
        self.assertIn("river", labels[1])
    
    def test_print_summary_statistics(self):
        """Test statistics printing (should not raise errors)."""
        embeddings = np.random.randn(10, 5)
        
        # This should not raise any exceptions
        try:
            DataUtils.print_summary_statistics(embeddings)
        except Exception as e:
            self.fail(f"print_summary_statistics raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
