"""
Unit tests for CounterFactualModel feature name matching functionality.

This module tests that feature names with different formats (with/without units,
different cases, whitespace variations) are properly matched when comparing
constraints against sample features.
"""

import unittest
import sys
import os

# Add parent directory to path to import CounterFactualModel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CounterFactualModel import CounterFactualModel


class TestFeatureNameMatching(unittest.TestCase):
    """Test cases for feature name normalization and matching."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal CounterFactualModel instance for testing
        # We only need the feature matching methods, so we can use None for model and constraints
        self.cf_model = CounterFactualModel(model=None, constraints={})
    
    def test_exact_match(self):
        """Test that identical feature names match."""
        self.assertTrue(
            self.cf_model._features_match('sepal width (cm)', 'sepal width (cm)')
        )
    
    def test_case_insensitive_match(self):
        """Test that feature names match regardless of case."""
        self.assertTrue(
            self.cf_model._features_match('Sepal Width (cm)', 'sepal width (cm)')
        )
        self.assertTrue(
            self.cf_model._features_match('SEPAL WIDTH (CM)', 'sepal width (cm)')
        )
    
    def test_whitespace_normalization(self):
        """Test that extra whitespace is handled correctly."""
        self.assertTrue(
            self.cf_model._features_match(' sepal width (cm) ', 'sepal width (cm)')
        )
        self.assertTrue(
            self.cf_model._features_match('sepal  width (cm)', 'sepal width (cm)')
        )
    
    def test_unit_stripping(self):
        """Test that units in parentheses are stripped for matching."""
        # This is the main fix - features with and without units should match
        self.assertTrue(
            self.cf_model._features_match('sepal width', 'sepal width (cm)')
        )
        self.assertTrue(
            self.cf_model._features_match('petal length', 'petal length (cm)')
        )
        self.assertTrue(
            self.cf_model._features_match('sepal width (cm)', 'sepal width')
        )
    
    def test_different_units_match(self):
        """Test that features with different units in parentheses still match."""
        self.assertTrue(
            self.cf_model._features_match('sepal width (cm)', 'sepal width (mm)')
        )
        self.assertTrue(
            self.cf_model._features_match('age (years)', 'age (months)')
        )
    
    def test_different_features_dont_match(self):
        """Test that genuinely different features don't match."""
        self.assertFalse(
            self.cf_model._features_match('sepal width (cm)', 'petal width (cm)')
        )
        self.assertFalse(
            self.cf_model._features_match('sepal length', 'petal length')
        )
    
    def test_normalize_feature_name(self):
        """Test the normalization function directly."""
        # Test unit removal
        self.assertEqual(
            self.cf_model._normalize_feature_name('sepal width (cm)'),
            'sepal width'
        )
        
        # Test case conversion
        self.assertEqual(
            self.cf_model._normalize_feature_name('Sepal Width'),
            'sepal width'
        )
        
        # Test whitespace stripping
        self.assertEqual(
            self.cf_model._normalize_feature_name('  sepal width  '),
            'sepal width'
        )
        
        # Test combined normalization
        self.assertEqual(
            self.cf_model._normalize_feature_name('  Sepal Width (CM)  '),
            'sepal width'
        )
    
    def test_complex_parentheses(self):
        """Test handling of complex content in parentheses."""
        self.assertTrue(
            self.cf_model._features_match(
                'temperature (degrees celsius)', 
                'temperature (fahrenheit)'
            )
        )
        self.assertTrue(
            self.cf_model._features_match(
                'speed (km/h)', 
                'speed'
            )
        )


class TestFeatureMatchingIntegration(unittest.TestCase):
    """Integration tests for feature matching in constraint validation."""
    
    def setUp(self):
        """Set up test fixtures with mock model and constraints."""
        # Mock model (just needs a predict method)
        class MockModel:
            def predict(self, X):
                return [0]
        
        # Constraints with feature names WITHOUT units
        self.constraints = {
            'Class 0': [
                {'feature': 'sepal width', 'operator': '>', 'value': 2.5},
                {'feature': 'petal length', 'operator': '<', 'value': 2.0}
            ],
            'Class 1': [
                {'feature': 'sepal width', 'operator': '<=', 'value': 3.0}
            ]
        }
        
        self.cf_model = CounterFactualModel(
            model=MockModel(),
            constraints=self.constraints
        )
    
    def test_validate_constraints_with_unit_mismatch(self):
        """Test that constraints work when sample has units but constraints don't."""
        # Sample WITH units
        sample = {
            'sepal width (cm)': 2.0,
            'petal length (cm)': 1.5
        }
        
        # Modified sample WITH units
        s_prime = {
            'sepal width (cm)': 2.8,
            'petal length (cm)': 1.5
        }
        
        # Should validate successfully even though constraint has 'sepal width'
        # and sample has 'sepal width (cm)'
        valid, penalty = self.cf_model.validate_constraints(s_prime, sample, 0)
        
        # The value 2.8 satisfies 'sepal width > 2.5', so should be valid
        self.assertTrue(valid)
        self.assertEqual(penalty, 0.0)
    
    def test_get_valid_sample_with_unit_mismatch(self):
        """Test that get_valid_sample works when sample has units but constraints don't."""
        # Sample WITH units
        sample = {
            'sepal width (cm)': 2.0,
            'petal length (cm)': 1.5
        }
        
        # Generate a valid sample for Class 0
        valid_sample = self.cf_model.get_valid_sample(sample, 0)
        
        # Check that the sample keys are preserved (with units)
        self.assertIn('sepal width (cm)', valid_sample)
        self.assertIn('petal length (cm)', valid_sample)
        
        # Validate the generated sample meets constraints
        # For Class 0: sepal width > 2.5, petal length < 2.0
        self.assertGreater(valid_sample['sepal width (cm)'], 2.5)
        self.assertLess(valid_sample['petal length (cm)'], 2.0)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureNameMatching))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureMatchingIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
