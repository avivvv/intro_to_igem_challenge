"""Unit tests for the promoter expression prediction project."""

import unittest
import numpy as np
import sys
sys.path.insert(0, '..')

from src.features.sequence_features import (
    nucleotide_composition, dinucleotide_features, one_hot_encode
)


class TestSequenceFeatures(unittest.TestCase):
    """Test sequence feature extraction functions."""
    
    def setUp(self):
        """Set up test sequences."""
        self.seq_simple = "ACGTACGT"
        self.seq_gc_rich = "GCGCGCGC"
        self.seq_at_rich = "ATATAT"
    
    def test_nucleotide_composition(self):
        """Test nucleotide composition calculation."""
        features = nucleotide_composition(self.seq_simple)
        
        # Check all keys present
        expected_keys = ['A_freq', 'C_freq', 'G_freq', 'T_freq', 'GC_content']
        for key in expected_keys:
            self.assertIn(key, features)
        
        # Check sum to 1
        freq_sum = features['A_freq'] + features['C_freq'] + features['G_freq'] + features['T_freq']
        self.assertAlmostEqual(freq_sum, 1.0)
    
    def test_gc_content(self):
        """Test GC content calculation."""
        features_gc = nucleotide_composition(self.seq_gc_rich)
        features_at = nucleotide_composition(self.seq_at_rich)
        
        # GC-rich should have higher GC content
        self.assertGreater(features_gc['GC_content'], features_at['GC_content'])
    
    def test_dinucleotide_features(self):
        """Test dinucleotide feature extraction."""
        features = dinucleotide_features(self.seq_simple)
        
        # Check all dinucleotides present
        expected_di = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
                       'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
        for di in expected_di:
            self.assertIn(f'{di}_freq', features)
        
        # Check sum to 1
        freq_sum = sum(features.values())
        self.assertAlmostEqual(freq_sum, 1.0)
    
    def test_one_hot_encode(self):
        """Test one-hot encoding."""
        encoded = one_hot_encode(self.seq_simple)
        
        # Check shape
        self.assertEqual(encoded.shape[0], len(self.seq_simple))
        self.assertEqual(encoded.shape[1], 4)  # 4 nucleotides
        
        # Check each row sums to 1
        for i in range(encoded.shape[0]):
            self.assertAlmostEqual(encoded[i].sum(), 1.0)


if __name__ == '__main__':
    unittest.main()
