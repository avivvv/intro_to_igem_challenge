"""Feature extraction from DNA sequences."""

import numpy as np
from collections import Counter


def one_hot_encode(sequence, alphabet="ACGT"):
    """
    Convert DNA sequence to one-hot encoded matrix.
    
    Args:
        sequence: DNA sequence string
        alphabet: Valid nucleotides
        
    Returns:
        numpy array of shape (len(sequence), len(alphabet))
    """
    encoding = np.zeros((len(sequence), len(alphabet)))
    for i, nucleotide in enumerate(sequence.upper()):
        if nucleotide in alphabet:
            encoding[i, alphabet.index(nucleotide)] = 1
    return encoding


def extract_kmers(sequence, k=3):
    """
    Extract k-mers from a sequence.
    
    Args:
        sequence: DNA sequence string
        k: K-mer size
        
    Returns:
        Counter object with k-mer frequencies
    """
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)


def nucleotide_composition(sequence):
    """
    Calculate nucleotide composition features.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Dictionary with nucleotide frequencies
    """
    sequence = sequence.upper()
    total = len(sequence)
    return {
        'A_freq': sequence.count('A') / total,
        'C_freq': sequence.count('C') / total,
        'G_freq': sequence.count('G') / total,
        'T_freq': sequence.count('T') / total,
        'GC_content': (sequence.count('G') + sequence.count('C')) / total
    }


def dinucleotide_features(sequence):
    """
    Calculate dinucleotide frequencies.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        Dictionary with dinucleotide frequencies
    """
    sequence = sequence.upper()
    dinucleotides = [sequence[i:i+2] for i in range(len(sequence) - 1)]
    total = len(dinucleotides)
    
    features = {}
    for di in ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
               'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']:
        count = sum(1 for d in dinucleotides if d == di)
        features[f'{di}_freq'] = count / total if total > 0 else 0
    
    return features
